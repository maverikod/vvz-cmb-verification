
#!/usr/bin/env python3
'''search_theory_index.py

Index manager and CLI helper to work with the theory index (ALL_index.yaml).

This script is designed to be:
- self-contained (only stdlib + PyYAML)
- robust in constrained environments
- convenient both for interactive use and as a helper in other tools.

Features
--------
Core:
- Transparent pickle cache for fast index loading.
- Search by tag/id substring, category substring and free-text phrase.
- Optional phrase search in the raw theory markdown file (e.g. All.md).
- Optional preview of text snippets for matched segments.
- Basic and extended statistics for the index.

Extra "quality-of-life" features:
- Validation of index structure (duplicates, bad ranges, etc).
- Reverse lookup by line number / line range in All.md.
- Category tree view.
- Export of filtered subsets of the index (YAML or JSON).
- Assembly of continuous text from All.md based on selected segments.

Requirements
------------
- Python 3.8+
- PyYAML (`pip install pyyaml`)

Usage examples
--------------
Search by tag and show snippets (assuming All.md is available):

    python search_theory_index.py --index ALL_index.yaml --theory All.md \
        --tag 7d-06 --show-text

Show basic stats for the index:

    python search_theory_index.py --index ALL_index.yaml --mode stats

Run extended stats and validation:

    python search_theory_index.py --index ALL_index.yaml --mode stats-extended
    python search_theory_index.py --index ALL_index.yaml --mode validate

Reverse lookup by line number (which segments cover line 12345 in All.md):

    python search_theory_index.py --index ALL_index.yaml --mode line --line 12345

Export subset of segments (by phrase) into YAML:

    python search_theory_index.py --index ALL_index.yaml \
        --phrase "электрон" --mode export --export-path subset.yaml

Assemble continuous markdown text from All.md based on selected segments:

    python search_theory_index.py --index ALL_index.yaml --theory All.md \
        --tag 7d-06 --mode assemble --output-path electron_chapter.md

This script intentionally avoids any heavy dependencies or non-portable tricks,
so it should work both in local environments and in constrained tools.
'''

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore[import]
except ImportError as exc:
    print("ERROR: PyYAML is required. Install it via:", file=sys.stderr)
    print("  pip install pyyaml", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Segment:
    '''Lightweight wrapper around a segment entry from the YAML index.'''

    id: str
    category: str
    keywords: List[str]
    summary: str
    start_line: int
    end_line: int
    ranges: List[Tuple[int, int]]
    raw: Dict[str, Any]

    @property
    def length(self) -> int:
        return max(0, self.end_line - self.start_line + 1)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Segment":
        seg_id = str(d.get("id", ""))
        category = str(d.get("category", ""))
        keywords = list(d.get("keywords") or [])
        summary = str(d.get("summary", ""))

        # Normalize ranges: either explicit "ranges" or a single [start_line, end_line]
        ranges_list: List[Tuple[int, int]] = []
        if "ranges" in d and isinstance(d["ranges"], list):
            for r in d["ranges"]:
                try:
                    s = int(r.get("start_line"))
                    e = int(r.get("end_line"))
                    ranges_list.append((s, e))
                except Exception:
                    continue
        else:
            try:
                s = int(d.get("start_line"))
                e = int(d.get("end_line"))
                ranges_list.append((s, e))
            except Exception:
                pass

        if not ranges_list:
            # Fallback: zero-length dummy
            ranges_list = [(int(d.get("start_line", 0) or 0), int(d.get("end_line", 0) or 0))]

        # Canonical aggregate start/end
        start_line = min(s for s, _ in ranges_list)
        end_line = max(e for _, e in ranges_list)

        return cls(
            id=seg_id,
            category=category,
            keywords=keywords,
            summary=summary,
            start_line=start_line,
            end_line=end_line,
            ranges=ranges_list,
            raw=d,
        )


@dataclasses.dataclass
class IndexData:
    segments: List[Segment]
    raw: Dict[str, Any]

    @property
    def total_segments(self) -> int:
        return len(self.segments)

    @property
    def line_span(self) -> Tuple[Optional[int], Optional[int]]:
        if not self.segments:
            return None, None
        min_line = min(seg.start_line for seg in self.segments)
        max_line = max(seg.end_line for seg in self.segments)
        return min_line, max_line


# ---------------------------------------------------------------------------
# Low-level index loading with pickle cache
# ---------------------------------------------------------------------------

def _get_cache_path(index_path: str) -> str:
    p = Path(index_path)
    return str(p.with_suffix(p.suffix + ".pkl"))


def _load_index_yaml(path: str) -> Dict[str, Any]:
    '''Load the YAML index file without any caching.'''
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Index YAML must contain a mapping at the top level.")
    return data


def load_index(path: str, use_cache: bool = True) -> IndexData:
    '''Load index with optional pickle-based caching.

    Cache layout::

        {
            "index_path": "/full/path/to/ALL_index.yaml",
            "mtime": <float>,
            "size": <int>,
            "data": <raw_yaml_dict>
        }
    '''
    path = str(Path(path).resolve())
    yaml_mtime = os.path.getmtime(path)
    yaml_size = os.path.getsize(path)
    cache_path = _get_cache_path(path)

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if (
                isinstance(payload, dict)
                and payload.get("index_path") == path
                and isinstance(payload.get("mtime"), (int, float))
                and isinstance(payload.get("size"), int)
                and payload.get("mtime") == yaml_mtime
                and payload.get("size") == yaml_size
                and isinstance(payload.get("data"), dict)
            ):
                raw = payload["data"]  # type: ignore[assignment]
                return _build_index(raw)
        except Exception as exc:  # pragma: no cover - best-effort cache
            print(f"WARNING: failed to read cache {cache_path}: {exc}", file=sys.stderr)

    # Fallback: load from YAML
    raw = _load_index_yaml(path)
    if use_cache:
        try:
            payload = {
                "index_path": path,
                "mtime": yaml_mtime,
                "size": yaml_size,
                "data": raw,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # pragma: no cover - best-effort cache
            print(f"WARNING: failed to write cache {cache_path}: {exc}", file=sys.stderr)

    return _build_index(raw)


def _build_index(raw: Dict[str, Any]) -> IndexData:
    segments_raw = raw.get("segments") or []
    if not isinstance(segments_raw, list):
        raise ValueError("Index YAML must contain 'segments' as a list.")

    segments: List[Segment] = []
    for d in segments_raw:
        if not isinstance(d, dict):
            continue
        try:
            segments.append(Segment.from_dict(d))
        except Exception as exc:
            print(f"WARNING: failed to parse segment {d!r}: {exc}", file=sys.stderr)

    return IndexData(segments=segments, raw=raw)


def load_theory_lines(path: str) -> List[str]:
    '''Load theory markdown file as a list of lines (1-based mapping).'''
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


# ---------------------------------------------------------------------------
# Matching / search helpers
# ---------------------------------------------------------------------------

def matches_tag(seg: Segment, tag_substr: Optional[str]) -> bool:
    if not tag_substr:
        return True
    return tag_substr.lower() in seg.id.lower()


def matches_category(seg: Segment, cat_substr: Optional[str]) -> bool:
    if not cat_substr:
        return True
    return cat_substr.lower() in seg.category.lower()


def _build_segment_text(seg: Segment, theory_lines: Optional[List[str]]) -> str:
    '''Aggregate text for phrase search: keywords, summary and (optionally) raw text.'''
    parts: List[str] = []
    if seg.keywords:
        parts.append(" ".join(seg.keywords))
    if seg.summary:
        parts.append(seg.summary)

    if theory_lines is not None:
        chunks: List[str] = []
        for s, e in seg.ranges:
            # Lines in YAML are 1-based; Python list is 0-based.
            s0 = max(0, s - 1)
            e0 = min(len(theory_lines), e)
            if s0 < e0:
                chunks.append("".join(theory_lines[s0:e0]))
        if chunks:
            parts.append("\n".join(chunks))

    return "\n".join(parts)


def matches_phrase(
    seg: Segment,
    phrase: Optional[str],
    theory_lines: Optional[List[str]],
    _cache: Dict[str, str],
) -> bool:
    if not phrase:
        return True
    key = seg.id
    text = _cache.get(key)
    if text is None:
        text = _build_segment_text(seg, theory_lines)
        _cache[key] = text
    return phrase.lower() in text.lower()


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_result_text(
    seg: Segment,
    theory_lines: Optional[List[str]] = None,
    show_text: bool = False,
    snippet_lines: int = 10,
) -> None:
    '''Pretty-print a single segment.'''
    print("=" * 80)
    print(f"id       : {seg.id}")
    print(f"category : {seg.category}")
    print(f"lines    : {seg.start_line} - {seg.end_line} (len={seg.length})")
    if seg.keywords:
        print("keywords : " + ", ".join(seg.keywords))
    if seg.summary:
        print("summary  : " + seg.summary.strip())

    if show_text and theory_lines is not None:
        print("-" * 80)
        first_start, _ = min(seg.ranges, key=lambda r: r[0])
        s0 = max(0, first_start - 1)
        e0 = min(len(theory_lines), s0 + snippet_lines)
        snippet = "".join(theory_lines[s0:e0])
        print(snippet.rstrip())
        print("-" * 80)


# ---------------------------------------------------------------------------
# Modes: stats, stats-extended, validate, tree, line/range, export, assemble, search
# ---------------------------------------------------------------------------

def mode_stats(index: IndexData) -> int:
    min_line, max_line = index.line_span
    print(f"Total segments : {index.total_segments}")
    print(f"Line span      : {min_line} - {max_line}")

    # Simple category counts
    counter = collections.Counter(seg.category for seg in index.segments)
    print("\nCategories:")
    for cat, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {cat}: {count}")
    return 0


def mode_stats_extended(index: IndexData) -> int:
    min_line, max_line = index.line_span
    print(f"Total segments : {index.total_segments}")
    print(f"Line span      : {min_line} - {max_line}")

    lengths = [seg.length for seg in index.segments if seg.length > 0]
    if lengths:
        print("\nSegment length stats (in lines):")
        print(f"  min   : {min(lengths)}")
        print(f"  max   : {max(lengths)}")
        print(f"  mean  : {statistics.mean(lengths):.1f}")
        print(f"  median: {statistics.median(lengths):.1f}")

    # Category stats
    cat_counter = collections.Counter(seg.category for seg in index.segments)
    print("\nCategories (count, total_len):")
    for cat in sorted(cat_counter.keys()):
        segs = [s for s in index.segments if s.category == cat]
        total_len = sum(s.length for s in segs)
        print(f"  {cat}: {len(segs)} segments, {total_len} lines")

    return 0


def mode_validate(index: IndexData) -> int:
    errors: List[str] = []

    # Duplicate ids
    id_counter = collections.Counter(seg.id for seg in index.segments)
    for seg_id, count in id_counter.items():
        if count > 1:
            errors.append(f"Duplicate id: {seg_id} (count={count})")

    # Check ranges
    for seg in index.segments:
        if seg.start_line > seg.end_line:
            errors.append(f"Bad range in {seg.id}: start_line > end_line")
        for s, e in seg.ranges:
            if s > e:
                errors.append(f"Bad sub-range in {seg.id}: {s} > {e}")

    if errors:
        print("VALIDATION ERRORS:")
        for msg in errors:
            print("  - " + msg)
        return 1

    print("Index validation: OK")
    return 0


def mode_tree(index: IndexData) -> int:
    '''Print simple category tree (flat, grouped by category).'''
    cat_to_ids: Dict[str, List[str]] = collections.defaultdict(list)
    for seg in index.segments:
        cat_to_ids[seg.category].append(seg.id)

    for cat in sorted(cat_to_ids.keys()):
        print(cat)
        for seg_id in sorted(cat_to_ids[cat]):
            print(f"  - {seg_id}")
        print()
    return 0


def _segments_overlapping_line(index: IndexData, line: int) -> List[Segment]:
    return [seg for seg in index.segments if seg.start_line <= line <= seg.end_line]


def _segments_overlapping_range(index: IndexData, lo: int, hi: int) -> List[Segment]:
    return [
        seg
        for seg in index.segments
        if not (seg.end_line < lo or seg.start_line > hi)
    ]


def mode_line(index: IndexData, line_no: int) -> int:
    segs = _segments_overlapping_line(index, line_no)
    if not segs:
        print(f"No segments cover line {line_no}.")
        return 0
    print(f"Segments covering line {line_no}:")
    for seg in sorted(segs, key=lambda s: (s.start_line, s.id)):
        print(f"  - {seg.id} ({seg.start_line}-{seg.end_line}) [{seg.category}]")
    return 0


def mode_range(index: IndexData, line_range: str) -> int:
    try:
        lo_str, hi_str = line_range.split("-", 1)
        lo = int(lo_str.strip())
        hi = int(hi_str.strip())
    except Exception:
        print(f"ERROR: invalid --range value: {line_range!r}, expected 'lo-hi'.", file=sys.stderr)
        return 2
    if lo > hi:
        lo, hi = hi, lo

    segs = _segments_overlapping_range(index, lo, hi)
    if not segs:
        print(f"No segments overlap range {lo}-{hi}.")
        return 0
    print(f"Segments overlapping range {lo}-{hi}:")
    for seg in sorted(segs, key=lambda s: (s.start_line, s.id)):
        print(f"  - {seg.id} ({seg.start_line}-{seg.end_line}) [{seg.category}]")
    return 0


def mode_export(filtered: List[Segment], export_path: str) -> int:
    if not export_path:
        print("ERROR: --export-path is required for mode=export.", file=sys.stderr)
        return 2

    # Prepare raw payload with the original dictionaries.
    out_segments = [seg.raw for seg in filtered]
    payload = {"segments": out_segments}
    ext = export_path.lower()

    try:
        if ext.endswith(".json"):
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        elif ext.endswith(".yaml") or ext.endswith(".yml"):
            with open(export_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        else:
            print(
                f"ERROR: unsupported export format for {export_path!r}, "
                "use .json or .yaml/.yml",
                file=sys.stderr,
            )
            return 2
    except Exception as exc:
        print(f"ERROR: failed to write export file {export_path!r}: {exc}", file=sys.stderr)
        return 1

    print(f"Exported {len(filtered)} segments to {export_path}")
    return 0


def mode_assemble(
    theory_lines: Optional[List[str]],
    filtered: List[Segment],
    output_path: str,
    add_headers: bool = True,
) -> int:
    '''Assemble continuous text from theory file based on selected segments.'''
    if not output_path:
        print("ERROR: --output-path is required for mode=assemble.", file=sys.stderr)
        return 2

    if theory_lines is None:
        print("ERROR: --theory is required for mode=assemble.", file=sys.stderr)
        return 2

    # Sort segments by their starting line
    segments_sorted = sorted(filtered, key=lambda s: (s.start_line, s.id))

    assembled_parts: List[str] = []

    for seg in segments_sorted:
        if add_headers:
            header = f"## [{seg.id}] {seg.category}".rstrip()
            assembled_parts.append(header + "\n\n")

        # Sort ranges within segment
        for s, e in sorted(seg.ranges, key=lambda r: r[0]):
            s0 = max(0, s - 1)
            e0 = min(len(theory_lines), e)
            if s0 < e0:
                assembled_parts.append("".join(theory_lines[s0:e0]))
            assembled_parts.append("\n")

        assembled_parts.append("\n")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("".join(assembled_parts))
    except Exception as exc:
        print(f"ERROR: failed to write assembled file {output_path!r}: {exc}", file=sys.stderr)
        return 1

    print(f"Assembled {len(segments_sorted)} segments into {output_path}")
    return 0


def mode_search(
    index: IndexData,
    theory_lines: Optional[List[str]],
    tag: Optional[str],
    category: Optional[str],
    phrase: Optional[str],
    show_text: bool,
    limit: int,
    sort_by: Optional[str],
    json_output: bool,
) -> int:
    cache_text: Dict[str, str] = {}
    results: List[Segment] = []

    for seg in index.segments:
        if not matches_tag(seg, tag):
            continue
        if not matches_category(seg, category):
            continue
        if not matches_phrase(seg, phrase, theory_lines, cache_text):
            continue
        results.append(seg)

    # Sorting
    if sort_by == "id":
        results.sort(key=lambda s: s.id)
    elif sort_by == "category":
        results.sort(key=lambda s: (s.category, s.id))
    elif sort_by == "start_line":
        results.sort(key=lambda s: (s.start_line, s.id))

    if json_output:
        # Emit raw payload for scripting.
        payload = {"segments": [s.raw for s in results]}
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    count = 0
    for seg in results:
        print_result_text(seg, theory_lines, show_text)
        count += 1
        if limit and count >= limit:
            break

    print(f"Total matched segments: {len(results)}")
    if limit and len(results) > limit:
        print(f"(Shown first {limit} results.)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Search / manage theory index (ALL_index.yaml).",
    )
    p.add_argument(
        "--index",
        required=True,
        help="Path to ALL_index.yaml (or compatible index file).",
    )
    p.add_argument(
        "--theory",
        help=(
            "Path to theory markdown file (e.g., All.md). "
            "If provided, phrase search also scans raw text and --show-text shows snippets."
        ),
    )
    p.add_argument(
        "--tag",
        help="Tag/id substring to search for (case-insensitive). Example: 7d-80.",
    )
    p.add_argument(
        "--category",
        help="Category substring to search for (case-insensitive).",
    )
    p.add_argument(
        "--phrase",
        help=(
            "Free-text phrase to search in keywords, summary and (optionally) raw text. "
            "Case-insensitive substring match."
        ),
    )
    p.add_argument(
        "--show-text",
        action="store_true",
        help="Show raw text snippet for each matched segment (requires --theory).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of results to show (0 = no limit).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable pickle cache for the index (always read YAML).",
    )
    p.add_argument(
        "--sort-by",
        choices=["id", "category", "start_line"],
        help="Optional sorting key for search results.",
    )
    p.add_argument(
        "--json-output",
        action="store_true",
        help="Emit search results as JSON instead of human-readable text.",
    )
    p.add_argument(
        "--mode",
        choices=[
            "search",
            "stats",
            "stats-extended",
            "validate",
            "tree",
            "line",
            "range",
            "export",
            "assemble",
        ],
        default="search",
        help="Operation mode (default: search).",
    )
    p.add_argument(
        "--line",
        dest="line_no",
        type=int,
        help="For mode=line: 1-based line number in the theory file.",
    )
    p.add_argument(
        "--range",
        dest="line_range",
        help="For mode=range: line range 'lo-hi' (inclusive, 1-based).",
    )
    p.add_argument(
        "--export-path",
        help="For mode=export: output file (.json or .yaml/.yml).",
    )
    p.add_argument(
        "--output-path",
        help="For mode=assemble: output markdown file to write assembled text.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Load index (with or without cache)
    index = load_index(args.index, use_cache=not args.no_cache)

    # Theory lines are optional; load when needed.
    theory_lines: Optional[List[str]] = None
    if args.theory and args.mode in ("search", "assemble"):
        try:
            theory_lines = load_theory_lines(args.theory)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}", file=sys.stderr)
            if args.mode == "search":
                print("Phrase matching will ignore raw text; --show-text will do nothing.", file=sys.stderr)
    elif args.theory and args.mode in ("line", "range"):
        try:
            theory_lines = load_theory_lines(args.theory)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}", file=sys.stderr)

    mode = args.mode

    if mode == "stats":
        return mode_stats(index)
    if mode == "stats-extended":
        return mode_stats_extended(index)
    if mode == "validate":
        return mode_validate(index)
    if mode == "tree":
        return mode_tree(index)
    if mode == "line":
        if args.line_no is None:
            print("ERROR: --line is required for mode=line.", file=sys.stderr)
            return 2
        return mode_line(index, args.line_no)
    if mode == "range":
        if args.line_range is None:
            print("ERROR: --range is required for mode=range.", file=sys.stderr)
            return 2
        return mode_range(index, args.line_range)
    if mode == "export":
        # Use search filters to get subset, then export it.
        theory_lines_for_export: Optional[List[str]] = None
        cache_text: Dict[str, str] = {}
        subset: List[Segment] = []
        for seg in index.segments:
            if not matches_tag(seg, args.tag):
                continue
            if not matches_category(seg, args.category):
                continue
            if not matches_phrase(seg, args.phrase, theory_lines_for_export, cache_text):
                continue
            subset.append(seg)
        return mode_export(subset, args.export_path or "")

    if mode == "assemble":
        # Use same filtering rules as search, but assemble instead of printing.
        cache_text: Dict[str, str] = {}
        subset: List[Segment] = []
        for seg in index.segments:
            if not matches_tag(seg, args.tag):
                continue
            if not matches_category(seg, args.category):
                continue
            if not matches_phrase(seg, args.phrase, theory_lines, cache_text):
                continue
            subset.append(seg)
        return mode_assemble(theory_lines, subset, args.output_path or "")

    # Default: search
    return mode_search(
        index=index,
        theory_lines=theory_lines,
        tag=args.tag,
        category=args.category,
        phrase=args.phrase,
        show_text=args.show_text,
        limit=args.limit,
        sort_by=args.sort_by,
        json_output=args.json_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
