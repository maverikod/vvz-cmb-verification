"""
Unit tests for data_index_loader module - validation and search.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import tempfile
from pathlib import Path
import yaml

from utils.io.data_index_loader import DataIndex


class TestValidateFiles:
    """Tests for validate_files() method."""

    def test_validate_existing_files(self):
        """Test validation of existing files."""
        test_data = {
            "data_index": {
                "test_category": {
                    "file": "test_file.txt",
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)

            # Create the file
            test_file = Path(tmpdir) / "test_file.txt"
            test_file.write_text("test content")

            result = index.validate_files()

            assert "found" in result
            assert "missing" in result
            assert len(result["found"]) >= 1
            assert any("test_file.txt" in f for f in result["found"])

    def test_validate_missing_files(self):
        """Test validation of missing files."""
        test_data = {
            "data_index": {
                "test_category": {
                    "file": "nonexistent_file.txt",
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)

            result = index.validate_files()

            assert "found" in result
            assert "missing" in result
            assert len(result["missing"]) >= 1
            assert any("nonexistent_file.txt" in f for f in result["missing"])


class TestGetUsageGuidelines:
    """Tests for get_usage_guidelines() method."""

    def test_get_usage_guidelines_from_section(self):
        """Test getting usage guidelines from usage_guidelines section."""
        test_data = {
            "data_index": {},
            "usage_guidelines": {
                "test_usage": {
                    "description": "Test usage",
                    "primary_data": ["file1.txt", "file2.txt"],
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)
            guidelines = index.get_usage_guidelines("test_usage")

            assert guidelines is not None
            assert guidelines["description"] == "Test usage"
            assert "primary_data" in guidelines

    def test_get_usage_guidelines_from_category(self):
        """Test getting usage guidelines from category data."""
        test_data = {
            "data_index": {
                "test_category": {
                    "description": "Test category",
                    "usage": "Test usage description",
                    "file": "test_file.txt",
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)
            guidelines = index.get_usage_guidelines("test_category")

            assert guidelines is not None
            assert guidelines["description"] == "Test category"
            assert guidelines["usage"] == "Test usage description"

    def test_get_usage_guidelines_nonexistent(self):
        """Test getting usage guidelines for non-existent category."""
        test_data = {"data_index": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)
            guidelines = index.get_usage_guidelines("nonexistent")

            assert guidelines is None


class TestSearchByName:
    """Tests for search_by_name() method."""

    def test_search_exact_match(self):
        """Test searching for exact file name."""
        test_data = {
            "data_index": {
                "test_category": {
                    "file": "test_file.txt",
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)
            matches = index.search_by_name("test_file.txt")

            assert len(matches) == 1
            assert matches[0]["name"] == "test_file.txt"

    def test_search_wildcard(self):
        """Test searching with wildcard pattern."""
        test_data = {
            "data_index": {
                "test_category": {
                    "files": [
                        {"name": "file1.txt"},
                        {"name": "file2.txt"},
                        {"name": "data.csv"},
                    ]
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)
            matches = index.search_by_name("file*.txt")

            assert len(matches) == 2
            assert all(
                "file" in m["name"] and m["name"].endswith(".txt")
                for m in matches
            )

    def test_search_no_matches(self):
        """Test searching with pattern that matches nothing."""
        test_data = {
            "data_index": {
                "test_category": {
                    "file": "test_file.txt",
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)
            matches = index.search_by_name("nonexistent*")

            assert matches == []
