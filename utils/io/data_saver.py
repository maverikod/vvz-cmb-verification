"""
Data saving utilities for CMB verification project.

Provides functions for saving HEALPix maps, power spectra, and other results.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import healpy as hp
import json
import csv

# Try to import pandas for optimized CSV writing
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore


def save_healpix_map(
    map_data: np.ndarray, file_path: Path, nside: int, overwrite: bool = False
) -> None:
    """
    Save HEALPix map to FITS file.

    Args:
        map_data: HEALPix map array
        file_path: Path to output FITS file
        nside: HEALPix NSIDE parameter
        overwrite: Whether to overwrite existing file

    Raises:
        FileExistsError: If file exists and overwrite=False
        ValueError: If map data is invalid
    """
    if not isinstance(map_data, np.ndarray):
        raise ValueError("Map data must be a numpy array")

    # Validate map size
    expected_npix = 12 * nside * nside
    if map_data.size != expected_npix:
        raise ValueError(
            f"Map size {map_data.size} does not match NSIDE {nside} "
            f"(expected {expected_npix} pixels)"
        )

    # Check if file exists
    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"File {file_path} already exists. Use overwrite=True to replace."
        )

    # Ensure output directory exists
    ensure_output_directory(file_path)

    try:
        # Write HEALPix map to FITS
        # Note: nside is determined automatically from map size
        hp.write_map(
            str(file_path),
            map_data,
            overwrite=overwrite,
            coord="G",  # Galactic coordinates
            dtype=None,  # Preserve input dtype
        )
    except Exception as e:
        raise ValueError(f"Failed to save HEALPix map to {file_path}: {e}") from e


def save_power_spectrum(
    spectrum_data: Dict[str, np.ndarray], file_path: Path, format: str = "json"
) -> None:
    """
    Save power spectrum data to file.

    Args:
        spectrum_data: Dictionary with spectrum data (keys: 'l', 'cl', 'error')
        file_path: Path to output file
        format: Output format ('json', 'csv', 'npy')

    Raises:
        ValueError: If format is invalid or data is missing
    """
    if "l" not in spectrum_data or "cl" not in spectrum_data:
        raise ValueError("spectrum_data must contain 'l' and 'cl' keys")

    ensure_output_directory(file_path)

    format_lower = format.lower()

    if format_lower == "json":
        # Convert numpy arrays to lists for JSON
        json_data: Dict[str, Any] = {}
        for key, value in spectrum_data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

    elif format_lower == "csv":
        # Use pandas for optimized vectorized CSV writing if available
        if PANDAS_AVAILABLE:
            # Convert to DataFrame for efficient writing
            # Get all keys and ensure they have same length
            keys = list(spectrum_data.keys())
            lengths = [
                len(spectrum_data[k])
                for k in keys
                if isinstance(spectrum_data[k], np.ndarray)
            ]

            if not lengths:
                raise ValueError("No valid array data in spectrum_data")

            max_len = max(lengths)
            if not all(
                len(spectrum_data[k]) == max_len
                for k in keys
                if isinstance(spectrum_data[k], np.ndarray)
            ):
                raise ValueError(
                    "All arrays in spectrum_data must have the same length"
                )

            # Build DataFrame
            df_data: Dict[str, Any] = {}
            for key in keys:
                value = spectrum_data[key]
                if isinstance(value, np.ndarray):
                    df_data[key] = value
                else:
                    # Repeat scalar value for all rows
                    df_data[key] = [value] * max_len

            df = pd.DataFrame(df_data)
            # Write with pandas (vectorized, faster)
            df.to_csv(file_path, index=False, encoding="utf-8")
        else:
            # Fallback to original csv.writer approach
            # Get all keys and ensure they have same length
            keys = list(spectrum_data.keys())
            lengths = [
                len(spectrum_data[k])
                for k in keys
                if isinstance(spectrum_data[k], np.ndarray)
            ]

            if not lengths:
                raise ValueError("No valid array data in spectrum_data")

            max_len = max(lengths)
            if not all(
                len(spectrum_data[k]) == max_len
                for k in keys
                if isinstance(spectrum_data[k], np.ndarray)
            ):
                raise ValueError(
                    "All arrays in spectrum_data must have the same length"
                )

            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(keys)
                # Write data rows
                for i in range(max_len):
                    row = []
                    for key in keys:
                        value = spectrum_data[key]
                        if isinstance(value, np.ndarray):
                            row.append(str(value[i]) if i < len(value) else "")
                        else:
                            row.append(str(value))
                    writer.writerow(row)

    elif format_lower == "npy":
        # Save as numpy archive
        np.savez(str(file_path), **spectrum_data)  # type: ignore[arg-type]

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json', 'csv', or 'npy'.")


def save_analysis_results(
    results: Dict[str, Any], file_path: Path, format: str = "json"
) -> None:
    """
    Save analysis results to file.

    Args:
        results: Dictionary with analysis results
        file_path: Path to output file
        format: Output format ('json', 'yaml')

    Raises:
        ValueError: If format is invalid
    """
    ensure_output_directory(file_path)

    format_lower = format.lower()

    if format_lower == "json":
        # Convert numpy arrays and other non-serializable types
        def convert_to_serializable(obj: Any) -> Any:
            """
            Convert object to JSON-serializable format.

            Args:
                obj: Object to convert (can be numpy array, dict, list, etc.)

            Returns:
                JSON-serializable representation of the object
            """
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        json_data = convert_to_serializable(results)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

    elif format_lower == "yaml":
        try:
            import yaml
        except ImportError:
            raise ValueError(
                "YAML format requires 'pyyaml' package. "
                "Install with: pip install pyyaml"
            )

        def convert_to_yaml_serializable(obj: Any) -> Any:
            """
            Convert object to YAML-serializable format.

            Args:
                obj: Object to convert (can be numpy array, dict, list, etc.)

            Returns:
                YAML-serializable representation of the object
            """
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_yaml_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_yaml_serializable(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        yaml_data = convert_to_yaml_serializable(results)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")


def ensure_output_directory(output_path: Path) -> None:
    """
    Ensure output directory exists, create if needed.

    Args:
        output_path: Path to output file or directory

    Raises:
        PermissionError: If directory cannot be created
    """
    output_dir = output_path.parent

    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create output directory {output_dir}: {e}"
            ) from e
        except Exception as e:
            raise PermissionError(
                f"Failed to create output directory {output_dir}: {e}"
            ) from e
