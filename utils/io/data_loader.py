"""
Data loading utilities for CMB verification project.

Provides functions for loading HEALPix maps, power spectra, and other data formats.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import healpy as hp
from astropy.io import fits
import tarfile
import json
import csv

# Try to import pandas for optimized CSV parsing
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore


def load_healpix_map(file_path: Path, field: int = 0, hdu: int = 1) -> np.ndarray:
    """
    Load HEALPix map from FITS file.

    Args:
        file_path: Path to HEALPix FITS file
        field: Field number to read (default: 0)
        hdu: HDU number to read (default: 1)

    Returns:
        HEALPix map array

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"HEALPix FITS file not found: {file_path}")

    try:
        with fits.open(file_path) as hdul:
            if hdu >= len(hdul):
                raise ValueError(
                    f"HDU {hdu} not found in file. File has {len(hdul)} HDUs."
                )
            map_data = hp.read_map(str(file_path), field=field, hdu=hdu, verbose=False)
            return np.asarray(map_data)
    except Exception as e:
        raise ValueError(f"Failed to load HEALPix map from {file_path}: {e}") from e


def load_power_spectrum_from_tar(
    tar_path: Path, spectrum_file: str
) -> Dict[str, np.ndarray]:
    """
    Load power spectrum data from tar.gz archive.

    Args:
        tar_path: Path to tar.gz archive
        spectrum_file: Name of spectrum file within archive

    Returns:
        Dictionary with spectrum data (keys: 'l', 'cl', 'error', etc.)

    Raises:
        FileNotFoundError: If archive or file doesn't exist
        ValueError: If data format is invalid
    """
    if not tar_path.exists():
        raise FileNotFoundError(f"Tar archive not found: {tar_path}")

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # Find the spectrum file in the archive
            members = tar.getnames()
            matching_files = [m for m in members if spectrum_file in m]

            if not matching_files:
                raise ValueError(
                    f"Spectrum file '{spectrum_file}' not found in archive. "
                    f"Available files: {members[:10]}"
                )

            # Use first matching file
            target_file = matching_files[0]
            extracted_file = tar.extractfile(target_file)

            if extracted_file is None:
                raise ValueError(f"Could not extract file '{target_file}' from archive")

            # Try to parse as CSV first (most common format)
            content = extracted_file.read().decode("utf-8")
            lines = content.strip().split("\n")

            # Parse header to find column indices
            header = lines[0].split()
            data_dict: Dict[str, np.ndarray] = {}

            # Common column names
            l_col = None
            cl_col = None
            error_col = None

            for i, col in enumerate(header):
                col_lower = col.lower()
                if col_lower in ["l", "ell", "multipole"]:
                    l_col = i
                elif col_lower in ["cl", "c_l", "power"]:
                    cl_col = i
                elif col_lower in ["error", "err", "sigma"]:
                    error_col = i

            # Parse data rows - use vectorized approach
            # Filter out comment lines and empty lines
            data_lines = [
                line
                for line in lines[1:]
                if line.strip() and not line.strip().startswith("#")
            ]

            if not data_lines:
                # No data lines found
                return data_dict

            # Use numpy genfromtxt for vectorized parsing
            try:
                # Convert to string buffer for genfromtxt
                data_str = "\n".join(data_lines)
                from io import StringIO

                # Parse with numpy (faster than loop)
                parsed_data = np.genfromtxt(
                    StringIO(data_str),
                    dtype=float,
                    invalid_raise=False,  # Skip invalid lines
                    filling_values=np.nan,
                )

                # Handle case where only one row exists
                if parsed_data.ndim == 1:
                    parsed_data = parsed_data.reshape(1, -1)

                # Extract columns based on indices
                if l_col is not None and l_col < parsed_data.shape[1]:
                    data_dict["l"] = parsed_data[:, l_col]
                    # Remove NaN values
                    data_dict["l"] = data_dict["l"][~np.isnan(data_dict["l"])]
                if cl_col is not None and cl_col < parsed_data.shape[1]:
                    data_dict["cl"] = parsed_data[:, cl_col]
                    data_dict["cl"] = data_dict["cl"][~np.isnan(data_dict["cl"])]
                if error_col is not None and error_col < parsed_data.shape[1]:
                    data_dict["error"] = parsed_data[:, error_col]
                    data_dict["error"] = data_dict["error"][
                        ~np.isnan(data_dict["error"])
                    ]
            except Exception:
                # Fallback to original loop-based parsing if genfromtxt fails
                l_values = []
                cl_values = []
                error_values = []

                for line in data_lines:
                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    try:
                        if l_col is not None and l_col < len(parts):
                            l_values.append(float(parts[l_col]))
                        if cl_col is not None and cl_col < len(parts):
                            cl_values.append(float(parts[cl_col]))
                        if error_col is not None and error_col < len(parts):
                            error_values.append(float(parts[error_col]))
                    except (ValueError, IndexError):
                        continue

                # Build result dictionary
                if l_values:
                    data_dict["l"] = np.array(l_values)
                if cl_values:
                    data_dict["cl"] = np.array(cl_values)
                if error_values:
                    data_dict["error"] = np.array(error_values)

            if not data_dict:
                raise ValueError(
                    f"Could not parse spectrum data from file '{target_file}'"
                )

            return data_dict

    except tarfile.TarError as e:
        raise ValueError(f"Failed to read tar archive {tar_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load power spectrum from {tar_path}: {e}") from e


def load_csv_data(
    file_path: Path, chunksize: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Load CSV data file.

    Uses pandas for optimized vectorized parsing if available,
    with optional block processing for large files.

    Args:
        file_path: Path to CSV file
        chunksize: Optional chunk size for block processing large files.
                   If None, loads entire file at once.

    Returns:
        Dictionary with column names as keys and arrays as values

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        # Use pandas for optimized CSV parsing if available
        if PANDAS_AVAILABLE and chunksize is None:
            # Load entire file with pandas (vectorized, fast)
            df = pd.read_csv(
                file_path, encoding="utf-8", comment="#", skipinitialspace=True
            )

            # Convert to dictionary with numpy arrays
            data_dict: Dict[str, np.ndarray] = {}
            for col in df.columns:
                # Try to convert to float, keep as object if fails
                try:
                    data_dict[col] = df[col].to_numpy(dtype=float, na_value=np.nan)
                except (ValueError, TypeError):
                    # Keep as string/object array
                    data_dict[col] = df[col].to_numpy()

            return data_dict

        elif PANDAS_AVAILABLE and chunksize is not None:
            # Block processing for large files
            chunk_data_dict: Dict[str, list] = {}
            first_chunk = True

            for chunk in pd.read_csv(
                file_path,
                encoding="utf-8",
                comment="#",
                skipinitialspace=True,
                chunksize=chunksize,
            ):
                if first_chunk:
                    # Initialize with column names
                    for col in chunk.columns:
                        chunk_data_dict[col] = []
                    first_chunk = False

                # Append chunk data
                for col in chunk.columns:
                    chunk_data_dict[col].extend(chunk[col].tolist())

            # Convert lists to numpy arrays
            result_dict: Dict[str, np.ndarray] = {}
            for col, values in chunk_data_dict.items():
                try:
                    result_dict[col] = np.array(values, dtype=float)
                except (ValueError, TypeError):
                    result_dict[col] = np.array(values, dtype=object)

            return result_dict

        else:
            # Fallback to original csv.DictReader approach
            fallback_data_dict: Dict[str, Any] = {}
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames

                if fieldnames is None:
                    raise ValueError("CSV file has no header row")

                # Initialize lists for each column
                for field in fieldnames:
                    fallback_data_dict[field] = []

                # Read rows
                for row in reader:
                    for field in fieldnames:
                        value = row.get(field, "")
                        try:
                            # Try to convert to float, fallback to string
                            if value.strip():
                                try:
                                    fallback_data_dict[field].append(float(value))
                                except ValueError:
                                    fallback_data_dict[field].append(value)
                            else:
                                fallback_data_dict[field].append(np.nan)
                        except Exception:
                            fallback_data_dict[field].append(np.nan)

            # Convert lists to numpy arrays
            for field in fieldnames:
                if field:
                    try:
                        fallback_data_dict[field] = np.array(fallback_data_dict[field])
                    except Exception:
                        # Keep as list if conversion fails
                        pass

            return fallback_data_dict

    except csv.Error as e:
        raise ValueError(f"Failed to parse CSV file {file_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load CSV data from {file_path}: {e}") from e


def load_json_data(file_path: Path) -> Dict[str, Any]:
    """
    Load JSON data file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary with JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in file {file_path}: {e.msg}", e.doc, e.pos
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to load JSON data from {file_path}: {e}") from e


def validate_healpix_map(map_data: np.ndarray, nside: Optional[int] = None) -> bool:
    """
    Validate HEALPix map data.

    Args:
        map_data: HEALPix map array
        nside: Expected NSIDE value (optional)

    Returns:
        True if map is valid

    Raises:
        ValueError: If map is invalid
    """
    if not isinstance(map_data, np.ndarray):
        raise ValueError("Map data must be a numpy array")

    if map_data.size == 0:
        raise ValueError("Map data is empty")

    # Check if size matches a valid HEALPix map
    npix = map_data.size
    try:
        valid_nside = hp.npix2nside(npix)
    except ValueError as e:
        raise ValueError(
            f"Invalid HEALPix map size: {npix}. "
            f"Size must be 12 * nside^2 for some integer nside."
        ) from e

    if valid_nside == 0:
        raise ValueError(
            f"Invalid HEALPix map size: {npix}. "
            f"Size must be 12 * nside^2 for some integer nside."
        )

    # Check NSIDE if specified
    if nside is not None:
        if valid_nside != nside:
            raise ValueError(
                f"NSIDE mismatch: expected {nside}, "
                f"but map size indicates NSIDE={valid_nside}"
            )

    # Check for NaN or Inf values (warn but don't fail)
    if np.any(np.isnan(map_data)):
        # NaN values are sometimes valid (masked pixels)
        pass

    return True
