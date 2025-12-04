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

            # Parse data rows
            l_values = []
            cl_values = []
            error_values = []

            for line in lines[1:]:
                if not line.strip() or line.strip().startswith("#"):
                    continue
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


def load_csv_data(file_path: Path) -> Dict[str, np.ndarray]:
    """
    Load CSV data file.

    Args:
        file_path: Path to CSV file

    Returns:
        Dictionary with column names as keys and arrays as values

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        data_dict: Dict[str, Any] = {}
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            if fieldnames is None:
                raise ValueError("CSV file has no header row")

            # Initialize lists for each column
            for field in fieldnames:
                data_dict[field] = []

            # Read rows
            for row in reader:
                for field in fieldnames:
                    value = row.get(field, "")
                    try:
                        # Try to convert to float, fallback to string
                        if value.strip():
                            try:
                                data_dict[field].append(float(value))
                            except ValueError:
                                data_dict[field].append(value)
                        else:
                            data_dict[field].append(np.nan)
                    except Exception:
                        data_dict[field].append(np.nan)

        # Convert lists to numpy arrays
        for field in fieldnames:
            if field:
                try:
                    data_dict[field] = np.array(data_dict[field])
                except Exception:
                    # Keep as list if conversion fails
                    pass

        return data_dict

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
