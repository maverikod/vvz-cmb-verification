"""
Data loading utilities for CMB verification project.

Provides functions for loading HEALPix maps, power spectra, and other data formats.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import healpy as hp
from astropy.io import fits
import tarfile
import json
import csv


def load_healpix_map(
    file_path: Path,
    field: int = 0,
    hdu: int = 1
) -> np.ndarray:
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
    pass


def load_power_spectrum_from_tar(
    tar_path: Path,
    spectrum_file: str
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
    pass


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
    pass


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
    pass


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
    pass

