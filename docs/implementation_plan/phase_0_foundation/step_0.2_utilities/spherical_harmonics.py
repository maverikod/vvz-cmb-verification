"""
Spherical harmonics utilities for CMB verification project.

Provides functions for spherical harmonic decomposition and synthesis.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Tuple
import numpy as np
import healpy as hp


def decompose_map(
    map_data: np.ndarray,
    nside: int,
    l_max: Optional[int] = None
) -> np.ndarray:
    """
    Decompose HEALPix map into spherical harmonics.
    
    Args:
        map_data: HEALPix map array
        nside: HEALPix NSIDE parameter
        l_max: Maximum multipole (if None, uses default)
        
    Returns:
        Array of harmonic coefficients a_lm
        
    Raises:
        ValueError: If map data is invalid
    """
    pass


def synthesize_map(
    alm: np.ndarray,
    nside: int
) -> np.ndarray:
    """
    Synthesize HEALPix map from spherical harmonics.
    
    Args:
        alm: Harmonic coefficients a_lm
        nside: HEALPix NSIDE parameter
        
    Returns:
        HEALPix map array
        
    Raises:
        ValueError: If coefficients are invalid
    """
    pass


def calculate_power_spectrum_from_alm(
    alm: np.ndarray,
    l_max: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate power spectrum C_l from harmonic coefficients.
    
    Args:
        alm: Harmonic coefficients a_lm
        l_max: Maximum multipole (if None, uses data limit)
        
    Returns:
        Tuple of (multipoles, C_l values)
        
    Raises:
        ValueError: If coefficients are invalid
    """
    pass


def convert_healpix_to_alm(
    map_data: np.ndarray,
    nside: int,
    l_max: Optional[int] = None
) -> np.ndarray:
    """
    Convert HEALPix map to spherical harmonic coefficients.
    
    Args:
        map_data: HEALPix map array
        nside: HEALPix NSIDE parameter
        l_max: Maximum multipole
        
    Returns:
        Array of harmonic coefficients a_lm
    """
    pass

