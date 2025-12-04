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
    map_data: np.ndarray, nside: int, l_max: Optional[int] = None
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
    if not isinstance(map_data, np.ndarray):
        raise ValueError("Map data must be a numpy array")

    # Validate map size
    expected_npix = 12 * nside * nside
    if map_data.size != expected_npix:
        raise ValueError(
            f"Map size {map_data.size} does not match NSIDE {nside} "
            f"(expected {expected_npix} pixels)"
        )

    # Determine l_max if not provided
    if l_max is None:
        # Default: 3 * nside - 1 (Nyquist limit for HEALPix)
        l_max = 3 * nside - 1

    if l_max < 0:
        raise ValueError(f"l_max must be non-negative, got {l_max}")

    try:
        # Perform spherical harmonic decomposition
        alm = hp.map2alm(map_data, lmax=l_max, iter=0)
        return alm
    except Exception as e:
        raise ValueError(f"Failed to decompose map: {e}") from e


def synthesize_map(alm: np.ndarray, nside: int) -> np.ndarray:
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
    if not isinstance(alm, np.ndarray):
        raise ValueError("Harmonic coefficients must be a numpy array")

    if alm.size == 0:
        raise ValueError("Harmonic coefficients array is empty")

    if nside <= 0:
        raise ValueError(f"NSIDE must be positive, got {nside}")

    try:
        # Synthesize map from spherical harmonics
        map_data = hp.alm2map(alm, nside=nside)
        return np.asarray(map_data)
    except Exception as e:
        raise ValueError(f"Failed to synthesize map: {e}") from e


def calculate_power_spectrum_from_alm(
    alm: np.ndarray, l_max: Optional[int] = None
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
    if not isinstance(alm, np.ndarray):
        raise ValueError("Harmonic coefficients must be a numpy array")

    if alm.size == 0:
        raise ValueError("Harmonic coefficients array is empty")

    try:
        # Calculate power spectrum
        cl = hp.alm2cl(alm)

        # Get multipole values
        if l_max is None:
            l_max = len(cl) - 1

        multipoles = np.arange(l_max + 1)

        # Truncate to requested l_max
        if len(cl) > l_max + 1:
            cl = cl[: l_max + 1]

        return (multipoles, cl)
    except Exception as e:
        raise ValueError(f"Failed to calculate power spectrum: {e}") from e


def convert_healpix_to_alm(
    map_data: np.ndarray, nside: int, l_max: Optional[int] = None
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
    # This is essentially the same as decompose_map
    return decompose_map(map_data, nside, l_max)
