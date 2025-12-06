"""
Map loader utility for CMB map validation.

Loads and validates HEALPix maps from FITS files.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
import numpy as np
import healpy as hp
from utils.io.data_loader import load_healpix_map


class MapLoader:
    """
    HEALPix map loader.

    Loads and validates HEALPix maps, handling resolution differences.
    """

    def __init__(self, nside: int):
        """
        Initialize map loader.

        Args:
            nside: Target HEALPix NSIDE parameter
        """
        self.nside = nside

    def load_map(self, map_path: Path) -> np.ndarray:
        """
        Load HEALPix map from FITS file.

        Args:
            map_path: Path to FITS file

        Returns:
            HEALPix map array with target NSIDE

        Raises:
            FileNotFoundError: If map file doesn't exist
            ValueError: If map format is invalid
        """
        if not map_path.exists():
            raise FileNotFoundError(f"Map file not found: {map_path}")

        try:
            # Load map using utility function
            map_data = load_healpix_map(map_path)

            # Validate and resample if needed
            expected_npix = 12 * self.nside * self.nside
            if map_data.size != expected_npix:
                # Try to handle different NSIDE by downgrading/upgrading
                observed_nside = hp.npix2nside(map_data.size)
                if observed_nside != self.nside:
                    # Resample to match nside
                    map_data = hp.ud_grade(map_data, nside_out=self.nside)

            # Validate final size
            if map_data.size != expected_npix:
                raise ValueError(
                    f"Map size mismatch: expected {expected_npix}, "
                    f"got {map_data.size}"
                )

            return map_data

        except Exception as e:
            raise ValueError(f"Failed to load map: {e}") from e
