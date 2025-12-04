"""
Unit tests for spherical_harmonics module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np
import healpy as hp

from utils.math.spherical_harmonics import (
    decompose_map,
    synthesize_map,
    calculate_power_spectrum_from_alm,
    convert_healpix_to_alm,
)


class TestDecomposeMap:
    """Tests for decompose_map function."""

    def test_decompose_valid_map(self):
        """Test decomposing a valid HEALPix map."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        alm = decompose_map(test_map, nside)

        assert isinstance(alm, np.ndarray)
        assert alm.size > 0

    def test_decompose_with_l_max(self):
        """Test decomposing with specified l_max."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)
        l_max = 50

        alm = decompose_map(test_map, nside, l_max=l_max)

        assert isinstance(alm, np.ndarray)

    def test_decompose_invalid_size(self):
        """Test decomposing map with invalid size raises ValueError."""
        nside = 64
        invalid_map = np.random.randn(100)  # Wrong size

        with pytest.raises(ValueError, match="does not match NSIDE"):
            decompose_map(invalid_map, nside)

    def test_decompose_negative_l_max(self):
        """Test decomposing with negative l_max raises ValueError."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        with pytest.raises(ValueError, match="must be non-negative"):
            decompose_map(test_map, nside, l_max=-1)


class TestSynthesizeMap:
    """Tests for synthesize_map function."""

    def test_synthesize_valid_alm(self):
        """Test synthesizing map from valid a_lm."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        # Decompose first
        alm = hp.map2alm(test_map, lmax=3 * nside - 1)

        # Synthesize
        synthesized_map = synthesize_map(alm, nside)

        assert isinstance(synthesized_map, np.ndarray)
        assert synthesized_map.size == npix

    def test_synthesize_empty_alm(self):
        """Test synthesizing with empty a_lm raises ValueError."""
        empty_alm = np.array([])

        with pytest.raises(ValueError, match="empty"):
            synthesize_map(empty_alm, nside=64)

    def test_synthesize_invalid_nside(self):
        """Test synthesizing with invalid NSIDE raises ValueError."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)
        alm = hp.map2alm(test_map, lmax=3 * nside - 1)

        with pytest.raises(ValueError, match="must be positive"):
            synthesize_map(alm, nside=-1)


class TestCalculatePowerSpectrumFromAlm:
    """Tests for calculate_power_spectrum_from_alm function."""

    def test_calculate_power_spectrum(self):
        """Test calculating power spectrum from a_lm."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)
        alm = hp.map2alm(test_map, lmax=3 * nside - 1)

        multipoles, cl = calculate_power_spectrum_from_alm(alm)

        assert isinstance(multipoles, np.ndarray)
        assert isinstance(cl, np.ndarray)
        assert len(multipoles) == len(cl)
        assert multipoles[0] == 0
        assert np.all(cl >= 0)  # Power spectrum should be non-negative

    def test_calculate_with_l_max(self):
        """Test calculating power spectrum with specified l_max."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)
        alm = hp.map2alm(test_map, lmax=3 * nside - 1)
        l_max = 50

        multipoles, cl = calculate_power_spectrum_from_alm(alm, l_max=l_max)

        assert len(multipoles) == l_max + 1
        assert len(cl) == l_max + 1

    def test_calculate_empty_alm(self):
        """Test calculating with empty a_lm raises ValueError."""
        empty_alm = np.array([])

        with pytest.raises(ValueError, match="empty"):
            calculate_power_spectrum_from_alm(empty_alm)


class TestConvertHealpixToAlm:
    """Tests for convert_healpix_to_alm function."""

    def test_convert_valid_map(self):
        """Test converting valid HEALPix map to a_lm."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        alm = convert_healpix_to_alm(test_map, nside)

        assert isinstance(alm, np.ndarray)
        assert alm.size > 0

    def test_round_trip_preserves_map(self):
        """Test round-trip conversion preserves map within precision."""
        nside = 32  # Use smaller nside for better precision
        npix = hp.nside2npix(nside)
        # Use smooth map (low l content) for better round-trip accuracy
        test_map = hp.smoothing(
            np.random.randn(npix) * 0.1, fwhm=np.radians(1.0), verbose=False
        )

        # Convert to a_lm and back using same l_max
        l_max = 3 * nside - 1
        alm = convert_healpix_to_alm(test_map, nside, l_max=l_max)
        recovered_map = synthesize_map(alm, nside)

        # For smooth maps, round-trip should be reasonably accurate
        # Check correlation coefficient instead of exact match
        correlation = np.corrcoef(test_map, recovered_map)[0, 1]
        assert correlation > 0.95, f"Correlation too low: {correlation}"

        # Also check that maps have similar statistics
        assert abs(np.mean(test_map) - np.mean(recovered_map)) < 0.1
        assert abs(np.std(test_map) - np.std(recovered_map)) < 0.1
