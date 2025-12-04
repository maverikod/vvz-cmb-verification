"""
Unit tests for frequency_conversion module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np

from utils.math.frequency_conversion import (
    frequency_to_multipole,
    multipole_to_frequency,
    get_frequency_range_for_multipole_range,
)


class TestFrequencyToMultipole:
    """Tests for frequency_to_multipole function."""

    def test_convert_single_frequency(self):
        """Test converting a single frequency value."""
        frequency = 1.0e11  # Hz
        D = 1.0e-8

        multipole = frequency_to_multipole(frequency, D)

        expected = np.pi * D * frequency
        assert abs(multipole - expected) < 1e-10
        assert isinstance(multipole, float)

    def test_convert_array_frequencies(self):
        """Test converting array of frequencies."""
        frequencies = np.array([1.0e11, 2.0e11, 3.0e11])
        D = 1.0e-8

        multipoles = frequency_to_multipole(frequencies, D)

        expected = np.pi * D * frequencies
        np.testing.assert_allclose(multipoles, expected)

    def test_negative_frequency_raises_error(self):
        """Test that negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            frequency_to_multipole(-1.0, D=1.0e-8)

    def test_zero_frequency_raises_error(self):
        """Test that zero frequency raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            frequency_to_multipole(0.0, D=1.0e-8)

    def test_negative_D_raises_error(self):
        """Test that negative D parameter raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            frequency_to_multipole(1.0e11, D=-1.0)


class TestMultipoleToFrequency:
    """Tests for multipole_to_frequency function."""

    def test_convert_single_multipole(self):
        """Test converting a single multipole value."""
        multipole = 1000.0
        D = 1.0e-8

        frequency = multipole_to_frequency(multipole, D)

        expected = multipole / (np.pi * D)
        assert abs(frequency - expected) < 1e-5
        assert isinstance(frequency, float)

    def test_convert_array_multipoles(self):
        """Test converting array of multipoles."""
        multipoles = np.array([100, 200, 300])
        D = 1.0e-8

        frequencies = multipole_to_frequency(multipoles, D)

        expected = multipoles / (np.pi * D)
        np.testing.assert_allclose(frequencies, expected)

    def test_round_trip_conversion(self):
        """Test round-trip conversion preserves values."""
        original_freq = 1.0e11
        D = 1.0e-8

        multipole = frequency_to_multipole(original_freq, D)
        recovered_freq = multipole_to_frequency(multipole, D)

        assert abs(original_freq - recovered_freq) / original_freq < 1e-10

    def test_negative_multipole_raises_error(self):
        """Test that negative multipole raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            multipole_to_frequency(-1.0, D=1.0e-8)

    def test_zero_multipole_raises_error(self):
        """Test that zero multipole raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            multipole_to_frequency(0.0, D=1.0e-8)


class TestGetFrequencyRangeForMultipoleRange:
    """Tests for get_frequency_range_for_multipole_range function."""

    def test_get_frequency_range(self):
        """Test getting frequency range for multipole range."""
        l_min = 100.0
        l_max = 1000.0
        D = 1.0e-8

        freq_min, freq_max = get_frequency_range_for_multipole_range(l_min, l_max, D)

        assert freq_min < freq_max
        assert freq_min > 0
        assert freq_max > 0

        # Verify using inverse conversion
        recovered_l_min = frequency_to_multipole(freq_min, D)
        recovered_l_max = frequency_to_multipole(freq_max, D)

        assert abs(recovered_l_min - l_min) / l_min < 1e-10
        assert abs(recovered_l_max - l_max) / l_max < 1e-10

    def test_invalid_range_raises_error(self):
        """Test that invalid range raises ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            get_frequency_range_for_multipole_range(1000.0, 100.0, D=1.0e-8)

    def test_equal_bounds_raises_error(self):
        """Test that equal bounds raises ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            get_frequency_range_for_multipole_range(100.0, 100.0, D=1.0e-8)

    def test_negative_bounds_raise_error(self):
        """Test that negative bounds raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            get_frequency_range_for_multipole_range(-100.0, 1000.0, D=1.0e-8)
