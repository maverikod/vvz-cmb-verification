"""
Unit tests for spectrum_plots module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import matplotlib

# Use non-interactive backend for tests
matplotlib.use("Agg")

from utils.visualization.spectrum_plots import (
    plot_power_spectrum,
    plot_spectrum_comparison,
    plot_high_l_peak,
    plot_subpeaks,
)


class TestPlotPowerSpectrum:
    """Tests for plot_power_spectrum function."""

    def test_plot_spectrum(self):
        """Test plotting power spectrum."""
        multipoles = np.arange(2, 100)
        spectrum = 1.0 / (multipoles * (multipoles + 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "spectrum.png"
            plot_power_spectrum(multipoles, spectrum, output_path=output_path)

            assert output_path.exists()

    def test_plot_spectrum_with_errors(self):
        """Test plotting power spectrum with error bars."""
        multipoles = np.arange(2, 100)
        spectrum = 1.0 / (multipoles * (multipoles + 1))
        errors = 0.1 * spectrum

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "spectrum.png"
            plot_power_spectrum(
                multipoles, spectrum, errors=errors, output_path=output_path
            )

            assert output_path.exists()

    def test_plot_spectrum_size_mismatch(self):
        """Test plotting with size mismatch raises ValueError."""
        multipoles = np.arange(2, 100)
        spectrum = np.arange(50)  # Different size

        with pytest.raises(ValueError, match="must match"):
            plot_power_spectrum(multipoles, spectrum)

    def test_plot_spectrum_invalid_data(self):
        """Test plotting with invalid data raises ValueError."""
        with pytest.raises(ValueError, match="must be numpy arrays"):
            plot_power_spectrum([1, 2, 3], [4, 5, 6])


class TestPlotSpectrumComparison:
    """Tests for plot_spectrum_comparison function."""

    def test_plot_comparison(self):
        """Test plotting spectrum comparison."""
        multipoles = np.arange(2, 100)
        spectrum1 = 1.0 / (multipoles * (multipoles + 1))
        spectrum2 = 1.2 / (multipoles * (multipoles + 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "comparison.png"
            plot_spectrum_comparison(
                multipoles, spectrum1, spectrum2, output_path=output_path
            )

            assert output_path.exists()

    def test_plot_comparison_size_mismatch(self):
        """Test plotting comparison with size mismatch raises ValueError."""
        multipoles = np.arange(2, 100)
        spectrum1 = np.arange(50)
        spectrum2 = np.arange(100)

        with pytest.raises(ValueError, match="same size"):
            plot_spectrum_comparison(multipoles, spectrum1, spectrum2)


class TestPlotHighLPeak:
    """Tests for plot_high_l_peak function."""

    def test_plot_peak(self):
        """Test plotting high-l peak."""
        multipoles = np.arange(2, 1000)
        spectrum = 1.0 / (multipoles * (multipoles + 1))
        peak_range = (800.0, 900.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "peak.png"
            plot_high_l_peak(multipoles, spectrum, peak_range, output_path=output_path)

            assert output_path.exists()

    def test_plot_peak_invalid_range(self):
        """Test plotting with invalid range raises ValueError."""
        multipoles = np.arange(2, 100)
        spectrum = 1.0 / (multipoles * (multipoles + 1))
        peak_range = (900.0, 800.0)  # l_min > l_max

        with pytest.raises(ValueError, match="must be less than"):
            plot_high_l_peak(multipoles, spectrum, peak_range)


class TestPlotSubpeaks:
    """Tests for plot_subpeaks function."""

    def test_plot_subpeaks(self):
        """Test plotting sub-peaks."""
        multipoles = np.arange(2, 1000)
        spectrum = 1.0 / (multipoles * (multipoles + 1))
        peak_positions = [100.0, 200.0, 300.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subpeaks.png"
            plot_subpeaks(multipoles, spectrum, peak_positions, output_path=output_path)

            assert output_path.exists()

    def test_plot_subpeaks_empty_list(self):
        """Test plotting with empty peak list raises ValueError."""
        multipoles = np.arange(2, 100)
        spectrum = 1.0 / (multipoles * (multipoles + 1))

        with pytest.raises(ValueError, match="cannot be empty"):
            plot_subpeaks(multipoles, spectrum, [])
