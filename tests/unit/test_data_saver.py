"""
Unit tests for data_saver module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import healpy as hp
import json

from utils.io.data_saver import (
    save_healpix_map,
    save_power_spectrum,
    save_analysis_results,
    ensure_output_directory,
)


class TestSaveHealpixMap:
    """Tests for save_healpix_map function."""

    def test_save_valid_map(self):
        """Test saving a valid HEALPix map."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "output_map.fits"
            save_healpix_map(test_map, file_path, nside, overwrite=True)

            assert file_path.exists()

            # Verify we can read it back
            loaded_map = hp.read_map(str(file_path))
            np.testing.assert_allclose(loaded_map, test_map, rtol=1e-5)

    def test_save_without_overwrite(self):
        """Test saving without overwrite flag raises FileExistsError."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "output_map.fits"
            save_healpix_map(test_map, file_path, nside, overwrite=True)

            with pytest.raises(FileExistsError):
                save_healpix_map(test_map, file_path, nside, overwrite=False)

    def test_save_invalid_size(self):
        """Test saving map with invalid size raises ValueError."""
        nside = 64
        invalid_map = np.random.randn(100)  # Wrong size

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "output_map.fits"

            with pytest.raises(ValueError, match="does not match NSIDE"):
                save_healpix_map(invalid_map, file_path, nside)


class TestSavePowerSpectrum:
    """Tests for save_power_spectrum function."""

    def test_save_spectrum_json(self):
        """Test saving power spectrum as JSON."""
        spectrum_data = {
            "l": np.arange(2, 100),
            "cl": 1.0 / (np.arange(2, 100) * (np.arange(2, 100) + 1)),
            "error": np.random.rand(98) * 0.1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "spectrum.json"
            save_power_spectrum(spectrum_data, file_path, format="json")

            assert file_path.exists()

            # Verify content
            with open(file_path) as f:
                loaded_data = json.load(f)

            assert "l" in loaded_data
            assert "cl" in loaded_data
            assert len(loaded_data["l"]) == 98

    def test_save_spectrum_csv(self):
        """Test saving power spectrum as CSV."""
        spectrum_data = {
            "l": np.arange(2, 10),
            "cl": 1.0 / (np.arange(2, 10) * (np.arange(2, 10) + 1)),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "spectrum.csv"
            save_power_spectrum(spectrum_data, file_path, format="csv")

            assert file_path.exists()

    def test_save_spectrum_missing_keys(self):
        """Test saving spectrum without required keys raises ValueError."""
        spectrum_data = {"l": np.arange(2, 10)}  # Missing 'cl'

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "spectrum.json"

            with pytest.raises(ValueError, match="must contain"):
                save_power_spectrum(spectrum_data, file_path)

    def test_save_spectrum_invalid_format(self):
        """Test saving with invalid format raises ValueError."""
        spectrum_data = {
            "l": np.arange(2, 10),
            "cl": np.arange(2, 10),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "spectrum.xyz"

            with pytest.raises(ValueError, match="Unsupported format"):
                save_power_spectrum(spectrum_data, file_path, format="xyz")


class TestSaveAnalysisResults:
    """Tests for save_analysis_results function."""

    def test_save_results_json(self):
        """Test saving analysis results as JSON."""
        results = {
            "mean": 1.5,
            "std": 0.3,
            "data": np.array([1, 2, 3, 4, 5]),
            "nested": {"key": "value"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "results.json"
            save_analysis_results(results, file_path, format="json")

            assert file_path.exists()

            # Verify content
            with open(file_path) as f:
                loaded_data = json.load(f)

            assert loaded_data["mean"] == 1.5
            assert loaded_data["std"] == 0.3
            assert len(loaded_data["data"]) == 5

    def test_save_results_invalid_format(self):
        """Test saving with invalid format raises ValueError."""
        results = {"key": "value"}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "results.xyz"

            with pytest.raises(ValueError, match="Unsupported format"):
                save_analysis_results(results, file_path, format="xyz")


class TestEnsureOutputDirectory:
    """Tests for ensure_output_directory function."""

    def test_create_directory(self):
        """Test creating output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "file.txt"
            ensure_output_directory(output_path)

            assert output_path.parent.exists()

    def test_existing_directory(self):
        """Test with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "file.txt"
            ensure_output_directory(output_path)
            # Should not raise
            ensure_output_directory(output_path)
