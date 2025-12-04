"""
Unit tests for data_loader module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import healpy as hp
import tarfile
import json
import csv

from utils.io.data_loader import (
    load_healpix_map,
    load_power_spectrum_from_tar,
    load_csv_data,
    load_json_data,
    validate_healpix_map,
)


class TestLoadHealpixMap:
    """Tests for load_healpix_map function."""

    def test_load_valid_healpix_map(self):
        """Test loading a valid HEALPix map."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_map.fits"
            hp.write_map(str(file_path), test_map, overwrite=True)

            loaded_map = load_healpix_map(file_path)

            assert isinstance(loaded_map, np.ndarray)
            assert loaded_map.size == npix
            np.testing.assert_allclose(loaded_map, test_map)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises FileNotFoundError."""
        file_path = Path("/nonexistent/path/map.fits")

        with pytest.raises(FileNotFoundError):
            load_healpix_map(file_path)

    def test_load_invalid_hdu(self):
        """Test loading with invalid HDU number."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_map.fits"
            hp.write_map(str(file_path), test_map, overwrite=True)

            with pytest.raises(ValueError, match="HDU"):
                load_healpix_map(file_path, hdu=10)


class TestLoadPowerSpectrumFromTar:
    """Tests for load_power_spectrum_from_tar function."""

    def test_load_spectrum_from_tar(self):
        """Test loading power spectrum from tar archive."""
        l_values = np.arange(2, 100)
        cl_values = 1.0 / (l_values * (l_values + 1))
        error_values = 0.1 * cl_values

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create spectrum file
            spectrum_file = Path(tmpdir) / "spectrum.txt"
            with open(spectrum_file, "w") as f:
                f.write("l cl error\n")
                for ell, cl, err in zip(l_values, cl_values, error_values):
                    f.write(f"{ell} {cl} {err}\n")

            # Create tar archive
            tar_path = Path(tmpdir) / "spectrum.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(spectrum_file, arcname="spectrum.txt")

            # Load spectrum
            data = load_power_spectrum_from_tar(tar_path, "spectrum.txt")

            assert "l" in data
            assert "cl" in data
            assert "error" in data
            np.testing.assert_allclose(data["l"], l_values)
            np.testing.assert_allclose(data["cl"], cl_values)

    def test_load_nonexistent_tar(self):
        """Test loading from non-existent tar raises FileNotFoundError."""
        tar_path = Path("/nonexistent/path/spectrum.tar.gz")

        with pytest.raises(FileNotFoundError):
            load_power_spectrum_from_tar(tar_path, "spectrum.txt")

    def test_load_nonexistent_file_in_tar(self):
        """Test loading non-existent file from tar raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "empty.tar.gz"
            with tarfile.open(tar_path, "w:gz"):
                pass

            with pytest.raises(ValueError, match="not found in archive"):
                load_power_spectrum_from_tar(tar_path, "nonexistent.txt")


class TestLoadCsvData:
    """Tests for load_csv_data function."""

    def test_load_valid_csv(self):
        """Test loading valid CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "z"])
                writer.writerow([1.0, 2.0, 3.0])
                writer.writerow([4.0, 5.0, 6.0])

            data = load_csv_data(csv_path)

            assert "x" in data
            assert "y" in data
            assert "z" in data
            assert len(data["x"]) == 2
            np.testing.assert_allclose(data["x"], [1.0, 4.0])

    def test_load_nonexistent_csv(self):
        """Test loading non-existent CSV raises FileNotFoundError."""
        csv_path = Path("/nonexistent/path/data.csv")

        with pytest.raises(FileNotFoundError):
            load_csv_data(csv_path)


class TestLoadJsonData:
    """Tests for load_json_data function."""

    def test_load_valid_json(self):
        """Test loading valid JSON file."""
        test_data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": True}}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "data.json"
            with open(json_path, "w") as f:
                json.dump(test_data, f)

            data = load_json_data(json_path)

            assert data == test_data

    def test_load_nonexistent_json(self):
        """Test loading non-existent JSON raises FileNotFoundError."""
        json_path = Path("/nonexistent/path/data.json")

        with pytest.raises(FileNotFoundError):
            load_json_data(json_path)

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises JSONDecodeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "invalid.json"
            with open(json_path, "w") as f:
                f.write("{ invalid json }")

            with pytest.raises(json.JSONDecodeError):
                load_json_data(json_path)


class TestValidateHealpixMap:
    """Tests for validate_healpix_map function."""

    def test_validate_valid_map(self):
        """Test validating a valid HEALPix map."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        result = validate_healpix_map(test_map, nside=nside)
        assert result is True

    def test_validate_invalid_size(self):
        """Test validating map with invalid size raises ValueError."""
        invalid_map = np.random.randn(100)  # Not a valid HEALPix size

        with pytest.raises(ValueError, match="Invalid HEALPix map size"):
            validate_healpix_map(invalid_map)

    def test_validate_nside_mismatch(self):
        """Test validating map with wrong NSIDE raises ValueError."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        with pytest.raises(ValueError, match="NSIDE mismatch"):
            validate_healpix_map(test_map, nside=32)

    def test_validate_empty_map(self):
        """Test validating empty map raises ValueError."""
        empty_map = np.array([])

        with pytest.raises(ValueError, match="empty"):
            validate_healpix_map(empty_map)
