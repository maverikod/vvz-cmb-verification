"""
Unit tests for theta_data_loader module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import json
import csv

from cmb.theta_data_loader import (
    ThetaFrequencySpectrum,
    ThetaEvolution,
    load_frequency_spectrum,
    load_evolution_data,
    validate_frequency_spectrum,
    validate_evolution_data,
)


class TestThetaFrequencySpectrum:
    """Tests for ThetaFrequencySpectrum dataclass."""

    def test_create_frequency_spectrum(self):
        """Test creating ThetaFrequencySpectrum instance."""
        frequencies = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 1.0])
        spectrum = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        metadata = {"test": "data"}

        spec = ThetaFrequencySpectrum(
            frequencies=frequencies,
            times=times,
            spectrum=spectrum,
            metadata=metadata,
        )

        assert np.array_equal(spec.frequencies, frequencies)
        assert np.array_equal(spec.times, times)
        assert np.array_equal(spec.spectrum, spectrum)
        assert spec.metadata == metadata


class TestThetaEvolution:
    """Tests for ThetaEvolution dataclass."""

    def test_create_evolution(self):
        """Test creating ThetaEvolution instance."""
        times = np.array([0.0, 1.0, 2.0])
        omega_min = np.array([1.0, 1.1, 1.2])
        omega_macro = np.array([10.0, 10.1, 10.2])
        metadata = {"test": "data"}

        evol = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        assert np.array_equal(evol.times, times)
        assert np.array_equal(evol.omega_min, omega_min)
        assert np.array_equal(evol.omega_macro, omega_macro)
        assert evol.metadata == metadata


class TestLoadFrequencySpectrum:
    """Tests for load_frequency_spectrum function."""

    def test_load_csv_frequency_spectrum(self):
        """Test loading frequency spectrum from CSV file."""
        frequencies = np.array([1.0e10, 2.0e10, 3.0e10])
        times = np.array([0.0, 1.0, 2.0])
        spectrum_2d = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "frequency_spectrum.csv"

            # Write CSV file: one row per (frequency, time) combination
            # with spectrum value. The parser will extract unique
            # frequencies and times, then reshape spectrum
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frequency", "time", "spectrum"])
                for i, freq in enumerate(frequencies):
                    for j, time in enumerate(times):
                        writer.writerow([freq, time, spectrum_2d[i, j]])

            # Load and check
            spec = load_frequency_spectrum(csv_path)

            assert isinstance(spec, ThetaFrequencySpectrum)
            assert len(spec.frequencies) == len(frequencies)
            assert len(spec.times) == len(times)
            assert spec.spectrum.shape == (len(frequencies), len(times))
            assert "source_file" in spec.metadata
            assert spec.metadata["format"] == "CSV"

    def test_load_json_frequency_spectrum(self):
        """Test loading frequency spectrum from JSON file."""
        frequencies = np.array([1.0e10, 2.0e10, 3.0e10])
        times = np.array([0.0, 1.0, 2.0])
        spectrum_2d = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frequency_spectrum.json"

            # Write JSON file
            data = {
                "frequencies": frequencies.tolist(),
                "times": times.tolist(),
                "spectrum": spectrum_2d.tolist(),
                "metadata": {"test": "data"},
            }

            with open(json_path, "w") as f:
                json.dump(data, f)

            # Load and check
            spec = load_frequency_spectrum(json_path)

            assert isinstance(spec, ThetaFrequencySpectrum)
            assert len(spec.frequencies) == len(frequencies)
            assert len(spec.times) == len(times)
            assert spec.spectrum.shape == (len(frequencies), len(times))
            assert "source_file" in spec.metadata
            assert spec.metadata["format"] == "JSON"

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises FileNotFoundError."""
        file_path = Path("/nonexistent/path/spectrum.csv")

        with pytest.raises(FileNotFoundError):
            load_frequency_spectrum(file_path)

    def test_load_csv_missing_column(self):
        """Test loading CSV with missing column raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "invalid.csv"

            # Write CSV without frequency column
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "spectrum"])
                writer.writerow([0.0, 1.0])

            with pytest.raises(ValueError, match="Frequency column not found"):
                load_frequency_spectrum(csv_path)

    def test_load_json_missing_key(self):
        """Test loading JSON with missing key raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "invalid.json"

            # Write JSON without frequencies key
            data = {"times": [0.0, 1.0], "spectrum": [[1.0], [2.0]]}

            with open(json_path, "w") as f:
                json.dump(data, f)

            with pytest.raises(
                ValueError, match="'frequencies' key not found"
            ):
                load_frequency_spectrum(json_path)


class TestLoadEvolutionData:
    """Tests for load_evolution_data function."""

    def test_load_csv_evolution_data(self):
        """Test loading evolution data from CSV file."""
        times = np.array([0.0, 1.0, 2.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "evolution.csv"

            # Write CSV file
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "omega_min", "omega_macro"])
                for t, om_min, om_macro in zip(times, omega_min, omega_macro):
                    writer.writerow([t, om_min, om_macro])

            # Load and check
            evol = load_evolution_data(csv_path)

            assert isinstance(evol, ThetaEvolution)
            assert len(evol.times) == len(times)
            assert len(evol.omega_min) == len(omega_min)
            assert len(evol.omega_macro) == len(omega_macro)
            assert "source_file" in evol.metadata
            assert evol.metadata["format"] == "CSV"

    def test_load_json_evolution_data(self):
        """Test loading evolution data from JSON file."""
        times = np.array([0.0, 1.0, 2.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "evolution.json"

            # Write JSON file
            data = {
                "times": times.tolist(),
                "omega_min": omega_min.tolist(),
                "omega_macro": omega_macro.tolist(),
                "metadata": {"test": "data"},
            }

            with open(json_path, "w") as f:
                json.dump(data, f)

            # Load and check
            evol = load_evolution_data(json_path)

            assert isinstance(evol, ThetaEvolution)
            assert len(evol.times) == len(times)
            assert len(evol.omega_min) == len(omega_min)
            assert len(evol.omega_macro) == len(omega_macro)
            assert "source_file" in evol.metadata
            assert evol.metadata["format"] == "JSON"

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises FileNotFoundError."""
        file_path = Path("/nonexistent/path/evolution.csv")

        with pytest.raises(FileNotFoundError):
            load_evolution_data(file_path)

    def test_load_csv_missing_column(self):
        """Test loading CSV with missing column raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "invalid.csv"

            # Write CSV without omega_min column
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "omega_macro"])
                writer.writerow([0.0, 10.0])

            with pytest.raises(ValueError, match="omega_min column not found"):
                load_evolution_data(csv_path)


class TestValidateFrequencySpectrum:
    """Tests for validate_frequency_spectrum function."""

    def test_validate_valid_spectrum(self):
        """Test validating valid frequency spectrum."""
        frequencies = np.array([1.0e10, 2.0e10, 3.0e10])
        times = np.array([0.0, 1.0])
        spectrum = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        metadata = {}

        spec = ThetaFrequencySpectrum(
            frequencies=frequencies,
            times=times,
            spectrum=spectrum,
            metadata=metadata,
        )

        assert validate_frequency_spectrum(spec) is True

    def test_validate_empty_frequencies(self):
        """Test validating spectrum with empty frequencies."""
        frequencies = np.array([])
        times = np.array([0.0, 1.0])
        spectrum = np.array([[1.0, 2.0]])
        metadata = {}

        spec = ThetaFrequencySpectrum(
            frequencies=frequencies,
            times=times,
            spectrum=spectrum,
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="Frequency array is empty"):
            validate_frequency_spectrum(spec)

    def test_validate_negative_frequencies(self):
        """Test validating spectrum with negative frequencies."""
        frequencies = np.array([-1.0, 2.0, 3.0])
        times = np.array([0.0, 1.0])
        spectrum = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        metadata = {}

        spec = ThetaFrequencySpectrum(
            frequencies=frequencies,
            times=times,
            spectrum=spectrum,
            metadata=metadata,
        )

        with pytest.raises(
            ValueError, match="All frequencies must be positive"
        ):
            validate_frequency_spectrum(spec)

    def test_validate_negative_spectrum(self):
        """Test validating spectrum with negative values raises ValueError."""
        frequencies = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 1.0])
        spectrum = np.array([[-1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        metadata = {}

        spec = ThetaFrequencySpectrum(
            frequencies=frequencies,
            times=times,
            spectrum=spectrum,
            metadata=metadata,
        )

        with pytest.raises(
            ValueError, match="All spectrum values must be non-negative"
        ):
            validate_frequency_spectrum(spec)

    def test_validate_shape_mismatch(self):
        """Test validating spectrum with shape mismatch raises ValueError."""
        frequencies = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 1.0])
        spectrum = np.array([[1.0, 2.0], [3.0, 4.0]])  # Wrong shape
        metadata = {}

        spec = ThetaFrequencySpectrum(
            frequencies=frequencies,
            times=times,
            spectrum=spectrum,
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="Spectrum shape mismatch"):
            validate_frequency_spectrum(spec)

    def test_validate_nan_frequencies(self):
        """Test validating spectrum with NaN frequencies raises ValueError."""
        frequencies = np.array([1.0, np.nan, 3.0])
        times = np.array([0.0, 1.0])
        spectrum = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        metadata = {}

        spec = ThetaFrequencySpectrum(
            frequencies=frequencies,
            times=times,
            spectrum=spectrum,
            metadata=metadata,
        )

        with pytest.raises(
            ValueError, match="Frequency array contains NaN values"
        ):
            validate_frequency_spectrum(spec)


class TestValidateEvolutionData:
    """Tests for validate_evolution_data function."""

    def test_validate_valid_evolution(self):
        """Test validating valid evolution data."""
        times = np.array([0.0, 1.0, 2.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])
        metadata = {}

        evol = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        assert validate_evolution_data(evol) is True

    def test_validate_empty_times(self):
        """Test validating evolution with empty times raises ValueError."""
        times = np.array([])
        omega_min = np.array([1.0e10])
        omega_macro = np.array([10.0e10])
        metadata = {}

        evol = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="Time array is empty"):
            validate_evolution_data(evol)

    def test_validate_length_mismatch(self):
        """Test validating evolution with length mismatch raises ValueError."""
        times = np.array([0.0, 1.0, 2.0])
        omega_min = np.array([1.0e10, 1.1e10])  # Wrong length
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])
        metadata = {}

        evol = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="Array length mismatch"):
            validate_evolution_data(evol)

    def test_validate_negative_omega_min(self):
        """Test validating evolution with negative omega_min."""
        times = np.array([0.0, 1.0, 2.0])
        omega_min = np.array([-1.0e10, 1.1e10, 1.2e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])
        metadata = {}

        evol = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        with pytest.raises(
            ValueError, match="All omega_min values must be positive"
        ):
            validate_evolution_data(evol)

    def test_validate_nan_times(self):
        """Test validating evolution with NaN times raises ValueError."""
        times = np.array([0.0, np.nan, 2.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])
        metadata = {}

        evol = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="Time array contains NaN values"):
            validate_evolution_data(evol)

    def test_validate_omega_min_greater_than_macro(self):
        """Test validating evolution where omega_min >= omega_macro."""
        times = np.array([0.0, 1.0, 2.0])
        # Second > omega_macro
        omega_min = np.array([1.0e10, 11.0e10, 1.2e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])
        metadata = {}

        evol = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        with pytest.raises(
            ValueError, match="omega_min must be < omega_macro"
        ):
            validate_evolution_data(evol)
