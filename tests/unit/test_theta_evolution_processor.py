"""
Unit tests for theta_evolution_processor module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np
from cmb.theta_data_loader import ThetaEvolution
from cmb.theta_evolution_processor import (
    ThetaEvolutionProcessor,
    process_evolution_data,
)


class TestThetaEvolutionProcessor:
    """Tests for ThetaEvolutionProcessor class."""

    def test_init_valid_evolution(self):
        """Test initializing processor with valid evolution data."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])
        metadata = {"test": "data"}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)

        assert processor.evolution == evolution
        assert processor._omega_min_interp is None
        assert processor._omega_macro_interp is None
        assert processor._time_min == 0.0
        assert processor._time_max == 4.0

    def test_init_invalid_evolution(self):
        """Test initializing processor with invalid evolution data."""
        times = np.array([0.0, 1.0])
        omega_min = np.array([-1.0e10, 1.1e10])  # Negative value
        omega_macro = np.array([10.0e10, 10.1e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="All omega_min values must be positive"):
            ThetaEvolutionProcessor(evolution)

    def test_process_creates_interpolators(self):
        """Test that process() creates interpolation functions."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)
        processor.process()

        assert processor._omega_min_interp is not None
        assert processor._omega_macro_interp is not None
        assert processor._omega_min_rate_interp is not None
        assert processor._omega_macro_rate_interp is not None

    def test_process_handles_unsorted_times(self):
        """Test that process() handles unsorted time arrays."""
        # Create unsorted times
        times = np.array([2.0, 0.0, 4.0, 1.0, 3.0])
        omega_min = np.array([1.2e10, 1.0e10, 1.4e10, 1.1e10, 1.3e10])
        omega_macro = np.array([10.2e10, 10.0e10, 10.4e10, 10.1e10, 10.3e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)
        processor.process()

        # Should work correctly after sorting
        assert processor._omega_min_interp is not None
        assert processor._omega_macro_interp is not None

    def test_get_omega_min_at_data_points(self):
        """Test get_omega_min() returns exact values at data points."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)
        processor.process()

        # Test at data points (should be exact)
        for t, om in zip(times, omega_min):
            result = processor.get_omega_min(t)
            assert abs(result - om) < 1e-6

    def test_get_omega_min_interpolated(self):
        """Test get_omega_min() interpolates correctly between data points."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)
        processor.process()

        # Test interpolation at midpoint
        result = processor.get_omega_min(0.5)
        expected = 1.05e10  # Linear interpolation between 1.0e10 and 1.1e10
        # Allow some tolerance for cubic interpolation
        assert abs(result - expected) < 0.01e10

    def test_get_omega_min_out_of_range(self):
        """Test get_omega_min() raises ValueError for out-of-range times."""
        times = np.array([0.0, 1.0, 2.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)
        processor.process()

        with pytest.raises(ValueError, match="Time .* is out of range"):
            processor.get_omega_min(-1.0)

        with pytest.raises(ValueError, match="Time .* is out of range"):
            processor.get_omega_min(3.0)

    def test_get_omega_min_not_processed(self):
        """Test get_omega_min() raises ValueError if process() not called."""
        times = np.array([0.0, 1.0, 2.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)

        with pytest.raises(ValueError, match="Processor not initialized"):
            processor.get_omega_min(1.0)

    def test_get_omega_macro_at_data_points(self):
        """Test get_omega_macro() returns exact values at data points."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)
        processor.process()

        # Test at data points (should be exact)
        for t, om in zip(times, omega_macro):
            result = processor.get_omega_macro(t)
            assert abs(result - om) < 1e-6

    def test_get_omega_macro_interpolated(self):
        """Test get_omega_macro() interpolates between data points."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)
        processor.process()

        # Test interpolation at midpoint
        result = processor.get_omega_macro(0.5)
        expected = 10.05e10  # Linear interpolation between 10.0e10 and 10.1e10
        # Allow some tolerance for cubic interpolation
        assert abs(result - expected) < 0.01e10

    def test_get_omega_macro_out_of_range(self):
        """Test get_omega_macro() raises ValueError for out-of-range times."""
        times = np.array([0.0, 1.0, 2.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)
        processor.process()

        with pytest.raises(ValueError, match="Time .* is out of range"):
            processor.get_omega_macro(-1.0)

    def test_get_evolution_rate_min(self):
        """Test get_evolution_rate_min() calculates d(ω_min)/dt correctly."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)
        processor.process()

        # Calculate expected rate (constant rate = 0.1e10 per unit time)
        rate = processor.get_evolution_rate_min(1.5)
        expected_rate = 0.1e10
        # Allow tolerance for numerical differentiation
        assert abs(rate - expected_rate) < 0.01e10

    def test_get_evolution_rate_min_not_processed(self):
        """Test get_evolution_rate_min() raises ValueError if not processed."""
        times = np.array([0.0, 1.0, 2.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)

        with pytest.raises(ValueError, match="Processor not initialized"):
            processor.get_evolution_rate_min(1.0)

    def test_get_evolution_rate_macro(self):
        """Test get_evolution_rate_macro() calculates d(ω_macro)/dt."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)
        processor.process()

        # Calculate expected rate (constant rate = 0.1e10 per unit time)
        rate = processor.get_evolution_rate_macro(1.5)
        expected_rate = 0.1e10
        # Allow tolerance for numerical differentiation
        assert abs(rate - expected_rate) < 0.01e10

    def test_get_evolution_rate_macro_not_processed(self):
        """Test get_evolution_rate_macro() raises ValueError."""
        times = np.array([0.0, 1.0, 2.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)

        with pytest.raises(ValueError, match="Processor not initialized"):
            processor.get_evolution_rate_macro(1.0)

    def test_validate_time_range_valid(self):
        """Test validate_time_range() returns True for valid times."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)

        # Test boundary conditions
        assert processor.validate_time_range(0.0) is True
        assert processor.validate_time_range(4.0) is True
        assert processor.validate_time_range(2.0) is True

    def test_validate_time_range_invalid(self):
        """Test validate_time_range() raises ValueError for invalid times."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = ThetaEvolutionProcessor(evolution)

        with pytest.raises(ValueError, match="Time .* is out of range"):
            processor.validate_time_range(-1.0)

        with pytest.raises(ValueError, match="Time .* is out of range"):
            processor.validate_time_range(5.0)


class TestProcessEvolutionData:
    """Tests for process_evolution_data function."""

    def test_process_evolution_data(self):
        """Test process_evolution_data() creates and processes processor."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = process_evolution_data(evolution)

        assert isinstance(processor, ThetaEvolutionProcessor)
        assert processor._omega_min_interp is not None
        assert processor._omega_macro_interp is not None

        # Test that it works
        result = processor.get_omega_min(1.0)
        assert abs(result - 1.1e10) < 1e-6

    def test_process_evolution_data_invalid(self):
        """Test process_evolution_data() raises ValueError for invalid data."""
        times = np.array([0.0, 1.0])
        omega_min = np.array([-1.0e10, 1.1e10])  # Invalid
        omega_macro = np.array([10.0e10, 10.1e10])
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        with pytest.raises(ValueError):
            process_evolution_data(evolution)
