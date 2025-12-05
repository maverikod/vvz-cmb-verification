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


class TestEvolutionStatistics:
    """Tests for evolution statistics functionality."""

    def test_get_evolution_statistics(self):
        """Test get_evolution_statistics() returns correct statistics."""
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

        stats = processor.get_evolution_statistics()

        assert "omega_min" in stats
        assert "omega_macro" in stats
        assert "evolution_rates" in stats
        assert "time_coverage" in stats

        # Check omega_min statistics
        assert "mean" in stats["omega_min"]
        assert "std" in stats["omega_min"]
        assert "min" in stats["omega_min"]
        assert "max" in stats["omega_min"]

        # Check that mean is reasonable
        assert 1.0e10 < stats["omega_min"]["mean"] < 1.5e10

    def test_get_evolution_statistics_not_processed(self):
        """Test get_evolution_statistics() raises error if not processed."""
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
            processor.get_evolution_statistics()


class TestTimeCoverageGaps:
    """Tests for time coverage gap detection."""

    def test_check_time_coverage_gaps_no_gaps(self):
        """Test check_time_coverage_gaps() with no gaps."""
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

        gaps = processor.check_time_coverage_gaps()
        assert len(gaps) == 0

    def test_check_time_coverage_gaps_with_gaps(self):
        """Test check_time_coverage_gaps() detects gaps."""
        # Create data with a large gap
        # Gap between 2.0 and 10.0
        times = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
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

        gaps = processor.check_time_coverage_gaps()
        assert len(gaps) > 0
        # Check that gap is between 2.0 and 10.0
        assert any(gap[0] == 2.0 and gap[1] == 10.0 for gap in gaps)

    def test_check_time_coverage_gaps_not_processed(self):
        """Test check_time_coverage_gaps() raises error if not processed."""
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
            processor.check_time_coverage_gaps()


class TestQualityReport:
    """Tests for quality report generation."""

    def test_generate_quality_report(self):
        """Test generate_quality_report() creates report."""
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

        report = processor.generate_quality_report()

        assert "quality_issues" in report
        assert "statistics" in report
        assert "time_coverage" in report
        assert "gaps" in report
        assert "warnings" in report

        assert isinstance(report["quality_issues"], list)
        assert isinstance(report["gaps"], list)
        assert isinstance(report["warnings"], list)

    def test_generate_quality_report_with_gaps(self):
        """Test generate_quality_report() includes gap warnings."""
        times = np.array([0.0, 1.0, 2.0, 10.0, 11.0])  # Has gap
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

        report = processor.generate_quality_report()

        # Should have warnings about gaps
        assert len(report["gaps"]) > 0
        assert len(report["warnings"]) > 0

    def test_generate_quality_report_includes_completeness(self):
        """Test generate_quality_report() includes completeness check."""
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

        report = processor.generate_quality_report()

        assert "completeness" in report
        assert "is_complete" in report["completeness"]
        assert "coverage_ratio" in report["completeness"]

    def test_generate_quality_report_includes_physical_constraints(self):
        """Test generate_quality_report() includes physical constraints."""
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

        report = processor.generate_quality_report()

        assert "physical_constraints" in report
        assert "omega_min_lt_omega_macro" in report["physical_constraints"]
        constraint = report["physical_constraints"]["omega_min_lt_omega_macro"]
        assert "valid" in constraint
        assert "violations" in constraint
        # Use == instead of is for NumPy bool
        assert constraint["valid"] == True  # noqa: E712
        assert constraint["violations"] == 0


class TestTimeArrayCompleteness:
    """Tests for time array completeness verification."""

    def test_verify_time_array_completeness_complete(self):
        """Test verify_time_array_completeness() with complete array."""
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

        completeness = processor.verify_time_array_completeness()

        assert completeness["is_complete"] is True
        assert len(completeness["missing_points"]) == 0
        assert completeness["coverage_ratio"] > 0.9

    def test_verify_time_array_completeness_incomplete(self):
        """Test verify_time_array_completeness() with incomplete array."""
        # Create data with missing points (gap between 2.0 and 5.0)
        times = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
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

        completeness = processor.verify_time_array_completeness()

        # Should detect missing points
        assert completeness["is_complete"] is False
        assert len(completeness["missing_points"]) > 0
        assert completeness["coverage_ratio"] < 1.0

    def test_verify_time_array_completeness_with_expected_interval(self):
        """Test verify_time_array_completeness() with specified interval."""
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

        completeness = processor.verify_time_array_completeness(expected_interval=1.0)

        assert completeness["is_complete"] is True
        assert completeness["coverage_ratio"] > 0.9

    def test_verify_time_array_completeness_not_processed(self):
        """Test verify_time_array_completeness() raises error."""
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
            processor.verify_time_array_completeness()


class TestValidateAgainstConfig:
    """Tests for validate_against_config functionality."""

    def test_validate_against_config_no_config_requirements(self):
        """Test validate_against_config() with no config requirements."""
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

        # Should pass if no config requirements
        result = processor.validate_against_config()
        assert result is True

    def test_validate_against_config_with_valid_range(self):
        """Test validate_against_config() with valid time range."""
        from config.settings import Config  # noqa: E501

        # Create data that fully covers the required range
        times = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10, 1.5e10, 1.6e10])
        omega_macro = np.array(
            [10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10, 10.5e10, 10.6e10]
        )
        metadata = {}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        # Create config with evolution requirements
        config = Config._load_defaults()
        if config.cmb_config is None:
            config.cmb_config = {}
        config.cmb_config["evolution"] = {
            "time_min": -1.0,  # Data starts at -1.0, requirement is -1.0
            "time_max": 5.0,  # Data ends at 5.0, requirement is 5.0
        }

        processor = ThetaEvolutionProcessor(evolution, config=config)

        # Should pass: data fully covers required range
        result = processor.validate_against_config()
        assert result is True

    def test_validate_against_config_with_invalid_min(self):
        """Test validate_against_config() raises error for invalid min time."""
        from config.settings import Config

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

        # Create config with evolution requirements
        # Set time_max to None or a value covered by data to test only time_min
        config = Config._load_defaults()
        if config.cmb_config is None:
            config.cmb_config = {}
        config.cmb_config["evolution"] = {
            "time_min": -1.0,  # Data starts at 0.0, but requirement is -1.0
            "time_max": None,  # No max requirement to isolate min test
        }

        processor = ThetaEvolutionProcessor(evolution, config=config)

        # Should raise error: data doesn't start early enough
        with pytest.raises(
            ValueError,
            match="Time coverage starts at 0.0, but config requires -1.0",
        ):
            processor.validate_against_config()

    def test_validate_against_config_with_invalid_max(self):
        """Test validate_against_config() raises error for invalid max time."""
        from config.settings import Config

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

        # Create config with evolution requirements
        # Set time_min to None or a value covered by data to test only time_max
        config = Config._load_defaults()
        if config.cmb_config is None:
            config.cmb_config = {}
        config.cmb_config["evolution"] = {
            "time_min": None,  # No min requirement to isolate max test
            "time_max": 5.0,  # Data ends at 4.0, but requirement is 5.0
        }

        processor = ThetaEvolutionProcessor(evolution, config=config)

        # Should raise error: data doesn't extend far enough
        with pytest.raises(ValueError, match="Time coverage ends at"):
            processor.validate_against_config()
