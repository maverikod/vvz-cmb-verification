"""
Integration tests for theta evolution processing.

Tests end-to-end evolution data processing workflow:
Load → Process → Query → Validate

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from pathlib import Path
import tempfile
import json
from cmb.theta_data_loader import ThetaEvolution, load_evolution_data
from cmb.theta_evolution_processor import (
    ThetaEvolutionProcessor,
    process_evolution_data,
)


class TestThetaEvolutionIntegration:
    """Integration tests for theta evolution processing."""

    def test_end_to_end_evolution_processing(self):
        """
        Test complete evolution processing workflow.

        Load data → Process → Query → Validate
        """
        # Create test evolution data
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10, 1.5e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10, 10.5e10])
        metadata = {"test": "integration", "source": "test_data"}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        # Create processor
        processor = ThetaEvolutionProcessor(evolution)

        # Process evolution data
        processor.process()

        # Query evolution values at various times
        test_times = [0.0, 1.5, 2.5, 3.0, 4.5, 5.0]

        for t in test_times:
            omega_min_val = processor.get_omega_min(t)
            omega_macro_val = processor.get_omega_macro(t)

            # Verify values are positive
            assert omega_min_val > 0
            assert omega_macro_val > 0

            # Verify values are reasonable (within expected range)
            assert 1.0e10 <= omega_min_val <= 1.6e10
            assert 10.0e10 <= omega_macro_val <= 10.6e10

        # Calculate evolution rates
        for t in [1.0, 2.0, 3.0, 4.0]:
            rate_min = processor.get_evolution_rate_min(t)
            rate_macro = processor.get_evolution_rate_macro(t)

            # Verify rates are reasonable
            # Expected rate is approximately 0.1e10 per unit time
            assert abs(rate_min - 0.1e10) < 0.05e10
            assert abs(rate_macro - 0.1e10) < 0.05e10

        # Validate all operations work correctly
        assert processor._omega_min_interp is not None
        assert processor._omega_macro_interp is not None
        assert processor._omega_min_rate_interp is not None
        assert processor._omega_macro_rate_interp is not None

    def test_process_evolution_data_function(self):
        """
        Test process_evolution_data() convenience function.

        Verify it creates and processes processor correctly.
        """
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

        # Use convenience function
        processor = process_evolution_data(evolution)

        # Verify processor is ready to use
        assert isinstance(processor, ThetaEvolutionProcessor)
        assert processor._omega_min_interp is not None
        assert processor._omega_macro_interp is not None

        # Test that it works
        result_min = processor.get_omega_min(1.0)
        result_macro = processor.get_omega_macro(1.0)

        assert abs(result_min - 1.1e10) < 1e-6
        assert abs(result_macro - 10.1e10) < 1e-6

        # Test interpolation
        result_min_interp = processor.get_omega_min(0.5)
        result_macro_interp = processor.get_omega_macro(0.5)

        # Should be between 1.0e10 and 1.1e10
        assert 1.0e10 < result_min_interp < 1.1e10
        assert 10.0e10 < result_macro_interp < 10.1e10

    def test_evolution_processing_with_json_file(self):
        """
        Test evolution processing with JSON file format.

        Creates temporary JSON file and tests loading and processing.
        """
        # Create test data
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])

        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_data = {
                "times": times.tolist(),
                "omega_min": omega_min.tolist(),
                "omega_macro": omega_macro.tolist(),
                "metadata": {"test": "json_file"},
            }
            json.dump(json_data, f)
            temp_path = Path(f.name)

        try:
            # Load evolution data from file
            evolution = load_evolution_data(temp_path)

            # Process evolution data
            processor = process_evolution_data(evolution)

            # Verify processing works
            assert processor._omega_min_interp is not None
            assert processor._omega_macro_interp is not None

            # Test queries
            result_min = processor.get_omega_min(1.0)
            result_macro = processor.get_omega_macro(1.0)

            assert abs(result_min - 1.1e10) < 1e-6
            assert abs(result_macro - 10.1e10) < 1e-6

            # Test interpolation
            result_min_interp = processor.get_omega_min(0.5)
            assert 1.0e10 < result_min_interp < 1.1e10

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    def test_evolution_processing_with_csv_file(self):
        """
        Test evolution processing with CSV file format.

        Creates temporary CSV file and tests loading and processing.
        """
        import csv

        # Create test data
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        omega_min = np.array([1.0e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        omega_macro = np.array([10.0e10, 10.1e10, 10.2e10, 10.3e10, 10.4e10])

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["time", "omega_min", "omega_macro"])
            for t, om_min, om_macro in zip(times, omega_min, omega_macro):
                writer.writerow([t, om_min, om_macro])
            temp_path = Path(f.name)

        try:
            # Load evolution data from file
            evolution = load_evolution_data(temp_path)

            # Process evolution data
            processor = process_evolution_data(evolution)

            # Verify processing works
            assert processor._omega_min_interp is not None
            assert processor._omega_macro_interp is not None

            # Test queries
            result_min = processor.get_omega_min(1.0)
            result_macro = processor.get_omega_macro(1.0)

            assert abs(result_min - 1.1e10) < 1e-6
            assert abs(result_macro - 10.1e10) < 1e-6

            # Test interpolation
            result_min_interp = processor.get_omega_min(0.5)
            assert 1.0e10 < result_min_interp < 1.1e10

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    def test_evolution_processing_realistic_time_range(self):
        """
        Test evolution processing with realistic time ranges and values.

        Uses time values typical for cosmological evolution.
        """
        # Create realistic time range (e.g., redshift 0 to 1100)
        # Time in some cosmological units
        times = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
        omega_min = np.array([1.0e10, 1.05e10, 1.1e10, 1.15e10, 1.2e10, 1.25e10])
        omega_macro = np.array(
            [10.0e10, 10.05e10, 10.1e10, 10.15e10, 10.2e10, 10.25e10]
        )
        metadata = {"time_unit": "cosmological", "source": "test"}

        evolution = ThetaEvolution(
            times=times,
            omega_min=omega_min,
            omega_macro=omega_macro,
            metadata=metadata,
        )

        processor = process_evolution_data(evolution)

        # Test queries at various times
        test_times = [0.0, 50.0, 150.0, 250.0, 350.0, 450.0, 500.0]

        for t in test_times:
            omega_min_val = processor.get_omega_min(t)
            omega_macro_val = processor.get_omega_macro(t)

            # Verify values are positive and reasonable
            assert omega_min_val > 0
            assert omega_macro_val > 0
            assert omega_min_val <= 1.3e10
            assert omega_macro_val <= 10.3e10

            # Calculate rates
            rate_min = processor.get_evolution_rate_min(t)
            rate_macro = processor.get_evolution_rate_macro(t)

            # Rates should be small positive values
            assert rate_min > 0
            assert rate_macro > 0
            assert rate_min < 1e9  # Reasonable upper bound
            assert rate_macro < 1e9

        # Verify interpolators are valid
        assert processor._omega_min_interp is not None
        assert processor._omega_macro_interp is not None
        assert processor._omega_min_rate_interp is not None
        assert processor._omega_macro_rate_interp is not None
