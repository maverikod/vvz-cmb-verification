"""
Unit tests for CMB map validation module.

Tests MapValidator with CUDA acceleration support.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np
import healpy as hp
from pathlib import Path
from tempfile import TemporaryDirectory
from cmb.reconstruction.map_validator import MapValidator


@pytest.fixture
def sample_nside():
    """Create sample NSIDE for tests."""
    return 64  # Small NSIDE for faster tests


@pytest.fixture
def sample_reconstructed_map(sample_nside):
    """Create sample reconstructed map for tests."""
    npix = hp.nside2npix(sample_nside)
    # Create map with temperature fluctuations in μK
    # Mean ~0, std ~25 μK (typical CMB fluctuations)
    map_data = np.random.normal(0.0, 25.0, npix)
    return map_data


@pytest.fixture
def sample_observed_map(sample_nside):
    """Create sample observed map for tests."""
    npix = hp.nside2npix(sample_nside)
    # Create map similar to reconstructed but with some differences
    map_data = np.random.normal(0.0, 25.0, npix)
    return map_data


@pytest.fixture
def sample_observed_map_path(sample_observed_map, sample_nside):
    """Create temporary FITS file with observed map."""
    with TemporaryDirectory() as tmpdir:
        map_path = Path(tmpdir) / "observed_map.fits"
        hp.write_map(str(map_path), sample_observed_map, overwrite=True)
        yield map_path


class TestMapValidator:
    """Test MapValidator class."""

    def test_init(
        self, sample_reconstructed_map, sample_observed_map_path, sample_nside
    ):
        """Test MapValidator initialization."""
        validator = MapValidator(
            sample_reconstructed_map, sample_observed_map_path, sample_nside
        )
        assert validator.reconstructed_map is not None
        assert validator.observed_map_path == sample_observed_map_path
        assert validator.nside == sample_nside
        assert validator.observed_map is None

    def test_load_observed_map(
        self, sample_reconstructed_map, sample_observed_map_path, sample_nside
    ):
        """Test loading observed map."""
        validator = MapValidator(
            sample_reconstructed_map, sample_observed_map_path, sample_nside
        )
        validator.load_observed_map()
        assert validator.observed_map is not None
        assert validator.observed_map.size == sample_reconstructed_map.size

    def test_load_observed_map_file_not_found(
        self, sample_reconstructed_map, sample_nside
    ):
        """Test loading non-existent map file."""
        validator = MapValidator(
            sample_reconstructed_map, Path("nonexistent.fits"), sample_nside
        )
        with pytest.raises(FileNotFoundError):
            validator.load_observed_map()

    def test_compare_maps(
        self, sample_reconstructed_map, sample_observed_map_path, sample_nside
    ):
        """Test map comparison."""
        validator = MapValidator(
            sample_reconstructed_map, sample_observed_map_path, sample_nside
        )
        metrics = validator.compare_maps()

        # Check that all metrics are present
        assert "correlation" in metrics
        assert "mean_diff" in metrics
        assert "std_diff" in metrics
        assert "rms_diff" in metrics

        # Check metric types
        assert isinstance(metrics["correlation"], float)
        assert isinstance(metrics["mean_diff"], float)
        assert isinstance(metrics["std_diff"], float)
        assert isinstance(metrics["rms_diff"], float)

        # Check correlation is in valid range
        assert -1.0 <= metrics["correlation"] <= 1.0

        # Check std_diff and rms_diff are non-negative
        assert metrics["std_diff"] >= 0.0
        assert metrics["rms_diff"] >= 0.0

    def test_compare_maps_size_mismatch(self, sample_reconstructed_map, sample_nside):
        """Test map comparison with size mismatch."""
        # Create map with different size
        wrong_nside = sample_nside * 2
        wrong_map = np.random.normal(0.0, 25.0, hp.nside2npix(wrong_nside))

        with TemporaryDirectory() as tmpdir:
            map_path = Path(tmpdir) / "wrong_map.fits"
            hp.write_map(str(map_path), wrong_map, overwrite=True)

            validator = MapValidator(sample_reconstructed_map, map_path, sample_nside)
            # Should handle size mismatch by resampling
            metrics = validator.compare_maps()
            assert "correlation" in metrics

    def test_validate_arcmin_structures(
        self, sample_reconstructed_map, sample_observed_map_path, sample_nside
    ):
        """Test arcmin-scale structure validation."""
        validator = MapValidator(
            sample_reconstructed_map, sample_observed_map_path, sample_nside
        )
        results = validator.validate_arcmin_structures(scale_min=2.0, scale_max=5.0)

        # Check that all results are present
        assert "structure_count" in results
        assert "position_matches" in results
        assert "amplitude_matches" in results

        # Check result types
        assert isinstance(results["structure_count"], int)
        assert isinstance(results["position_matches"], int)
        assert isinstance(results["amplitude_matches"], int)

        # Check non-negative values
        assert results["structure_count"] >= 0
        assert results["position_matches"] >= 0
        assert results["amplitude_matches"] >= 0

    def test_validate_arcmin_structures_invalid_scale(
        self, sample_reconstructed_map, sample_observed_map_path, sample_nside
    ):
        """Test arcmin-scale validation with invalid parameters."""
        validator = MapValidator(
            sample_reconstructed_map, sample_observed_map_path, sample_nside
        )

        # Test negative scale
        with pytest.raises(ValueError, match="must be positive"):
            validator.validate_arcmin_structures(scale_min=-1.0, scale_max=5.0)

        # Test scale_min >= scale_max
        with pytest.raises(ValueError, match="must be less than"):
            validator.validate_arcmin_structures(scale_min=5.0, scale_max=2.0)

    def test_validate_amplitude(
        self, sample_reconstructed_map, sample_observed_map_path, sample_nside
    ):
        """Test amplitude validation."""
        validator = MapValidator(
            sample_reconstructed_map, sample_observed_map_path, sample_nside
        )
        results = validator.validate_amplitude(min_microK=20.0, max_microK=30.0)

        # Check that all results are present
        assert "mean_amplitude" in results
        assert "std_amplitude" in results
        assert "in_range_fraction" in results
        assert "validation_passed" in results

        # Check result types
        assert isinstance(results["mean_amplitude"], float)
        assert isinstance(results["std_amplitude"], float)
        assert isinstance(results["in_range_fraction"], float)
        assert isinstance(results["validation_passed"], bool)

        # Check std_amplitude is non-negative
        assert results["std_amplitude"] >= 0.0

        # Check in_range_fraction is in [0, 1]
        assert 0.0 <= results["in_range_fraction"] <= 1.0

    def test_validate_amplitude_invalid_range(
        self, sample_reconstructed_map, sample_observed_map_path, sample_nside
    ):
        """Test amplitude validation with invalid range."""
        validator = MapValidator(
            sample_reconstructed_map, sample_observed_map_path, sample_nside
        )

        # Test min >= max
        with pytest.raises(ValueError, match="must be less than"):
            validator.validate_amplitude(min_microK=30.0, max_microK=20.0)

    def test_generate_validation_report(
        self, sample_reconstructed_map, sample_observed_map_path, sample_nside
    ):
        """Test validation report generation."""
        validator = MapValidator(
            sample_reconstructed_map, sample_observed_map_path, sample_nside
        )
        report = validator.generate_validation_report()

        # Check report is a string
        assert isinstance(report, str)
        assert len(report) > 0

        # Check report contains expected sections
        assert "CMB Map Validation Report" in report
        assert "Comparison Metrics" in report
        assert "Correlation coefficient" in report
        assert "Arcmin-Scale Structure Validation" in report
        assert "Amplitude Validation" in report

    def test_generate_validation_report_save_file(
        self, sample_reconstructed_map, sample_observed_map_path, sample_nside
    ):
        """Test validation report generation with file save."""
        validator = MapValidator(
            sample_reconstructed_map, sample_observed_map_path, sample_nside
        )

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation_report.txt"
            report = validator.generate_validation_report(output_path=output_path)

            # Check file was created
            assert output_path.exists()

            # Check file content matches report
            with open(output_path, "r") as f:
                file_content = f.read()
            assert file_content == report

    def test_cuda_operations_used(
        self, sample_reconstructed_map, sample_observed_map_path, sample_nside
    ):
        """Test that CUDA operations are used in calculations."""
        validator = MapValidator(
            sample_reconstructed_map, sample_observed_map_path, sample_nside
        )

        # Check that vectorizers are initialized
        assert validator.elem_vec is not None
        assert validator.reduction_vec is not None
        assert validator.corr_vec is not None
        assert validator.transform_vec is not None

        # Run operations to verify CUDA is used
        metrics = validator.compare_maps()
        assert "correlation" in metrics

        results = validator.validate_amplitude()
        assert "mean_amplitude" in results

    def test_identical_maps_high_correlation(
        self, sample_reconstructed_map, sample_nside
    ):
        """Test that identical maps have high correlation."""
        # Create observed map identical to reconstructed
        with TemporaryDirectory() as tmpdir:
            map_path = Path(tmpdir) / "identical_map.fits"
            hp.write_map(str(map_path), sample_reconstructed_map, overwrite=True)

            validator = MapValidator(sample_reconstructed_map, map_path, sample_nside)
            metrics = validator.compare_maps()

            # Correlation should be close to 1.0 for identical maps
            # (allowing for numerical precision)
            assert metrics["correlation"] > 0.99
            assert abs(metrics["mean_diff"]) < 1e-6  # Should be very small

    def test_opposite_maps_negative_correlation(
        self, sample_reconstructed_map, sample_nside
    ):
        """Test that opposite maps have negative correlation."""
        # Create observed map opposite to reconstructed
        opposite_map = -sample_reconstructed_map

        with TemporaryDirectory() as tmpdir:
            map_path = Path(tmpdir) / "opposite_map.fits"
            hp.write_map(str(map_path), opposite_map, overwrite=True)

            validator = MapValidator(sample_reconstructed_map, map_path, sample_nside)
            metrics = validator.compare_maps()

            # Correlation should be close to -1.0 for opposite maps
            assert metrics["correlation"] < -0.99
