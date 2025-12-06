"""
CMB map validation module.

Validates reconstructed CMB maps by comparing with ACT DR6.02
observational data.

This is a facade class that coordinates:
- MapLoader: loads observed maps
- MapComparator: compares maps
- StructureValidator: validates arcmin-scale structures
- AmplitudeValidator: validates amplitude range

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from cmb.reconstruction.map_loader import MapLoader
from cmb.reconstruction.map_comparator import MapComparator
from cmb.reconstruction.structure_validator import StructureValidator
from cmb.reconstruction.amplitude_validator import AmplitudeValidator


class MapValidator:
    """
    CMB map validator (facade class).

    Validates reconstructed maps against ACT DR6.02 observations.
    Coordinates validation operations using helper classes.
    """

    def __init__(
        self, reconstructed_map: np.ndarray, observed_map_path: Path, nside: int
    ):
        """
        Initialize validator.

        Args:
            reconstructed_map: Reconstructed CMB map
            observed_map_path: Path to ACT DR6.02 map
            nside: HEALPix NSIDE parameter
        """
        self.reconstructed_map = reconstructed_map
        self.observed_map_path = observed_map_path
        self.nside = nside
        self.observed_map: Optional[np.ndarray] = None

        # Initialize helper classes
        self.map_loader = MapLoader(nside)
        self.map_comparator = MapComparator()
        self.structure_validator = StructureValidator(nside)
        self.amplitude_validator = AmplitudeValidator()

    def load_observed_map(self) -> None:
        """
        Load observed ACT DR6.02 map.

        Raises:
            FileNotFoundError: If map file doesn't exist
            ValueError: If map format is invalid
        """
        self.observed_map = self.map_loader.load_map(self.observed_map_path)

    def compare_maps(self) -> Dict[str, float]:
        """
        Compare reconstructed and observed maps.

        Returns:
            Dictionary with comparison metrics:
            - correlation: Correlation coefficient (Pearson)
            - mean_diff: Mean difference
            - std_diff: Standard deviation of difference
            - rms_diff: RMS difference

        Raises:
            ValueError: If maps have incompatible formats
        """
        if self.observed_map is None:
            self.load_observed_map()

        if self.observed_map is None:
            raise ValueError("Observed map is None")

        return self.map_comparator.compare(self.reconstructed_map, self.observed_map)

    def validate_arcmin_structures(
        self, scale_min: float = 2.0, scale_max: float = 5.0
    ) -> Dict[str, Any]:
        """
        Validate arcmin-scale structures (2-5′).

        Args:
            scale_min: Minimum scale in arcmin
            scale_max: Maximum scale in arcmin

        Returns:
            Dictionary with structure validation results:
            - structure_count: Number of structures found
            - position_matches: Number of matching positions
            - amplitude_matches: Number of matching amplitudes

        Raises:
            ValueError: If scale parameters are invalid
        """
        if self.observed_map is None:
            self.load_observed_map()

        if self.observed_map is None:
            raise ValueError("Observed map is None")

        return self.structure_validator.validate(
            self.reconstructed_map, self.observed_map, scale_min, scale_max
        )

    def validate_amplitude(
        self, min_microK: float = 20.0, max_microK: float = 30.0
    ) -> Dict[str, Any]:
        """
        Validate temperature fluctuation amplitude (20-30 μK).

        Args:
            min_microK: Minimum expected amplitude in μK
            max_microK: Maximum expected amplitude in μK

        Returns:
            Dictionary with amplitude validation results:
            - mean_amplitude: Mean amplitude
            - std_amplitude: Standard deviation
            - in_range_fraction: Fraction of pixels in range
            - validation_passed: Boolean validation result
        """
        return self.amplitude_validator.validate(
            self.reconstructed_map, min_microK, max_microK
        )

    def generate_validation_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive validation report.

        Args:
            output_path: Path to save report (optional)

        Returns:
            Validation report as string
        """
        # Run all validations
        comparison_metrics = self.compare_maps()
        structure_validation = self.validate_arcmin_structures()
        amplitude_validation = self.validate_amplitude()

        # Generate report
        report_lines = [
            "=" * 80,
            "CMB Map Validation Report",
            "=" * 80,
            "",
            "Comparison Metrics:",
            f"  Correlation coefficient: {comparison_metrics['correlation']:.4f}",
            f"  Mean difference: {comparison_metrics['mean_diff']:.4f} μK",
            f"  Std difference: {comparison_metrics['std_diff']:.4f} μK",
            f"  RMS difference: {comparison_metrics['rms_diff']:.4f} μK",
            "",
            "Arcmin-Scale Structure Validation (2-5′):",
            f"  Structure count: {structure_validation['structure_count']}",
            f"  Position matches: {structure_validation['position_matches']}",
            f"  Amplitude matches: {structure_validation['amplitude_matches']}",
            "",
            "Amplitude Validation (20-30 μK):",
            f"  Mean amplitude: {amplitude_validation['mean_amplitude']:.4f} μK",
            f"  Std amplitude: {amplitude_validation['std_amplitude']:.4f} μK",
            f"  In-range fraction: {amplitude_validation['in_range_fraction']:.4f}",
            f"  Validation passed: {amplitude_validation['validation_passed']}",
            "",
            "=" * 80,
        ]

        report = "\n".join(report_lines)

        # Save to file if requested
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)

        return report
