"""
Θ-field evolution data processing for CMB verification project.

Processes temporal evolution data ω_min(t) and ω_macro(t) from
data/theta/evolution/ directory.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np
from scipy.interpolate import interp1d
from cmb.theta_data_loader import ThetaEvolution, validate_evolution_data
from config.settings import Config


class ThetaEvolutionProcessor:
    """
    Θ-field evolution data processor.

    Processes and provides interface for temporal evolution data
    ω_min(t) and ω_macro(t).
    """

    def __init__(
        self, evolution: ThetaEvolution, config: Optional[Config] = None
    ):
        """
        Initialize evolution processor.

        Args:
            evolution: ThetaEvolution data instance
            config: Configuration instance (uses global if None)

        Raises:
            ValueError: If evolution data is invalid
        """
        # Validate evolution data
        validate_evolution_data(evolution)

        # Get config
        if config is None:
            try:
                # Try to get existing instance
                if Config._instance is not None:
                    config = Config._instance
                else:
                    # Load config if not initialized
                    config = Config.load()
            except Exception:
                # Fallback to defaults if loading fails
                config = Config._load_defaults()

        self.evolution = evolution
        self.config = config
        self._omega_min_interp: Optional[Callable] = None
        self._omega_macro_interp: Optional[Callable] = None
        self._omega_min_rate_interp: Optional[Callable] = None
        self._omega_macro_rate_interp: Optional[Callable] = None

        # Store time range
        self._time_min = float(np.min(evolution.times))
        self._time_max = float(np.max(evolution.times))

        # Statistics (computed during process())
        self._statistics: Optional[Dict[str, Any]] = None

        # Quality issues (warnings)
        self._quality_issues: List[str] = []

    def process(self) -> None:
        """
        Process evolution data and create interpolators.

        Creates interpolation functions for ω_min(t) and ω_macro(t)
        and their derivatives for evolution rate calculations.
        Also computes evolution statistics and checks data quality.

        Raises:
            ValueError: If processing fails
        """
        # Check that times are sorted (required for interpolation)
        if not np.all(np.diff(self.evolution.times) >= 0):
            # Sort if not already sorted
            sort_indices = np.argsort(self.evolution.times)
            times_sorted = self.evolution.times[sort_indices]
            omega_min_sorted = self.evolution.omega_min[sort_indices]
            omega_macro_sorted = self.evolution.omega_macro[sort_indices]
        else:
            times_sorted = self.evolution.times
            omega_min_sorted = self.evolution.omega_min
            omega_macro_sorted = self.evolution.omega_macro

        # Check for gaps in time coverage
        gaps = self._check_time_coverage_gaps(times_sorted)
        if gaps:
            self._quality_issues.append(
                f"Found {len(gaps)} gap(s) in time coverage"
            )

        # Create interpolation functions for values
        # Use cubic interpolation for smooth derivatives if enough points
        # Otherwise use linear interpolation
        n_points = len(times_sorted)
        if n_points >= 4:
            interp_kind = "cubic"
        elif n_points >= 2:
            interp_kind = "linear"
        else:
            raise ValueError(
                f"Need at least 2 data points for interpolation, "
                f"got {n_points}"
            )

        self._omega_min_interp = interp1d(
            times_sorted,
            omega_min_sorted,
            kind=interp_kind,
            bounds_error=True,
            fill_value=np.nan,
        )

        self._omega_macro_interp = interp1d(
            times_sorted,
            omega_macro_sorted,
            kind=interp_kind,
            bounds_error=True,
            fill_value=np.nan,
        )

        # Calculate derivatives for evolution rates using central differences
        # For interior points: (f[i+1] - f[i-1]) / (t[i+1] - t[i-1])
        # For boundaries: use forward/backward differences
        domega_min_dt = self._calculate_derivative_central(
            times_sorted, omega_min_sorted
        )
        domega_macro_dt = self._calculate_derivative_central(
            times_sorted, omega_macro_sorted
        )

        # Create time array for derivatives (same as input times for central)
        # For central differences, derivatives are at original time points
        times_deriv = times_sorted.copy()

        # Create interpolation functions for rates
        # Use linear interpolation for derivatives
        self._omega_min_rate_interp = interp1d(
            times_deriv,
            domega_min_dt,
            kind="linear",
            bounds_error=False,
            fill_value=(domega_min_dt[0], domega_min_dt[-1]),
        )

        self._omega_macro_rate_interp = interp1d(
            times_deriv,
            domega_macro_dt,
            kind="linear",
            bounds_error=False,
            fill_value=(domega_macro_dt[0], domega_macro_dt[-1]),
        )

        # Compute evolution statistics
        self._compute_statistics(
            times_sorted,
            omega_min_sorted,
            omega_macro_sorted,
            domega_min_dt,
            domega_macro_dt,
        )

    def _calculate_derivative_central(
        self, times: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """
        Calculate derivative using central differences.

        For interior points: (f[i+1] - f[i-1]) / (t[i+1] - t[i-1])
        For boundaries: forward/backward differences

        Args:
            times: Time array (must be sorted)
            values: Value array

        Returns:
            Derivative array
        """
        n = len(times)
        derivatives = np.zeros(n)

        if n == 1:
            return np.array([0.0])

        if n == 2:
            # Only two points: use forward difference
            dt = times[1] - times[0]
            if dt > 0:
                derivatives[0] = (values[1] - values[0]) / dt
                derivatives[1] = derivatives[0]  # Same for both
            return derivatives

        # Interior points: central differences
        for i in range(1, n - 1):
            dt = times[i + 1] - times[i - 1]
            if dt > 0:
                derivatives[i] = (values[i + 1] - values[i - 1]) / dt

        # Boundary points: forward/backward differences
        # First point: forward difference
        dt0 = times[1] - times[0]
        if dt0 > 0:
            derivatives[0] = (values[1] - values[0]) / dt0

        # Last point: backward difference
        dt_last = times[n - 1] - times[n - 2]
        if dt_last > 0:
            derivatives[n - 1] = (values[n - 1] - values[n - 2]) / dt_last

        return derivatives

    def _check_time_coverage_gaps(
        self, times: np.ndarray, max_gap_ratio: float = 5.0
    ) -> List[Tuple[float, float]]:
        """
        Check for gaps in time coverage.

        Args:
            times: Sorted time array
            max_gap_ratio: Maximum allowed gap ratio (gap / median_interval)

        Returns:
            List of (gap_start, gap_end) tuples for gaps exceeding threshold
        """
        if len(times) < 2:
            return []

        gaps = []
        intervals = np.diff(times)
        if len(intervals) == 0:
            return []

        median_interval = np.median(intervals)
        if median_interval <= 0:
            return []

        threshold = max_gap_ratio * median_interval

        for i, interval in enumerate(intervals):
            if interval > threshold:
                gaps.append((float(times[i]), float(times[i + 1])))

        return gaps

    def _compute_statistics(
        self,
        times: np.ndarray,
        omega_min: np.ndarray,
        omega_macro: np.ndarray,
        domega_min_dt: np.ndarray,
        domega_macro_dt: np.ndarray,
    ) -> None:
        """
        Compute evolution statistics.

        Args:
            times: Time array
            omega_min: ω_min values
            omega_macro: ω_macro values
            domega_min_dt: Evolution rates for ω_min
            domega_macro_dt: Evolution rates for ω_macro
        """
        self._statistics = {
            "omega_min": {
                "mean": float(np.mean(omega_min)),
                "std": float(np.std(omega_min)),
                "min": float(np.min(omega_min)),
                "max": float(np.max(omega_min)),
                "range": float(np.max(omega_min) - np.min(omega_min)),
            },
            "omega_macro": {
                "mean": float(np.mean(omega_macro)),
                "std": float(np.std(omega_macro)),
                "min": float(np.min(omega_macro)),
                "max": float(np.max(omega_macro)),
                "range": float(np.max(omega_macro) - np.min(omega_macro)),
            },
            "evolution_rates": {
                "omega_min_rate": {
                    "mean": float(np.mean(domega_min_dt)),
                    "std": float(np.std(domega_min_dt)),
                    "min": float(np.min(domega_min_dt)),
                    "max": float(np.max(domega_min_dt)),
                },
                "omega_macro_rate": {
                    "mean": float(np.mean(domega_macro_dt)),
                    "std": float(np.std(domega_macro_dt)),
                    "min": float(np.min(domega_macro_dt)),
                    "max": float(np.max(domega_macro_dt)),
                },
            },
            "time_coverage": {
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "span": float(np.max(times) - np.min(times)),
                "n_points": len(times),
                "mean_interval": float(np.mean(np.diff(times))),
            },
        }

    def get_omega_min(self, time: float) -> float:
        """
        Get ω_min value at given time.

        Args:
            time: Time value

        Returns:
            ω_min(t) value

        Raises:
            ValueError: If time is out of range or processor not initialized
        """
        if self._omega_min_interp is None:
            raise ValueError(
                "Processor not initialized. Call process() first."
            )

        self.validate_time_range(time)

        result = self._omega_min_interp(time)
        return float(result)

    def get_omega_macro(self, time: float) -> float:
        """
        Get ω_macro value at given time.

        Args:
            time: Time value

        Returns:
            ω_macro(t) value

        Raises:
            ValueError: If time is out of range or processor not initialized
        """
        if self._omega_macro_interp is None:
            raise ValueError(
                "Processor not initialized. Call process() first."
            )

        self.validate_time_range(time)

        result = self._omega_macro_interp(time)
        return float(result)

    def get_evolution_rate_min(self, time: float) -> float:
        """
        Calculate evolution rate d(ω_min)/dt at given time.

        Args:
            time: Time value

        Returns:
            Evolution rate

        Raises:
            ValueError: If time is out of range or processor not initialized
        """
        if self._omega_min_rate_interp is None:
            raise ValueError(
                "Processor not initialized. Call process() first."
            )

        self.validate_time_range(time)

        result = self._omega_min_rate_interp(time)
        return float(result)

    def get_evolution_rate_macro(self, time: float) -> float:
        """
        Calculate evolution rate d(ω_macro)/dt at given time.

        Args:
            time: Time value

        Returns:
            Evolution rate

        Raises:
            ValueError: If time is out of range or processor not initialized
        """
        if self._omega_macro_rate_interp is None:
            raise ValueError(
                "Processor not initialized. Call process() first."
            )

        self.validate_time_range(time)

        result = self._omega_macro_rate_interp(time)
        return float(result)

    def validate_time_range(self, time: float) -> bool:
        """
        Validate that time is within data range.

        Args:
            time: Time value to validate

        Returns:
            True if time is within range

        Raises:
            ValueError: If time is out of range
        """
        if time < self._time_min or time > self._time_max:
            raise ValueError(
                f"Time {time} is out of range "
                f"[{self._time_min}, {self._time_max}]"
            )
        return True

    def validate_against_config(self) -> bool:
        """
        Validate evolution data against configuration requirements.

        Checks if time coverage meets requirements from config.

        Returns:
            True if data meets configuration requirements

        Raises:
            ValueError: If data does not meet requirements
        """
        # Check if cmb_config has time range requirements
        if self.config.cmb_config and "evolution" in self.config.cmb_config:
            evolution_config = self.config.cmb_config["evolution"]
            time_min_req = evolution_config.get("time_min")
            time_max_req = evolution_config.get("time_max")

            if time_min_req is not None and self._time_min > time_min_req:
                raise ValueError(
                    f"Time coverage starts at {self._time_min}, "
                    f"but config requires {time_min_req}"
                )

            if time_max_req is not None and self._time_max < time_max_req:
                raise ValueError(
                    f"Time coverage ends at {self._time_max}, "
                    f"but config requires {time_max_req}"
                )

        return True

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """
        Get evolution statistics.

        Returns:
            Dictionary with statistics about evolution data

        Raises:
            ValueError: If processor not initialized
        """
        if self._statistics is None:
            raise ValueError(
                "Processor not initialized. Call process() first."
            )
        return self._statistics.copy()

    def check_time_coverage_gaps(
        self, max_gap_ratio: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """
        Check for gaps in time coverage.

        Args:
            max_gap_ratio: Maximum allowed gap ratio.
                          If None, uses default (5.0)

        Returns:
            List of (gap_start, gap_end) tuples for gaps exceeding threshold
        """
        if self._omega_min_interp is None:
            raise ValueError(
                "Processor not initialized. Call process() first."
            )

        if max_gap_ratio is None:
            max_gap_ratio = 5.0

        # Get sorted times from evolution data
        times = np.sort(self.evolution.times)
        return self._check_time_coverage_gaps(times, max_gap_ratio)

    def verify_time_array_completeness(
        self, expected_interval: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Verify time array completeness.

        Checks if time array has expected coverage and no missing points.

        Args:
            expected_interval: Expected time interval between points.
                             If None, uses mean interval from data.

        Returns:
            Dictionary with completeness check results:
            - is_complete: bool - Whether array appears complete
            - missing_points: List[float] - List of expected but missing times
            - coverage_ratio: float - Ratio of actual to expected points
        """
        if self._omega_min_interp is None:
            raise ValueError(
                "Processor not initialized. Call process() first."
            )

        times = np.sort(self.evolution.times)
        n_points = len(times)

        if n_points < 2:
            return {
                "is_complete": True,
                "missing_points": [],
                "coverage_ratio": 1.0,
            }

        # Calculate expected interval
        if expected_interval is None:
            intervals = np.diff(times)
            expected_interval = float(np.median(intervals))

        if expected_interval <= 0:
            return {
                "is_complete": False,
                "missing_points": [],
                "coverage_ratio": 0.0,
            }

        # Check for missing points
        missing_points = []

        for i in range(len(times) - 1):
            interval = times[i + 1] - times[i]
            # If interval is significantly larger than expected, check for gaps
            if interval > expected_interval * 1.5:
                # Calculate how many points might be missing
                n_expected = int(round(interval / expected_interval))
                if n_expected > 1:
                    # Generate expected times in the gap
                    for j in range(1, n_expected):
                        expected_time = times[i] + j * expected_interval
                        missing_points.append(float(expected_time))

        # Calculate coverage ratio
        time_span = times[-1] - times[0]
        if time_span > 0:
            expected_n_points = int(round(time_span / expected_interval)) + 1
            coverage_ratio = (
                n_points / expected_n_points if expected_n_points > 0 else 0.0
            )
        else:
            coverage_ratio = 1.0

        return {
            "is_complete": len(missing_points) == 0,
            "missing_points": missing_points,
            "coverage_ratio": float(coverage_ratio),
        }

    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate data quality report.

        Includes checks for:
        - Time coverage gaps
        - Time array completeness
        - Physical constraints (ω_min < ω_macro)
        - Data quality issues

        Returns:
            Dictionary with quality issues, warnings, and statistics
        """
        report: Dict[str, Any] = {
            "quality_issues": self._quality_issues.copy(),
            "statistics": None,
            "time_coverage": {
                "min": self._time_min,
                "max": self._time_max,
                "span": self._time_max - self._time_min,
            },
            "gaps": [],
            "completeness": {},
            "physical_constraints": {},
            "warnings": [],
        }

        # Add statistics if available
        if self._statistics is not None:
            report["statistics"] = self._statistics.copy()

        # Check for gaps
        gaps = self.check_time_coverage_gaps()
        report["gaps"] = gaps
        if gaps:
            report["warnings"].append(
                f"Found {len(gaps)} gap(s) in time coverage"
            )

        # Check time coverage completeness
        completeness = self.verify_time_array_completeness()
        report["completeness"] = completeness
        if not completeness["is_complete"]:
            n_missing = len(completeness["missing_points"])
            report["warnings"].append(
                f"Time array appears incomplete: {n_missing} expected "
                f"points may be missing (coverage ratio: "
                f"{completeness['coverage_ratio']:.2%})"
            )

        # Check physical constraints: ω_min < ω_macro
        times_sorted = np.sort(self.evolution.times)
        sort_indices = np.argsort(self.evolution.times)
        omega_min_sorted = self.evolution.omega_min[sort_indices]
        omega_macro_sorted = self.evolution.omega_macro[sort_indices]

        violations = np.sum(omega_min_sorted >= omega_macro_sorted)
        report["physical_constraints"] = {
            "omega_min_lt_omega_macro": {
                "valid": violations == 0,
                "violations": int(violations),
                "violation_ratio": (
                    float(violations / len(times_sorted))
                    if len(times_sorted) > 0
                    else 0.0
                ),
            }
        }

        if violations > 0:
            report["warnings"].append(
                f"Physical constraint violation: ω_min >= ω_macro at "
                f"{violations} point(s) ({violations/len(times_sorted):.2%})"
            )

        # Check time coverage completeness (interval validation)
        if self._statistics:
            mean_interval = self._statistics["time_coverage"]["mean_interval"]
            if mean_interval <= 0:
                report["warnings"].append("Invalid time intervals detected")

        return report


def process_evolution_data(
    evolution: ThetaEvolution, config: Optional[Config] = None
) -> ThetaEvolutionProcessor:
    """
    Process Θ-field evolution data.

    Args:
        evolution: ThetaEvolution data instance
        config: Configuration instance (uses global if None)

    Returns:
        ThetaEvolutionProcessor instance with processed data

    Raises:
        ValueError: If processing fails
    """
    processor = ThetaEvolutionProcessor(evolution, config)
    processor.process()
    return processor
