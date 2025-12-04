"""
Θ-field evolution data processing for CMB verification project.

Processes temporal evolution data ω_min(t) and ω_macro(t) from
data/theta/evolution/ directory.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Callable
import numpy as np
from scipy.interpolate import interp1d
from cmb.theta_data_loader import ThetaEvolution, validate_evolution_data


class ThetaEvolutionProcessor:
    """
    Θ-field evolution data processor.

    Processes and provides interface for temporal evolution data
    ω_min(t) and ω_macro(t).
    """

    def __init__(self, evolution: ThetaEvolution):
        """
        Initialize evolution processor.

        Args:
            evolution: ThetaEvolution data instance

        Raises:
            ValueError: If evolution data is invalid
        """
        # Validate evolution data
        validate_evolution_data(evolution)

        self.evolution = evolution
        self._omega_min_interp: Optional[Callable] = None
        self._omega_macro_interp: Optional[Callable] = None
        self._omega_min_rate_interp: Optional[Callable] = None
        self._omega_macro_rate_interp: Optional[Callable] = None

        # Store time range
        self._time_min = float(np.min(evolution.times))
        self._time_max = float(np.max(evolution.times))

    def process(self) -> None:
        """
        Process evolution data and create interpolators.

        Creates interpolation functions for ω_min(t) and ω_macro(t)
        and their derivatives for evolution rate calculations.

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

        # Calculate derivatives for evolution rates
        # Use numerical differentiation with central differences
        dt = np.diff(times_sorted)
        domega_min_dt = np.diff(omega_min_sorted) / dt
        domega_macro_dt = np.diff(omega_macro_sorted) / dt

        # Create time array for derivatives (midpoints)
        times_mid = (times_sorted[:-1] + times_sorted[1:]) / 2.0

        # Create interpolation functions for rates
        # Use linear interpolation for derivatives
        self._omega_min_rate_interp = interp1d(
            times_mid,
            domega_min_dt,
            kind="linear",
            bounds_error=False,
            fill_value=(domega_min_dt[0], domega_min_dt[-1]),
        )

        self._omega_macro_rate_interp = interp1d(
            times_mid,
            domega_macro_dt,
            kind="linear",
            bounds_error=False,
            fill_value=(domega_macro_dt[0], domega_macro_dt[-1]),
        )

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


def process_evolution_data(
    evolution: ThetaEvolution,
) -> ThetaEvolutionProcessor:
    """
    Process Θ-field evolution data.

    Args:
        evolution: ThetaEvolution data instance

    Returns:
        ThetaEvolutionProcessor instance with processed data

    Raises:
        ValueError: If processing fails
    """
    processor = ThetaEvolutionProcessor(evolution)
    processor.process()
    return processor
