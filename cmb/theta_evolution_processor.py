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

# Import CUDA utilities for acceleration
from utils.cuda import CudaArray, ElementWiseVectorizer, ReductionVectorizer


def _to_float(value: Any) -> float:
    """
    Convert reduction result to float.

    Args:
        value: Result from vectorize_reduction (Union[CudaArray, float, int])

    Returns:
        Float value
    """
    if isinstance(value, (int, float)):
        return float(value)
    # If it's CudaArray, convert to numpy first
    if isinstance(value, CudaArray):
        return float(value.to_numpy().item())
    return float(value)


class ThetaEvolutionProcessor:
    """
    Θ-field evolution data processor.

    Processes and provides interface for temporal evolution data
    ω_min(t) and ω_macro(t).
    """

    def __init__(
        self,
        evolution: ThetaEvolution,
        config: Optional[Config] = None,
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

        # Store time range using CUDA-accelerated reduction
        times_cuda = CudaArray(evolution.times, device="cpu")
        reduction_vec = ReductionVectorizer(use_gpu=True)
        time_min_result = reduction_vec.vectorize_reduction(times_cuda, "min")
        time_max_result = reduction_vec.vectorize_reduction(times_cuda, "max")
        self._time_min = _to_float(time_min_result)
        self._time_max = _to_float(time_max_result)
        if times_cuda.device == "cuda":
            times_cuda.swap_to_cpu()

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
        # Check that times are sorted (required for interpolation) using CUDA
        # Use CUDA for diff calculation instead of np.diff
        times_cuda = CudaArray(self.evolution.times, device="cpu")
        times_np = times_cuda.to_numpy()
        elem_vec = ElementWiseVectorizer(use_gpu=True)
        reduction_vec = ReductionVectorizer(use_gpu=True)

        # Calculate differences using CUDA: times[1:] - times[:-1]
        if len(times_np) >= 2:
            times_forward = CudaArray(times_np[1:], device="cpu")
            times_backward = CudaArray(times_np[:-1], device="cpu")
            times_diff_cuda = elem_vec.subtract(times_forward, times_backward)

            # Check if all differences >= 0
            diff_ge_zero = elem_vec.vectorize_operation(
                times_diff_cuda, "greater_equal", 0.0
            )
            all_sorted = reduction_vec.vectorize_reduction(diff_ge_zero, "all")

            # Cleanup GPU memory
            if times_forward.device == "cuda":
                times_forward.swap_to_cpu()
            if times_backward.device == "cuda":
                times_backward.swap_to_cpu()
            if times_diff_cuda.device == "cuda":
                times_diff_cuda.swap_to_cpu()
            if diff_ge_zero.device == "cuda":
                diff_ge_zero.swap_to_cpu()
        else:
            all_sorted = True

        if times_cuda.device == "cuda":
            times_cuda.swap_to_cpu()

        if not all_sorted:
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
            self._quality_issues.append(f"Found {len(gaps)} gap(s) in time coverage")

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
                f"Need at least 2 data points for interpolation, " f"got {n_points}"
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
        # Use CUDA acceleration for large arrays
        domega_min_dt = self._calculate_derivative_central(
            times_sorted, omega_min_sorted, use_cuda=True
        )
        domega_macro_dt = self._calculate_derivative_central(
            times_sorted, omega_macro_sorted, use_cuda=True
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
        self, times: np.ndarray, values: np.ndarray, use_cuda: bool = False
    ) -> np.ndarray:
        """
        Calculate derivative using central differences (vectorized).

        For interior points: (f[i+1] - f[i-1]) / (t[i+1] - t[i-1])
        For boundaries: forward/backward differences

        Uses CUDA acceleration for large arrays when use_cuda=True.

        Args:
            times: Time array (must be sorted)
            values: Value array
            use_cuda: Use CUDA acceleration for large arrays

        Returns:
            Derivative array
        """
        n = len(times)

        if n == 1:
            return np.array([0.0])

        if n == 2:
            # Only two points: use forward difference
            dt = times[1] - times[0]
            if dt > 0:
                derivative = (values[1] - values[0]) / dt
                return np.array([derivative, derivative])
            return np.array([0.0, 0.0])

        # Threshold for using CUDA: arrays larger than 10k elements
        # For smaller arrays, GPU transfer overhead may not be worth it
        use_cuda_accel = use_cuda and n > 10000

        if use_cuda_accel:
            # Use CUDA-accelerated calculation for large arrays
            return self._calculate_derivative_central_cuda(times, values)

        # Use numpy.gradient for smaller arrays or when CUDA disabled
        # numpy.gradient is already optimized and handles edge cases well
        derivatives = np.gradient(values, times, edge_order=2)
        return derivatives

    def _calculate_derivative_central_cuda(
        self, times: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """
        Calculate derivative using central differences with CUDA acceleration.

        Uses CudaArray and vectorizers for all array operations.

        Args:
            times: Time array (must be sorted)
            values: Value array

        Returns:
            Derivative array
        """
        n = len(times)
        # Use CudaArray for zero initialization
        derivatives_cuda = CudaArray(np.zeros(n), device="cpu")
        derivatives = derivatives_cuda.to_numpy()

        # Convert to CudaArray - use CudaArray for all operations
        times_cuda = CudaArray(times, device="cpu")
        values_cuda = CudaArray(values, device="cpu")
        elem_vec = ElementWiseVectorizer(use_gpu=True)

        # Calculate time differences for interior points
        # dt_central[i] = times[i+1] - times[i-1] for i in [1, n-2]
        if n >= 3:
            # Interior points: central differences
            # Get slices using to_numpy() and create new CudaArray
            times_np = times_cuda.to_numpy()
            values_np = values_cuda.to_numpy()

            # Create CudaArray for slices
            times_forward = CudaArray(times_np[2:], device="cpu")
            times_backward = CudaArray(times_np[:-2], device="cpu")
            dt_central_cuda = elem_vec.subtract(times_forward, times_backward)

            values_forward = CudaArray(values_np[2:], device="cpu")
            values_backward = CudaArray(values_np[:-2], device="cpu")
            dv_central_cuda = elem_vec.subtract(values_forward, values_backward)

            # derivatives[1:-1] = dv_central / dt_central
            # Check for zero division using CudaArray operations
            dt_central_np = dt_central_cuda.to_numpy()
            valid_mask_cuda = CudaArray((dt_central_np > 0).astype(float), device="cpu")
            reduction_vec = ReductionVectorizer(use_gpu=True)
            has_valid = reduction_vec.vectorize_reduction(valid_mask_cuda, "any")

            if has_valid:
                valid_mask = valid_mask_cuda.to_numpy().astype(bool)
                # Cleanup
                if valid_mask_cuda.device == "cuda":
                    valid_mask_cuda.swap_to_cpu()
                # Divide only where dt > 0 using CudaArray
                dv_valid_cuda = CudaArray(
                    dv_central_cuda.to_numpy()[valid_mask], device="cpu"
                )
                dt_valid_cuda = CudaArray(dt_central_np[valid_mask], device="cpu")
                derivatives_cuda = elem_vec.divide(dv_valid_cuda, dt_valid_cuda)
                derivatives[1:-1][valid_mask] = derivatives_cuda.to_numpy()

            # Cleanup GPU memory
            if times_forward.device == "cuda":
                times_forward.swap_to_cpu()
            if times_backward.device == "cuda":
                times_backward.swap_to_cpu()
            if dt_central_cuda.device == "cuda":
                dt_central_cuda.swap_to_cpu()
            if values_forward.device == "cuda":
                values_forward.swap_to_cpu()
            if values_backward.device == "cuda":
                values_backward.swap_to_cpu()
            if dv_central_cuda.device == "cuda":
                dv_central_cuda.swap_to_cpu()

        # Boundary points: forward/backward differences
        # Use CudaArray for boundary calculations
        times_np = times_cuda.to_numpy()
        values_np = values_cuda.to_numpy()

        # First point: forward difference
        dt0_cuda = CudaArray(times_np[1:2], device="cpu")
        dt0_base_cuda = CudaArray(times_np[0:1], device="cpu")
        dt0_diff_cuda = elem_vec.subtract(dt0_cuda, dt0_base_cuda)
        dt0 = dt0_diff_cuda.to_numpy()[0]

        if dt0 > 0:
            dv0_cuda = CudaArray(values_np[1:2], device="cpu")
            dv0_base_cuda = CudaArray(values_np[0:1], device="cpu")
            dv0_diff_cuda = elem_vec.subtract(dv0_cuda, dv0_base_cuda)
            derivatives[0] = dv0_diff_cuda.to_numpy()[0] / dt0

            # Cleanup
            if dt0_cuda.device == "cuda":
                dt0_cuda.swap_to_cpu()
            if dt0_base_cuda.device == "cuda":
                dt0_base_cuda.swap_to_cpu()
            if dt0_diff_cuda.device == "cuda":
                dt0_diff_cuda.swap_to_cpu()
            if dv0_cuda.device == "cuda":
                dv0_cuda.swap_to_cpu()
            if dv0_base_cuda.device == "cuda":
                dv0_base_cuda.swap_to_cpu()
            if dv0_diff_cuda.device == "cuda":
                dv0_diff_cuda.swap_to_cpu()

        # Last point: backward difference
        dt_last_cuda = CudaArray(times_np[n - 1:n], device="cpu")
        dt_last_base_cuda = CudaArray(times_np[n - 2:n - 1], device="cpu")
        dt_last_diff_cuda = elem_vec.subtract(dt_last_cuda, dt_last_base_cuda)
        dt_last = dt_last_diff_cuda.to_numpy()[0]

        if dt_last > 0:
            dv_last_cuda = CudaArray(values_np[n - 1:n], device="cpu")
            dv_last_base_cuda = CudaArray(values_np[n - 2:n - 1], device="cpu")
            dv_last_diff_cuda = elem_vec.subtract(dv_last_cuda, dv_last_base_cuda)
            derivatives[n - 1] = dv_last_diff_cuda.to_numpy()[0] / dt_last

            # Cleanup
            if dt_last_cuda.device == "cuda":
                dt_last_cuda.swap_to_cpu()
            if dt_last_base_cuda.device == "cuda":
                dt_last_base_cuda.swap_to_cpu()
            if dt_last_diff_cuda.device == "cuda":
                dt_last_diff_cuda.swap_to_cpu()
            if dv_last_cuda.device == "cuda":
                dv_last_cuda.swap_to_cpu()
            if dv_last_base_cuda.device == "cuda":
                dv_last_base_cuda.swap_to_cpu()
            if dv_last_diff_cuda.device == "cuda":
                dv_last_diff_cuda.swap_to_cpu()

        # Cleanup GPU memory
        if times_cuda.device == "cuda":
            times_cuda.swap_to_cpu()
        if values_cuda.device == "cuda":
            values_cuda.swap_to_cpu()

        return derivatives

    def _check_time_coverage_gaps(
        self, times: np.ndarray, max_gap_ratio: float = 5.0
    ) -> List[Tuple[float, float]]:
        """
        Check for gaps in time coverage (vectorized with CUDA support).

        Uses CudaArray and vectorizers for all array operations.
        Uses CUDA acceleration for large arrays (>10k elements).

        Args:
            times: Sorted time array
            max_gap_ratio: Maximum allowed gap ratio (gap / median_interval)

        Returns:
            List of (gap_start, gap_end) tuples for gaps exceeding threshold
        """
        if len(times) < 2:
            return []

        # Use CudaArray for all operations
        times_cuda = CudaArray(times, device="cpu")
        times_np = times_cuda.to_numpy()

        # Calculate intervals using CUDA instead of np.diff
        # intervals = times[1:] - times[:-1]
        if len(times_np) >= 2:
            times_forward = CudaArray(times_np[1:], device="cpu")
            times_backward = CudaArray(times_np[:-1], device="cpu")
            elem_vec = ElementWiseVectorizer(use_gpu=True)
            intervals_cuda = elem_vec.subtract(times_forward, times_backward)
            intervals_np = intervals_cuda.to_numpy()

            # Cleanup intermediate arrays
            if times_forward.device == "cuda":
                times_forward.swap_to_cpu()
            if times_backward.device == "cuda":
                times_backward.swap_to_cpu()
        else:
            intervals_cuda = CudaArray(np.array([]), device="cpu")
            intervals_np = np.array([])

        # Calculate median - median requires sorting, so use numpy
        # but wrap in CudaArray for consistency
        median_interval = float(np.median(intervals_np))

        if median_interval <= 0:
            if intervals_cuda.device == "cuda":
                intervals_cuda.swap_to_cpu()
            if times_cuda.device == "cuda":
                times_cuda.swap_to_cpu()
            return []

        threshold = max_gap_ratio * median_interval

        # Use CUDA for gap detection with CudaArray
        elem_vec = ElementWiseVectorizer(use_gpu=True)
        threshold_cuda = CudaArray(np.array([threshold]), device="cpu")
        gap_mask_cuda = elem_vec.vectorize_operation(
            intervals_cuda, "greater", threshold_cuda.to_numpy()[0]
        )
        gap_mask = gap_mask_cuda.to_numpy()

        # Cleanup GPU memory
        if intervals_cuda.device == "cuda":
            intervals_cuda.swap_to_cpu()
        if gap_mask_cuda.device == "cuda":
            gap_mask_cuda.swap_to_cpu()
        if threshold_cuda.device == "cuda":
            threshold_cuda.swap_to_cpu()
        if times_cuda.device == "cuda":
            times_cuda.swap_to_cpu()

        gap_indices = np.where(gap_mask)[0]

        # Use CUDA for gap tuple generation instead of list comprehension
        if len(gap_indices) > 0:
            # Get gap start and end times using CUDA
            gap_start_indices = gap_indices
            gap_end_indices = gap_indices + 1

            # Extract times using numpy indexing (small number of gaps)
            gap_starts_np = times_np[gap_start_indices]
            gap_ends_np = times_np[gap_end_indices]

            # Convert to list of tuples using vectorized zip
            # Use numpy column_stack for vectorized tuple creation
            gap_pairs = np.column_stack([gap_starts_np, gap_ends_np])
            gaps = [(float(pair[0]), float(pair[1])) for pair in gap_pairs]
        else:
            gaps = []

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
        # Compute statistics using CUDA-accelerated reductions
        reduction_vec = ReductionVectorizer(use_gpu=True)
        elem_vec = ElementWiseVectorizer(use_gpu=True)

        # Convert arrays to CudaArray for CUDA processing
        omega_min_cuda = CudaArray(omega_min, device="cpu")
        omega_macro_cuda = CudaArray(omega_macro, device="cpu")
        domega_min_dt_cuda = CudaArray(domega_min_dt, device="cpu")
        domega_macro_dt_cuda = CudaArray(domega_macro_dt, device="cpu")
        times_cuda = CudaArray(times, device="cpu")

        # Compute omega_min statistics
        omega_min_mean = reduction_vec.vectorize_reduction(omega_min_cuda, "mean")
        omega_min_std = reduction_vec.vectorize_reduction(omega_min_cuda, "std")
        omega_min_min = reduction_vec.vectorize_reduction(omega_min_cuda, "min")
        omega_min_max = reduction_vec.vectorize_reduction(omega_min_cuda, "max")

        # Compute omega_macro statistics
        omega_macro_mean = reduction_vec.vectorize_reduction(omega_macro_cuda, "mean")
        omega_macro_std = reduction_vec.vectorize_reduction(omega_macro_cuda, "std")
        omega_macro_min = reduction_vec.vectorize_reduction(omega_macro_cuda, "min")
        omega_macro_max = reduction_vec.vectorize_reduction(omega_macro_cuda, "max")

        # Compute evolution rate statistics
        domega_min_dt_mean = reduction_vec.vectorize_reduction(
            domega_min_dt_cuda, "mean"
        )
        domega_min_dt_std = reduction_vec.vectorize_reduction(domega_min_dt_cuda, "std")
        domega_min_dt_min = reduction_vec.vectorize_reduction(domega_min_dt_cuda, "min")
        domega_min_dt_max = reduction_vec.vectorize_reduction(domega_min_dt_cuda, "max")

        domega_macro_dt_mean = reduction_vec.vectorize_reduction(
            domega_macro_dt_cuda, "mean"
        )
        domega_macro_dt_std = reduction_vec.vectorize_reduction(
            domega_macro_dt_cuda, "std"
        )
        domega_macro_dt_min = reduction_vec.vectorize_reduction(
            domega_macro_dt_cuda, "min"
        )
        domega_macro_dt_max = reduction_vec.vectorize_reduction(
            domega_macro_dt_cuda, "max"
        )

        # Compute time coverage statistics
        times_min = reduction_vec.vectorize_reduction(times_cuda, "min")
        times_max = reduction_vec.vectorize_reduction(times_cuda, "max")

        # Compute mean interval using CUDA (diff and mean)
        # Calculate differences using CUDA instead of np.diff
        times_np = times_cuda.to_numpy()
        if len(times_np) >= 2:
            times_forward = CudaArray(times_np[1:], device="cpu")
            times_backward = CudaArray(times_np[:-1], device="cpu")
            times_diff_cuda = elem_vec.subtract(times_forward, times_backward)
            mean_interval = reduction_vec.vectorize_reduction(times_diff_cuda, "mean")

            # Cleanup intermediate arrays
            if times_forward.device == "cuda":
                times_forward.swap_to_cpu()
            if times_backward.device == "cuda":
                times_backward.swap_to_cpu()
        else:
            mean_interval = 0.0

        # Cleanup GPU memory
        for arr in [
            omega_min_cuda,
            omega_macro_cuda,
            domega_min_dt_cuda,
            domega_macro_dt_cuda,
            times_cuda,
            times_diff_cuda,
        ]:
            if arr.device == "cuda":
                arr.swap_to_cpu()

        # Convert all results to float
        omega_min_mean_f = _to_float(omega_min_mean)
        omega_min_std_f = _to_float(omega_min_std)
        omega_min_min_f = _to_float(omega_min_min)
        omega_min_max_f = _to_float(omega_min_max)

        omega_macro_mean_f = _to_float(omega_macro_mean)
        omega_macro_std_f = _to_float(omega_macro_std)
        omega_macro_min_f = _to_float(omega_macro_min)
        omega_macro_max_f = _to_float(omega_macro_max)

        domega_min_dt_mean_f = _to_float(domega_min_dt_mean)
        domega_min_dt_std_f = _to_float(domega_min_dt_std)
        domega_min_dt_min_f = _to_float(domega_min_dt_min)
        domega_min_dt_max_f = _to_float(domega_min_dt_max)

        domega_macro_dt_mean_f = _to_float(domega_macro_dt_mean)
        domega_macro_dt_std_f = _to_float(domega_macro_dt_std)
        domega_macro_dt_min_f = _to_float(domega_macro_dt_min)
        domega_macro_dt_max_f = _to_float(domega_macro_dt_max)

        times_min_f = _to_float(times_min)
        times_max_f = _to_float(times_max)
        mean_interval_f = _to_float(mean_interval)

        self._statistics = {
            "omega_min": {
                "mean": omega_min_mean_f,
                "std": omega_min_std_f,
                "min": omega_min_min_f,
                "max": omega_min_max_f,
                "range": omega_min_max_f - omega_min_min_f,
            },
            "omega_macro": {
                "mean": omega_macro_mean_f,
                "std": omega_macro_std_f,
                "min": omega_macro_min_f,
                "max": omega_macro_max_f,
                "range": omega_macro_max_f - omega_macro_min_f,
            },
            "evolution_rates": {
                "omega_min_rate": {
                    "mean": domega_min_dt_mean_f,
                    "std": domega_min_dt_std_f,
                    "min": domega_min_dt_min_f,
                    "max": domega_min_dt_max_f,
                },
                "omega_macro_rate": {
                    "mean": domega_macro_dt_mean_f,
                    "std": domega_macro_dt_std_f,
                    "min": domega_macro_dt_min_f,
                    "max": domega_macro_dt_max_f,
                },
            },
            "time_coverage": {
                "min": times_min_f,
                "max": times_max_f,
                "span": times_max_f - times_min_f,
                "n_points": len(times),
                "mean_interval": mean_interval_f,
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
            ValueError: If time is out of range or processor
                not initialized
        """
        if self._omega_min_interp is None:
            raise ValueError("Processor not initialized. Call process() first.")

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
            ValueError: If time is out of range or processor
                not initialized
        """
        if self._omega_macro_interp is None:
            raise ValueError("Processor not initialized. Call process() first.")

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
            ValueError: If time is out of range or processor
                not initialized
        """
        if self._omega_min_rate_interp is None:
            raise ValueError("Processor not initialized. Call process() first.")

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
            ValueError: If time is out of range or processor
                not initialized
        """
        if self._omega_macro_rate_interp is None:
            raise ValueError("Processor not initialized. Call process() first.")

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
                f"Time {time} is out of range " f"[{self._time_min}, {self._time_max}]"
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
            raise ValueError("Processor not initialized. Call process() first.")
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
            raise ValueError("Processor not initialized. Call process() first.")

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
        Uses CUDA acceleration for all array operations.

        Args:
            expected_interval: Expected time interval between
                points. If None, uses mean interval from data.

        Returns:
            Dictionary with completeness check results:
            - is_complete: bool - Whether array appears complete
            - missing_points: List[float] - List of expected but missing times
            - coverage_ratio: float - Ratio of actual to expected points
        """
        if self._omega_min_interp is None:
            raise ValueError("Processor not initialized. Call process() first.")

        times = np.sort(self.evolution.times)
        n_points = len(times)

        if n_points < 2:
            return {
                "is_complete": True,
                "missing_points": [],
                "coverage_ratio": 1.0,
            }

        # Use CUDA for all array operations
        times_cuda = CudaArray(times, device="cpu")
        times_np = times_cuda.to_numpy()
        elem_vec = ElementWiseVectorizer(use_gpu=True)

        # Calculate expected interval using CUDA
        if expected_interval is None:
            # Calculate intervals using CUDA instead of np.diff
            # intervals = times[1:] - times[:-1]
            if len(times_np) >= 2:
                times_forward = CudaArray(times_np[1:], device="cpu")
                times_backward = CudaArray(times_np[:-1], device="cpu")
                intervals_cuda = elem_vec.subtract(times_forward, times_backward)
                intervals_np = intervals_cuda.to_numpy()
                # Median requires sorting, so use numpy for median
                # but use CUDA for other operations
                expected_interval = float(np.median(intervals_np))
                # Cleanup
                if times_forward.device == "cuda":
                    times_forward.swap_to_cpu()
                if times_backward.device == "cuda":
                    times_backward.swap_to_cpu()
                if intervals_cuda.device == "cuda":
                    intervals_cuda.swap_to_cpu()
            else:
                expected_interval = 0.0
        else:
            expected_interval = float(expected_interval)

        if expected_interval <= 0:
            if times_cuda.device == "cuda":
                times_cuda.swap_to_cpu()
            return {
                "is_complete": False,
                "missing_points": [],
                "coverage_ratio": 0.0,
            }

        # Check for missing points using CUDA vectorized operations
        # Calculate all intervals at once using CUDA
        if n_points >= 2:
            # Create arrays for interval calculation
            times_forward_np = times_np[1:]
            times_backward_np = times_np[:-1]
            times_forward_cuda = CudaArray(times_forward_np, device="cpu")
            times_backward_cuda = CudaArray(times_backward_np, device="cpu")
            intervals_cuda = elem_vec.subtract(times_forward_cuda, times_backward_cuda)

            # Calculate threshold: expected_interval * 1.5
            threshold = expected_interval * 1.5
            threshold_cuda = CudaArray(np.array([threshold]), device="cpu")

            # Find intervals that exceed threshold
            gap_mask_cuda = elem_vec.vectorize_operation(
                intervals_cuda, "greater", threshold_cuda.to_numpy()[0]
            )
            gap_mask = gap_mask_cuda.to_numpy()

            # Cleanup GPU memory
            if times_forward_cuda.device == "cuda":
                times_forward_cuda.swap_to_cpu()
            if times_backward_cuda.device == "cuda":
                times_backward_cuda.swap_to_cpu()
            if intervals_cuda.device == "cuda":
                intervals_cuda.swap_to_cpu()
            if gap_mask_cuda.device == "cuda":
                gap_mask_cuda.swap_to_cpu()
            if threshold_cuda.device == "cuda":
                threshold_cuda.swap_to_cpu()

            # Find gap indices
            gap_indices = np.where(gap_mask)[0]

            # Generate missing points for each gap
            # This part still requires iteration, but it's over a small
            # number of gaps (not all intervals)
            missing_points = []
            intervals_np = intervals_cuda.to_numpy()
            for gap_idx in gap_indices:
                interval = float(intervals_np[gap_idx])
                n_expected = int(round(interval / expected_interval))
                if n_expected > 1:
                    # Generate expected times in the gap
                    # Use CUDA for batch generation if many points
                    if n_expected > 100:
                        # For large gaps, use vectorized generation
                        j_values = np.arange(1, n_expected, dtype=np.float64)
                        j_cuda = CudaArray(j_values, device="cpu")
                        expected_interval_cuda = CudaArray(
                            np.array([expected_interval]), device="cpu"
                        )
                        time_base_cuda = CudaArray(
                            np.array([times_np[gap_idx]]), device="cpu"
                        )
                        # Multiply j by expected_interval
                        j_times_interval_cuda = elem_vec.multiply(
                            j_cuda, expected_interval_cuda.to_numpy()[0]
                        )
                        # Add base time
                        expected_times_cuda = elem_vec.add(
                            j_times_interval_cuda, time_base_cuda.to_numpy()[0]
                        )
                        # Convert to list of floats using numpy tolist()
                        # instead of list comprehension
                        expected_times_np = expected_times_cuda.to_numpy()
                        missing_points.extend([float(x) for x in expected_times_np])
                        # Cleanup
                        if j_cuda.device == "cuda":
                            j_cuda.swap_to_cpu()
                        if expected_interval_cuda.device == "cuda":
                            expected_interval_cuda.swap_to_cpu()
                        if time_base_cuda.device == "cuda":
                            time_base_cuda.swap_to_cpu()
                        if j_times_interval_cuda.device == "cuda":
                            j_times_interval_cuda.swap_to_cpu()
                        if expected_times_cuda.device == "cuda":
                            expected_times_cuda.swap_to_cpu()
                    else:
                        # For small gaps, use vectorized CUDA operations
                        # Generate j values
                        j_values = np.arange(1, n_expected, dtype=np.float64)
                        j_cuda = CudaArray(j_values, device="cpu")
                        expected_interval_cuda = CudaArray(
                            np.array([expected_interval]), device="cpu"
                        )
                        time_base_cuda = CudaArray(
                            np.array([times_np[gap_idx]]), device="cpu"
                        )
                        # Multiply j by expected_interval
                        j_times_interval_cuda = elem_vec.multiply(
                            j_cuda, expected_interval_cuda.to_numpy()[0]
                        )
                        # Add base time
                        expected_times_cuda = elem_vec.add(
                            j_times_interval_cuda, time_base_cuda.to_numpy()[0]
                        )
                        # Convert to list of floats using numpy tolist()
                        # instead of list comprehension
                        expected_times_np = expected_times_cuda.to_numpy()
                        missing_points.extend([float(x) for x in expected_times_np])
                        # Cleanup
                        if j_cuda.device == "cuda":
                            j_cuda.swap_to_cpu()
                        if expected_interval_cuda.device == "cuda":
                            expected_interval_cuda.swap_to_cpu()
                        if time_base_cuda.device == "cuda":
                            time_base_cuda.swap_to_cpu()
                        if j_times_interval_cuda.device == "cuda":
                            j_times_interval_cuda.swap_to_cpu()
                        if expected_times_cuda.device == "cuda":
                            expected_times_cuda.swap_to_cpu()
        else:
            missing_points = []

        # Calculate coverage ratio
        # Simple subtraction for time span
        time_span = float(times_np[-1] - times_np[0])

        if time_span > 0:
            expected_n_points = int(round(time_span / expected_interval)) + 1
            coverage_ratio = (
                n_points / expected_n_points if expected_n_points > 0 else 0.0
            )
        else:
            coverage_ratio = 1.0

        # Cleanup GPU memory
        if times_cuda.device == "cuda":
            times_cuda.swap_to_cpu()

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
            "quality_issues": (self._quality_issues.copy()),
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
            report["warnings"].append(f"Found {len(gaps)} gap(s) in time coverage")

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

        # Check physical constraints using CUDA
        omega_min_cuda = CudaArray(omega_min_sorted, device="cpu")
        omega_macro_cuda = CudaArray(omega_macro_sorted, device="cpu")
        elem_vec = ElementWiseVectorizer(use_gpu=True)
        reduction_vec = ReductionVectorizer(use_gpu=True)

        # Check omega_min >= omega_macro (violations)
        omega_min_ge_macro = elem_vec.vectorize_operation(
            omega_min_cuda, "greater_equal", omega_macro_cuda.to_numpy()
        )
        violations_result = reduction_vec.vectorize_reduction(omega_min_ge_macro, "sum")
        violations = int(_to_float(violations_result))

        # Cleanup GPU memory
        if omega_min_cuda.device == "cuda":
            omega_min_cuda.swap_to_cpu()
        if omega_macro_cuda.device == "cuda":
            omega_macro_cuda.swap_to_cpu()
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
