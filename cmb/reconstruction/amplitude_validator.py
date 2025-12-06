"""
Amplitude validation utility for CMB map validation.

Validates temperature fluctuation amplitude using CUDA-accelerated operations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Dict, Any
import numpy as np
from utils.cuda import CudaArray, ElementWiseVectorizer, ReductionVectorizer


def _to_float(value) -> float:
    """Convert reduction result to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, CudaArray):
        return float(value.to_numpy().item())
    return float(value)


class AmplitudeValidator:
    """
    Amplitude validator for CMB maps.

    Validates temperature fluctuation amplitude range.
    """

    def __init__(self):
        """Initialize amplitude validator."""
        self.elem_vec = ElementWiseVectorizer(use_gpu=True, whole_array=True)
        self.reduction_vec = ReductionVectorizer(use_gpu=True)

    def validate(
        self, map_data: np.ndarray, min_microK: float = 20.0, max_microK: float = 30.0
    ) -> Dict[str, Any]:
        """
        Validate temperature fluctuation amplitude.

        Args:
            map_data: CMB map array
            min_microK: Minimum expected amplitude in μK
            max_microK: Maximum expected amplitude in μK

        Returns:
            Dictionary with amplitude validation results

        Raises:
            ValueError: If range parameters are invalid
        """
        if min_microK >= max_microK:
            raise ValueError("min_microK must be less than max_microK")

        # Convert map to CudaArray
        map_cuda = CudaArray(map_data, device="cpu", block_size=None)

        # Calculate statistics
        mean_amplitude = self._calculate_mean(map_cuda)
        std_amplitude = self._calculate_std(map_cuda)

        # Calculate fraction in range
        in_range_fraction = self._calculate_in_range_fraction(
            map_cuda, min_microK, max_microK
        )

        # Validation passed if significant fraction is in range
        validation_passed = in_range_fraction >= 0.5

        # Cleanup
        self._cleanup_arrays([map_cuda])

        return {
            "mean_amplitude": float(mean_amplitude),
            "std_amplitude": float(std_amplitude),
            "in_range_fraction": float(in_range_fraction),
            "validation_passed": bool(validation_passed),
        }

    def _calculate_mean(self, map_cuda: CudaArray) -> float:
        """Calculate mean amplitude."""
        mean_result = self.reduction_vec.vectorize_reduction(map_cuda, "mean")
        return _to_float(mean_result)

    def _calculate_std(self, map_cuda: CudaArray) -> float:
        """Calculate standard deviation."""
        std_result = self.reduction_vec.vectorize_reduction(map_cuda, "std")
        return _to_float(std_result)

    def _calculate_in_range_fraction(
        self, map_cuda: CudaArray, min_microK: float, max_microK: float
    ) -> float:
        """Calculate fraction of pixels in range using ElementWiseVectorizer."""
        # Calculate absolute values using ElementWiseVectorizer
        abs_map_cuda = self.elem_vec.abs(map_cuda)

        # Create threshold arrays
        min_cuda = CudaArray(
            np.full(map_cuda.shape, min_microK, dtype=np.float64),
            device="cpu",
            block_size=None,
        )
        max_cuda = CudaArray(
            np.full(map_cuda.shape, max_microK, dtype=np.float64),
            device="cpu",
            block_size=None,
        )

        # Check range using ElementWiseVectorizer
        # abs_map >= min_microK
        ge_min_cuda = self.elem_vec.vectorize_operation(
            abs_map_cuda, "greater_equal", min_cuda
        )
        # abs_map <= max_microK
        le_max_cuda = self.elem_vec.vectorize_operation(
            abs_map_cuda, "less_equal", max_cuda
        )
        # in_range = (abs_map >= min) AND (abs_map <= max)
        in_range_cuda = self.elem_vec.multiply(ge_min_cuda, le_max_cuda)
        # Ensure float64 for reduction
        in_range_np = in_range_cuda.to_numpy().astype(np.float64)
        in_range_cuda = CudaArray(in_range_np, device="cpu", block_size=None)

        # Count pixels in range
        in_range_count = int(
            _to_float(self.reduction_vec.vectorize_reduction(in_range_cuda, "sum"))
        )
        total_pixels = map_cuda.shape[0]
        in_range_fraction = (
            float(in_range_count) / float(total_pixels) if total_pixels > 0 else 0.0
        )

        # Cleanup
        self._cleanup_arrays(
            [abs_map_cuda, min_cuda, max_cuda, ge_min_cuda, le_max_cuda, in_range_cuda]
        )

        return in_range_fraction

    def _cleanup_arrays(self, arrays: list) -> None:
        """Cleanup GPU memory."""
        for arr in arrays:
            if arr.device == "cuda":
                arr.swap_to_cpu()
