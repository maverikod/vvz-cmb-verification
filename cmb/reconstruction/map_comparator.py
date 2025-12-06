"""
Map comparison utility for CMB map validation.

Compares reconstructed and observed maps using CUDA-accelerated operations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Dict
import numpy as np
from utils.cuda import CudaArray, ElementWiseVectorizer, ReductionVectorizer


def _to_float(value) -> float:
    """
    Convert reduction result to float.

    Args:
        value: Result from vectorize_reduction (Union[CudaArray, float, int])

    Returns:
        Float value
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, CudaArray):
        return float(value.to_numpy().item())
    return float(value)


class MapComparator:
    """
    Map comparison calculator.

    Compares two maps and calculates correlation, difference statistics.
    Uses CUDA acceleration for all operations.
    """

    def __init__(self):
        """Initialize comparator."""
        # Initialize CUDA vectorizers
        self.elem_vec = ElementWiseVectorizer(use_gpu=True, whole_array=True)
        self.reduction_vec = ReductionVectorizer(use_gpu=True)

    def compare(
        self, reconstructed_map: np.ndarray, observed_map: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare reconstructed and observed maps.

        Uses CUDA-accelerated operations for all calculations.

        Args:
            reconstructed_map: Reconstructed CMB map
            observed_map: Observed CMB map

        Returns:
            Dictionary with comparison metrics:
            - correlation: Correlation coefficient (Pearson)
            - mean_diff: Mean difference
            - std_diff: Standard deviation of difference
            - rms_diff: RMS difference

        Raises:
            ValueError: If maps have incompatible formats
        """
        # Validate map sizes
        if reconstructed_map.size != observed_map.size:
            raise ValueError(
                f"Map size mismatch: reconstructed {reconstructed_map.size}, "
                f"observed {observed_map.size}"
            )

        # Convert maps to CudaArray
        recon_cuda = CudaArray(reconstructed_map, device="cpu", block_size=None)
        obs_cuda = CudaArray(observed_map, device="cpu", block_size=None)

        # Calculate difference map using CUDA
        diff_cuda = self._calculate_difference(recon_cuda, obs_cuda)

        # Calculate statistics
        mean_diff = self._calculate_mean(diff_cuda)
        std_diff = self._calculate_std(diff_cuda, mean_diff)
        rms_diff = self._calculate_rms(diff_cuda)

        # Calculate Pearson correlation coefficient
        correlation = self._calculate_correlation(recon_cuda, obs_cuda)

        # Cleanup GPU memory
        self._cleanup_arrays([recon_cuda, obs_cuda, diff_cuda])

        return {
            "correlation": float(correlation),
            "mean_diff": float(mean_diff),
            "std_diff": float(std_diff),
            "rms_diff": float(rms_diff),
        }

    def _calculate_difference(
        self, recon_cuda: CudaArray, obs_cuda: CudaArray
    ) -> CudaArray:
        """Calculate difference map using ElementWiseVectorizer."""
        diff_cuda = self.elem_vec.subtract(recon_cuda, obs_cuda)
        return diff_cuda

    def _calculate_mean(self, diff_cuda: CudaArray) -> float:
        """Calculate mean difference."""
        mean_result = self.reduction_vec.vectorize_reduction(diff_cuda, "mean")
        return _to_float(mean_result)

    def _calculate_std(self, diff_cuda: CudaArray, mean_diff: float) -> float:
        """Calculate standard deviation of difference using ElementWiseVectorizer."""
        mean_diff_cuda = CudaArray(
            np.full(diff_cuda.shape, mean_diff, dtype=np.float64),
            device="cpu",
            block_size=None,
        )
        # Center array: diff - mean
        diff_centered_cuda = self.elem_vec.subtract(diff_cuda, mean_diff_cuda)
        # Square: (diff - mean)^2
        diff_squared_cuda = self.elem_vec.multiply(
            diff_centered_cuda, diff_centered_cuda
        )
        # Calculate variance
        var_result = self.reduction_vec.vectorize_reduction(diff_squared_cuda, "mean")
        variance = _to_float(var_result)
        self._cleanup_arrays([mean_diff_cuda, diff_centered_cuda, diff_squared_cuda])
        return np.sqrt(variance)

    def _calculate_rms(self, diff_cuda: CudaArray) -> float:
        """Calculate RMS difference using ElementWiseVectorizer."""
        # Square: diff^2
        diff_squared_cuda = self.elem_vec.multiply(diff_cuda, diff_cuda)
        # Calculate mean of squares
        mean_squared_result = self.reduction_vec.vectorize_reduction(
            diff_squared_cuda, "mean"
        )
        mean_squared = _to_float(mean_squared_result)
        self._cleanup_arrays([diff_squared_cuda])
        return np.sqrt(mean_squared)

    def _calculate_correlation(
        self, recon_cuda: CudaArray, obs_cuda: CudaArray
    ) -> float:
        """Calculate Pearson correlation coefficient."""
        # Calculate means
        recon_mean_result = self.reduction_vec.vectorize_reduction(recon_cuda, "mean")
        obs_mean_result = self.reduction_vec.vectorize_reduction(obs_cuda, "mean")
        recon_mean = _to_float(recon_mean_result)
        obs_mean = _to_float(obs_mean_result)

        # Center arrays using ElementWiseVectorizer
        recon_mean_cuda = CudaArray(
            np.full(recon_cuda.shape, recon_mean, dtype=np.float64),
            device="cpu",
            block_size=None,
        )
        obs_mean_cuda = CudaArray(
            np.full(obs_cuda.shape, obs_mean, dtype=np.float64),
            device="cpu",
            block_size=None,
        )

        # Center: array - mean
        recon_centered_cuda = self.elem_vec.subtract(recon_cuda, recon_mean_cuda)
        obs_centered_cuda = self.elem_vec.subtract(obs_cuda, obs_mean_cuda)
        # Covariance product: (recon - mean_recon) * (obs - mean_obs)
        cov_product_cuda = self.elem_vec.multiply(
            recon_centered_cuda, obs_centered_cuda
        )

        # Calculate covariance
        cov_result = self.reduction_vec.vectorize_reduction(cov_product_cuda, "mean")
        covariance = _to_float(cov_result)

        # Calculate standard deviations using ElementWiseVectorizer
        # Square centered arrays: (recon - mean)^2, (obs - mean)^2
        recon_squared_cuda = self.elem_vec.multiply(
            recon_centered_cuda, recon_centered_cuda
        )
        obs_squared_cuda = self.elem_vec.multiply(obs_centered_cuda, obs_centered_cuda)
        recon_var_result = self.reduction_vec.vectorize_reduction(
            recon_squared_cuda, "mean"
        )
        obs_var_result = self.reduction_vec.vectorize_reduction(
            obs_squared_cuda, "mean"
        )
        recon_std = np.sqrt(_to_float(recon_var_result))
        obs_std = np.sqrt(_to_float(obs_var_result))

        # Cleanup
        self._cleanup_arrays(
            [
                recon_mean_cuda,
                obs_mean_cuda,
                recon_centered_cuda,
                obs_centered_cuda,
                cov_product_cuda,
                recon_squared_cuda,
                obs_squared_cuda,
            ]
        )

        # Calculate correlation
        if recon_std > 0 and obs_std > 0:
            return covariance / (recon_std * obs_std)
        return 0.0

    def _cleanup_arrays(self, arrays: list) -> None:
        """Cleanup GPU memory for arrays."""
        for arr in arrays:
            if arr.device == "cuda":
                arr.swap_to_cpu()
