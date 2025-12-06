"""
Structure validation utility for CMB map validation.

Validates arcmin-scale structures using spherical harmonic decomposition.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Dict, Any
import numpy as np
import healpy as hp
from utils.cuda import CudaArray, ElementWiseVectorizer, ReductionVectorizer
from utils.math.spherical_harmonics import decompose_map, synthesize_map


def _to_float(value) -> float:
    """Convert reduction result to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, CudaArray):
        return float(value.to_numpy().item())
    return float(value)


class StructureValidator:
    """
    Arcmin-scale structure validator.

    Validates structures at specified angular scales using spherical harmonics.
    """

    def __init__(self, nside: int):
        """
        Initialize structure validator.

        Args:
            nside: HEALPix NSIDE parameter
        """
        self.nside = nside
        self.elem_vec = ElementWiseVectorizer(use_gpu=True, whole_array=True)
        self.reduction_vec = ReductionVectorizer(use_gpu=True)

    def validate(
        self,
        reconstructed_map: np.ndarray,
        observed_map: np.ndarray,
        scale_min: float = 2.0,
        scale_max: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Validate arcmin-scale structures.

        Args:
            reconstructed_map: Reconstructed CMB map
            observed_map: Observed CMB map
            scale_min: Minimum scale in arcmin
            scale_max: Maximum scale in arcmin

        Returns:
            Dictionary with structure validation results

        Raises:
            ValueError: If scale parameters are invalid
        """
        if scale_min <= 0 or scale_max <= 0:
            raise ValueError("Scale parameters must be positive")
        if scale_min >= scale_max:
            raise ValueError("scale_min must be less than scale_max")

        # Convert arcmin to multipole l
        l_min = int(180 * 60 / scale_max)
        l_max = int(180 * 60 / scale_min)

        # Filter maps by scale
        recon_filtered, obs_filtered = self._filter_by_scale(
            reconstructed_map, observed_map, l_min, l_max
        )

        # Find structures
        recon_count, obs_count = self._count_structures(recon_filtered, obs_filtered)

        # Match positions and amplitudes
        position_matches, amplitude_matches = self._match_structures(
            recon_filtered, obs_filtered
        )

        return {
            "structure_count": max(recon_count, obs_count),
            "position_matches": position_matches,
            "amplitude_matches": amplitude_matches,
        }

    def _filter_by_scale(
        self, recon_map: np.ndarray, obs_map: np.ndarray, l_min: int, l_max: int
    ) -> tuple:
        """Filter maps by multipole range."""
        # Decompose maps
        recon_alm = decompose_map(recon_map, self.nside, l_max=l_max, use_cuda=True)
        obs_alm = decompose_map(obs_map, self.nside, l_max=l_max, use_cuda=True)

        # Convert to numpy
        recon_alm_np = (
            recon_alm.to_numpy() if isinstance(recon_alm, CudaArray) else recon_alm
        )
        obs_alm_np = obs_alm.to_numpy() if isinstance(obs_alm, CudaArray) else obs_alm

        # Create filter mask
        filter_mask = np.zeros(l_max + 1)
        filter_mask[l_min : l_max + 1] = 1.0
        recon_alm_filtered = hp.almxfl(recon_alm_np, filter_mask)
        obs_alm_filtered = hp.almxfl(obs_alm_np, filter_mask)

        # Synthesize filtered maps
        recon_filtered = synthesize_map(recon_alm_filtered, self.nside, use_cuda=True)
        obs_filtered = synthesize_map(obs_alm_filtered, self.nside, use_cuda=True)

        # Convert to numpy
        recon_filtered_np = (
            recon_filtered.to_numpy()
            if isinstance(recon_filtered, CudaArray)
            else recon_filtered
        )
        obs_filtered_np = (
            obs_filtered.to_numpy()
            if isinstance(obs_filtered, CudaArray)
            else obs_filtered
        )

        return recon_filtered_np, obs_filtered_np

    def _count_structures(
        self, recon_filtered: np.ndarray, obs_filtered: np.ndarray
    ) -> tuple:
        """Count structures above threshold."""
        recon_cuda = CudaArray(recon_filtered, device="cpu", block_size=None)
        obs_cuda = CudaArray(obs_filtered, device="cpu", block_size=None)

        # Calculate thresholds
        recon_mean = _to_float(
            self.reduction_vec.vectorize_reduction(recon_cuda, "mean")
        )
        obs_mean = _to_float(self.reduction_vec.vectorize_reduction(obs_cuda, "mean"))
        recon_std = _to_float(self.reduction_vec.vectorize_reduction(recon_cuda, "std"))
        obs_std = _to_float(self.reduction_vec.vectorize_reduction(obs_cuda, "std"))

        recon_threshold = recon_mean + 2.0 * recon_std
        obs_threshold = obs_mean + 2.0 * obs_std

        # Find structures above threshold
        recon_mask = self._create_threshold_mask(
            recon_cuda, recon_threshold, recon_filtered.shape
        )
        obs_mask = self._create_threshold_mask(
            obs_cuda, obs_threshold, obs_filtered.shape
        )

        # Count structures
        recon_count = int(
            _to_float(self.reduction_vec.vectorize_reduction(recon_mask, "sum"))
        )
        obs_count = int(
            _to_float(self.reduction_vec.vectorize_reduction(obs_mask, "sum"))
        )

        self._cleanup_arrays([recon_cuda, obs_cuda, recon_mask, obs_mask])

        return recon_count, obs_count

    def _create_threshold_mask(
        self, map_cuda: CudaArray, threshold: float, shape: tuple
    ) -> CudaArray:
        """Create mask for values above threshold using ElementWiseVectorizer."""
        threshold_cuda = CudaArray(
            np.full(shape, threshold, dtype=np.float64), device="cpu", block_size=None
        )
        # Compare: map > threshold
        mask_cuda = self.elem_vec.vectorize_operation(
            map_cuda, "greater", threshold_cuda
        )
        # Convert boolean mask to float64
        mask_np = mask_cuda.to_numpy().astype(np.float64)
        self._cleanup_arrays([threshold_cuda, mask_cuda])
        return CudaArray(mask_np, device="cpu", block_size=None)

    def _match_structures(
        self, recon_filtered: np.ndarray, obs_filtered: np.ndarray
    ) -> tuple:
        """Match structures by position and amplitude using CUDA utilities."""
        # Convert to CudaArray
        recon_cuda = CudaArray(recon_filtered, device="cpu", block_size=None)
        obs_cuda = CudaArray(obs_filtered, device="cpu", block_size=None)

        # Calculate means using ReductionVectorizer
        recon_mean = _to_float(
            self.reduction_vec.vectorize_reduction(recon_cuda, "mean")
        )
        obs_mean = _to_float(self.reduction_vec.vectorize_reduction(obs_cuda, "mean"))

        # Create threshold masks using ElementWiseVectorizer
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

        # Compare: array > mean using ElementWiseVectorizer
        recon_mask_cuda = self.elem_vec.vectorize_operation(
            recon_cuda, "greater", recon_mean_cuda
        )
        obs_mask_cuda = self.elem_vec.vectorize_operation(
            obs_cuda, "greater", obs_mean_cuda
        )

        # Convert masks to numpy for np.where (index operations)
        recon_mask = recon_mask_cuda.to_numpy()
        obs_mask = obs_mask_cuda.to_numpy()

        # Find pixel indices (np.where is acceptable for index operations)
        recon_pixels = np.where(recon_mask > 0.5)[0]
        obs_pixels = np.where(obs_mask > 0.5)[0]
        matching_pixels = np.intersect1d(recon_pixels, obs_pixels)
        position_matches = len(matching_pixels)

        # Match amplitudes using CUDA utilities
        amplitude_matches = 0
        if len(matching_pixels) > 0:
            # Get values at matching pixels (numpy indexing is acceptable)
            recon_values_np = recon_filtered[matching_pixels]
            obs_values_np = obs_filtered[matching_pixels]

            # Convert to CudaArray for calculations
            recon_values_cuda = CudaArray(
                recon_values_np, device="cpu", block_size=None
            )
            obs_values_cuda = CudaArray(obs_values_np, device="cpu", block_size=None)

            # Calculate absolute values using ElementWiseVectorizer
            recon_abs_cuda = self.elem_vec.abs(recon_values_cuda)
            obs_abs_cuda = self.elem_vec.abs(obs_values_cuda)

            # Calculate means using ReductionVectorizer
            recon_abs_mean = _to_float(
                self.reduction_vec.vectorize_reduction(recon_abs_cuda, "mean")
            )
            obs_abs_mean = _to_float(
                self.reduction_vec.vectorize_reduction(obs_abs_cuda, "mean")
            )
            mean_amp = (recon_abs_mean + obs_abs_mean) / 2.0

            if mean_amp > 0:
                # Calculate relative difference: |recon - obs| / mean_amp
                diff_cuda = self.elem_vec.subtract(recon_values_cuda, obs_values_cuda)
                diff_abs_cuda = self.elem_vec.abs(diff_cuda)
                rel_diff_cuda = self.elem_vec.divide(
                    diff_abs_cuda, CudaArray(np.array([mean_amp]), device="cpu")
                )

                # Count pixels within tolerance using ElementWiseVectorizer
                tolerance_cuda = CudaArray(
                    np.full(rel_diff_cuda.shape, 0.2, dtype=np.float64),
                    device="cpu",
                    block_size=None,
                )
                in_tolerance_mask_cuda = self.elem_vec.vectorize_operation(
                    rel_diff_cuda, "less_equal", tolerance_cuda
                )

                # Count matches using ReductionVectorizer
                amplitude_matches = int(
                    _to_float(
                        self.reduction_vec.vectorize_reduction(
                            in_tolerance_mask_cuda, "sum"
                        )
                    )
                )

            # Cleanup
            self._cleanup_arrays(
                [
                    recon_values_cuda,
                    obs_values_cuda,
                    recon_abs_cuda,
                    obs_abs_cuda,
                    diff_cuda,
                    diff_abs_cuda,
                    rel_diff_cuda,
                    tolerance_cuda,
                    in_tolerance_mask_cuda,
                ]
            )

        # Cleanup
        self._cleanup_arrays(
            [
                recon_cuda,
                obs_cuda,
                recon_mean_cuda,
                obs_mean_cuda,
                recon_mask_cuda,
                obs_mask_cuda,
            ]
        )

        return position_matches, amplitude_matches

    def _cleanup_arrays(self, arrays: list) -> None:
        """Cleanup GPU memory."""
        for arr in arrays:
            if arr.device == "cuda":
                arr.swap_to_cpu()
