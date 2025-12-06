"""
CMB map validation module.

Validates reconstructed CMB maps by comparing with ACT DR6.02
observational data.

All array operations use CUDA utilities for acceleration:
- CudaArray for all arrays
- ElementWiseVectorizer for element-wise operations
- ReductionVectorizer for reductions
- CorrelationVectorizer for correlation calculations
- TransformVectorizer for spherical harmonic operations

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import healpy as hp
from utils.io.data_loader import load_healpix_map
from utils.cuda import (
    CudaArray,
    ElementWiseVectorizer,
    ReductionVectorizer,
    CorrelationVectorizer,
    TransformVectorizer,
)
from utils.math.spherical_harmonics import decompose_map, synthesize_map

# Try to import CuPy for CUDA operations
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


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


class MapValidator:
    """
    CMB map validator.

    Validates reconstructed maps against ACT DR6.02 observations.
    Uses CUDA acceleration for all array operations.
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

        # Initialize CUDA vectorizers
        # Use whole_array=True for operations with arrays of same size
        self.elem_vec = ElementWiseVectorizer(use_gpu=True, whole_array=True)
        self.reduction_vec = ReductionVectorizer(use_gpu=True)
        self.corr_vec = CorrelationVectorizer(use_gpu=True, whole_array=True)
        self.transform_vec = TransformVectorizer(use_gpu=True, whole_array=True)

    def load_observed_map(self) -> None:
        """
        Load observed ACT DR6.02 map.

        Raises:
            FileNotFoundError: If map file doesn't exist
            ValueError: If map format is invalid
        """
        if not self.observed_map_path.exists():
            raise FileNotFoundError(
                f"Observed map file not found: {self.observed_map_path}"
            )

        try:
            # Load map using utility function
            self.observed_map = load_healpix_map(self.observed_map_path)

            # Validate map size
            expected_npix = 12 * self.nside * self.nside
            if self.observed_map.size != expected_npix:
                # Try to handle different NSIDE by downgrading/upgrading
                observed_nside = hp.npix2nside(self.observed_map.size)
                if observed_nside != self.nside:
                    # Resample to match nside
                    if observed_nside > self.nside:
                        # Downgrade
                        self.observed_map = hp.ud_grade(
                            self.observed_map, nside_out=self.nside
                        )
                    else:
                        # Upgrade
                        self.observed_map = hp.ud_grade(
                            self.observed_map, nside_out=self.nside
                        )

            # Validate final size
            if self.observed_map.size != expected_npix:
                raise ValueError(
                    f"Map size mismatch: expected {expected_npix}, "
                    f"got {self.observed_map.size}"
                )

        except Exception as e:
            raise ValueError(f"Failed to load observed map: {e}") from e

    def compare_maps(self) -> Dict[str, float]:
        """
        Compare reconstructed and observed maps.

        Uses CUDA-accelerated operations for all calculations.

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

        # Validate map sizes
        if self.observed_map is None:
            raise ValueError("Observed map is None")
        if self.reconstructed_map.size != self.observed_map.size:
            raise ValueError(
                f"Map size mismatch: reconstructed {self.reconstructed_map.size}, "
                f"observed {self.observed_map.size}"
            )

        # Convert maps to CudaArray
        recon_cuda = CudaArray(self.reconstructed_map, device="cpu", block_size=None)
        obs_cuda = CudaArray(self.observed_map, device="cpu", block_size=None)

        # Calculate difference map using CUDA
        # Use whole arrays for same-size operations
        recon_whole = recon_cuda.use_whole_array()
        obs_whole = obs_cuda.use_whole_array()

        # Perform subtraction on whole arrays
        if self.elem_vec.use_gpu and CUPY_AVAILABLE:
            if not isinstance(recon_whole, cp.ndarray):
                recon_whole = cp.asarray(recon_whole)
            if not isinstance(obs_whole, cp.ndarray):
                obs_whole = cp.asarray(obs_whole)
            diff_whole = recon_whole - obs_whole
            diff_np = cp.asnumpy(diff_whole)
        else:
            diff_np = recon_whole - obs_whole

        diff_cuda = CudaArray(diff_np, device="cpu", block_size=None)

        # Calculate mean difference using CUDA
        mean_diff_result = self.reduction_vec.vectorize_reduction(diff_cuda, "mean")
        mean_diff = _to_float(mean_diff_result)

        # Calculate standard deviation of difference using CUDA
        # std = sqrt(mean((x - mean(x))^2))
        mean_diff_cuda = CudaArray(
            np.full(diff_cuda.shape, mean_diff, dtype=np.float64),
            device="cpu",
            block_size=None,
        )
        # Use whole arrays for same-size operations
        diff_whole = diff_cuda.use_whole_array()
        mean_diff_whole = mean_diff_cuda.use_whole_array()

        if self.elem_vec.use_gpu and CUPY_AVAILABLE:
            if not isinstance(diff_whole, cp.ndarray):
                diff_whole = cp.asarray(diff_whole)
            if not isinstance(mean_diff_whole, cp.ndarray):
                mean_diff_whole = cp.asarray(mean_diff_whole)
            diff_centered_whole = diff_whole - mean_diff_whole
            diff_squared_whole = diff_centered_whole * diff_centered_whole
            diff_centered_np = cp.asnumpy(diff_centered_whole)
            diff_squared_np = cp.asnumpy(diff_squared_whole)
        else:
            diff_centered_np = diff_whole - mean_diff_whole
            diff_squared_np = diff_centered_np * diff_centered_np

        diff_centered_cuda = CudaArray(diff_centered_np, device="cpu", block_size=None)
        diff_squared_cuda = CudaArray(diff_squared_np, device="cpu", block_size=None)
        var_result = self.reduction_vec.vectorize_reduction(diff_squared_cuda, "mean")
        variance = _to_float(var_result)
        std_diff = np.sqrt(variance)

        # Calculate RMS difference: sqrt(mean(diff^2))
        diff_whole = diff_cuda.use_whole_array()
        if self.elem_vec.use_gpu and CUPY_AVAILABLE:
            if not isinstance(diff_whole, cp.ndarray):
                diff_whole = cp.asarray(diff_whole)
            diff_squared_whole = diff_whole * diff_whole
            diff_squared_np = cp.asnumpy(diff_squared_whole)
        else:
            diff_squared_np = diff_whole * diff_whole
        diff_squared_for_rms = CudaArray(diff_squared_np, device="cpu", block_size=None)
        mean_squared_result = self.reduction_vec.vectorize_reduction(
            diff_squared_for_rms, "mean"
        )
        mean_squared = _to_float(mean_squared_result)
        rms_diff = np.sqrt(mean_squared)

        # Calculate Pearson correlation coefficient using CUDA
        # corr = mean((x - mean(x)) * (y - mean(y))) / (std(x) * std(y))
        recon_mean_result = self.reduction_vec.vectorize_reduction(recon_cuda, "mean")
        obs_mean_result = self.reduction_vec.vectorize_reduction(obs_cuda, "mean")
        recon_mean = _to_float(recon_mean_result)
        obs_mean = _to_float(obs_mean_result)

        # Center arrays
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
        # Use whole arrays for same-size operations
        recon_whole = recon_cuda.use_whole_array()
        obs_whole = obs_cuda.use_whole_array()
        recon_mean_whole = recon_mean_cuda.use_whole_array()
        obs_mean_whole = obs_mean_cuda.use_whole_array()

        if self.elem_vec.use_gpu and CUPY_AVAILABLE:
            if not isinstance(recon_whole, cp.ndarray):
                recon_whole = cp.asarray(recon_whole)
            if not isinstance(obs_whole, cp.ndarray):
                obs_whole = cp.asarray(obs_whole)
            if not isinstance(recon_mean_whole, cp.ndarray):
                recon_mean_whole = cp.asarray(recon_mean_whole)
            if not isinstance(obs_mean_whole, cp.ndarray):
                obs_mean_whole = cp.asarray(obs_mean_whole)
            recon_centered_whole = recon_whole - recon_mean_whole
            obs_centered_whole = obs_whole - obs_mean_whole
            cov_product_whole = recon_centered_whole * obs_centered_whole
            recon_centered_np = cp.asnumpy(recon_centered_whole)
            obs_centered_np = cp.asnumpy(obs_centered_whole)
            cov_product_np = cp.asnumpy(cov_product_whole)
        else:
            recon_centered_np = recon_whole - recon_mean_whole
            obs_centered_np = obs_whole - obs_mean_whole
            cov_product_np = recon_centered_np * obs_centered_np

        recon_centered_cuda = CudaArray(
            recon_centered_np, device="cpu", block_size=None
        )
        obs_centered_cuda = CudaArray(obs_centered_np, device="cpu", block_size=None)
        cov_product_cuda = CudaArray(cov_product_np, device="cpu", block_size=None)
        cov_result = self.reduction_vec.vectorize_reduction(cov_product_cuda, "mean")
        covariance = _to_float(cov_result)

        # Calculate standard deviations
        recon_centered_whole = recon_centered_cuda.use_whole_array()
        obs_centered_whole = obs_centered_cuda.use_whole_array()

        if self.elem_vec.use_gpu and CUPY_AVAILABLE:
            if not isinstance(recon_centered_whole, cp.ndarray):
                recon_centered_whole = cp.asarray(recon_centered_whole)
            if not isinstance(obs_centered_whole, cp.ndarray):
                obs_centered_whole = cp.asarray(obs_centered_whole)
            recon_squared_whole = recon_centered_whole * recon_centered_whole
            obs_squared_whole = obs_centered_whole * obs_centered_whole
            recon_squared_np = cp.asnumpy(recon_squared_whole)
            obs_squared_np = cp.asnumpy(obs_squared_whole)
        else:
            recon_squared_np = recon_centered_whole * recon_centered_whole
            obs_squared_np = obs_centered_whole * obs_centered_whole

        recon_squared_cuda = CudaArray(recon_squared_np, device="cpu", block_size=None)
        obs_squared_cuda = CudaArray(obs_squared_np, device="cpu", block_size=None)
        recon_var_result = self.reduction_vec.vectorize_reduction(
            recon_squared_cuda, "mean"
        )
        obs_var_result = self.reduction_vec.vectorize_reduction(
            obs_squared_cuda, "mean"
        )
        recon_std = np.sqrt(_to_float(recon_var_result))
        obs_std = np.sqrt(_to_float(obs_var_result))

        # Calculate correlation
        if recon_std > 0 and obs_std > 0:
            correlation = covariance / (recon_std * obs_std)
        else:
            correlation = 0.0

        # Cleanup GPU memory
        for arr in [
            recon_cuda,
            obs_cuda,
            diff_cuda,
            mean_diff_cuda,
            diff_centered_cuda,
            diff_squared_cuda,
            diff_squared_for_rms,
            recon_mean_cuda,
            obs_mean_cuda,
            recon_centered_cuda,
            obs_centered_cuda,
            cov_product_cuda,
            recon_squared_cuda,
            obs_squared_cuda,
        ]:
            if arr.device == "cuda":
                arr.swap_to_cpu()

        return {
            "correlation": float(correlation),
            "mean_diff": float(mean_diff),
            "std_diff": float(std_diff),
            "rms_diff": float(rms_diff),
        }

    def validate_arcmin_structures(
        self, scale_min: float = 2.0, scale_max: float = 5.0
    ) -> Dict[str, Any]:
        """
        Validate arcmin-scale structures (2-5′).

        Uses spherical harmonic decomposition to filter structures
        at specified angular scales.

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
        if scale_min <= 0 or scale_max <= 0:
            raise ValueError("Scale parameters must be positive")
        if scale_min >= scale_max:
            raise ValueError("scale_min must be less than scale_max")

        if self.observed_map is None:
            self.load_observed_map()

        # Convert arcmin to multipole l
        # l ≈ 180 * 60 / θ_arcmin (approximate)
        l_min = int(180 * 60 / scale_max)
        l_max = int(180 * 60 / scale_min)

        # Decompose maps into spherical harmonics
        # Use whole array mode for FFT operations
        if self.observed_map is None:
            raise ValueError("Observed map is None")
        recon_alm = decompose_map(
            self.reconstructed_map, self.nside, l_max=l_max, use_cuda=True
        )
        obs_alm = decompose_map(
            self.observed_map, self.nside, l_max=l_max, use_cuda=True
        )

        # Filter by multipole range
        # Extract a_lm coefficients for l in [l_min, l_max]
        # healpy stores a_lm as complex array
        # Need to filter by l value
        recon_alm_np = (
            recon_alm.to_numpy() if isinstance(recon_alm, CudaArray) else recon_alm
        )
        obs_alm_np = obs_alm.to_numpy() if isinstance(obs_alm, CudaArray) else obs_alm

        # Filter a_lm by l range
        # healpy stores a_lm in order: l=0, m=0; l=1, m=-1,0,1; ...
        # Need to extract indices for l in [l_min, l_max]
        recon_alm_filtered = hp.almxfl(recon_alm_np, np.ones(l_max + 1))
        obs_alm_filtered = hp.almxfl(obs_alm_np, np.ones(l_max + 1))

        # Create filter mask for l range
        filter_mask = np.zeros(l_max + 1)
        filter_mask[l_min : l_max + 1] = 1.0
        recon_alm_filtered = hp.almxfl(recon_alm_np, filter_mask)
        obs_alm_filtered = hp.almxfl(obs_alm_np, filter_mask)

        # Synthesize filtered maps
        recon_filtered = synthesize_map(recon_alm_filtered, self.nside, use_cuda=True)
        obs_filtered = synthesize_map(obs_alm_filtered, self.nside, use_cuda=True)

        # Convert to numpy if needed
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

        # Find structures (local extrema) using CUDA
        # Use threshold to identify significant structures
        recon_cuda = CudaArray(recon_filtered_np, device="cpu", block_size=None)
        obs_cuda = CudaArray(obs_filtered_np, device="cpu", block_size=None)

        # Calculate thresholds (e.g., 2 sigma above mean)
        recon_mean_result = self.reduction_vec.vectorize_reduction(recon_cuda, "mean")
        obs_mean_result = self.reduction_vec.vectorize_reduction(obs_cuda, "mean")
        recon_std_result = self.reduction_vec.vectorize_reduction(recon_cuda, "std")
        obs_std_result = self.reduction_vec.vectorize_reduction(obs_cuda, "std")
        recon_mean = _to_float(recon_mean_result)
        obs_mean = _to_float(obs_mean_result)
        recon_std = _to_float(recon_std_result)
        obs_std = _to_float(obs_std_result)

        recon_threshold = recon_mean + 2.0 * recon_std
        obs_threshold = obs_mean + 2.0 * obs_std

        # Find structures above threshold
        recon_threshold_cuda = CudaArray(
            np.full(recon_cuda.shape, recon_threshold, dtype=np.float64),
            device="cpu",
            block_size=None,
        )
        obs_threshold_cuda = CudaArray(
            np.full(obs_cuda.shape, obs_threshold, dtype=np.float64),
            device="cpu",
            block_size=None,
        )

        # Use whole arrays for same-size operations
        recon_whole = recon_cuda.use_whole_array()
        obs_whole = obs_cuda.use_whole_array()
        recon_threshold_whole = recon_threshold_cuda.use_whole_array()
        obs_threshold_whole = obs_threshold_cuda.use_whole_array()

        if self.elem_vec.use_gpu and CUPY_AVAILABLE:
            if not isinstance(recon_whole, cp.ndarray):
                recon_whole = cp.asarray(recon_whole)
            if not isinstance(obs_whole, cp.ndarray):
                obs_whole = cp.asarray(obs_whole)
            if not isinstance(recon_threshold_whole, cp.ndarray):
                recon_threshold_whole = cp.asarray(recon_threshold_whole)
            if not isinstance(obs_threshold_whole, cp.ndarray):
                obs_threshold_whole = cp.asarray(obs_threshold_whole)
            recon_mask_whole = recon_whole > recon_threshold_whole
            obs_mask_whole = obs_whole > obs_threshold_whole
            recon_mask_np = cp.asnumpy(recon_mask_whole).astype(np.float64)
            obs_mask_np = cp.asnumpy(obs_mask_whole).astype(np.float64)
        else:
            recon_mask_np = (recon_whole > recon_threshold_whole).astype(np.float64)
            obs_mask_np = (obs_whole > obs_threshold_whole).astype(np.float64)

        recon_mask_cuda = CudaArray(recon_mask_np, device="cpu", block_size=None)
        obs_mask_cuda = CudaArray(obs_mask_np, device="cpu", block_size=None)

        # Count structures
        recon_count_result = self.reduction_vec.vectorize_reduction(
            recon_mask_cuda, "sum"
        )
        obs_count_result = self.reduction_vec.vectorize_reduction(obs_mask_cuda, "sum")
        recon_count = int(_to_float(recon_count_result))
        obs_count = int(_to_float(obs_count_result))

        # Find matching positions (within angular tolerance)
        # Convert masks to numpy for position finding
        recon_mask = recon_mask_cuda.to_numpy()
        obs_mask = obs_mask_cuda.to_numpy()

        # Find pixel indices with structures
        recon_pixels = np.where(recon_mask)[0]
        obs_pixels = np.where(obs_mask)[0]

        # Match positions (pixels within tolerance)
        # For simplicity, count exact pixel matches
        # In production, would use angular distance
        matching_pixels = np.intersect1d(recon_pixels, obs_pixels)
        position_matches = len(matching_pixels)

        # Match amplitudes (within tolerance)
        # Compare amplitudes at matching pixels
        recon_values = recon_filtered_np[matching_pixels]
        obs_values = obs_filtered_np[matching_pixels]
        amplitude_tolerance = 0.2  # 20% tolerance
        amplitude_matches = 0
        if len(matching_pixels) > 0:
            # Use CUDA for amplitude comparison
            recon_values_cuda = CudaArray(recon_values, device="cpu", block_size=None)
            obs_values_cuda = CudaArray(obs_values, device="cpu", block_size=None)
            # Calculate relative difference
            recon_values_whole = recon_values_cuda.use_whole_array()
            obs_values_whole = obs_values_cuda.use_whole_array()

            if self.elem_vec.use_gpu and CUPY_AVAILABLE:
                if not isinstance(recon_values_whole, cp.ndarray):
                    recon_values_whole = cp.asarray(recon_values_whole)
                if not isinstance(obs_values_whole, cp.ndarray):
                    obs_values_whole = cp.asarray(obs_values_whole)
                diff_whole = recon_values_whole - obs_values_whole
                abs_diff_whole = cp.abs(diff_whole)
                diff_np = cp.asnumpy(diff_whole)
                abs_diff_np = cp.asnumpy(abs_diff_whole)
            else:
                diff_np = recon_values_whole - obs_values_whole
                abs_diff_np = np.abs(diff_np)

            diff_cuda = CudaArray(diff_np, device="cpu", block_size=None)
            abs_diff_cuda = CudaArray(abs_diff_np, device="cpu", block_size=None)

            # Normalize by mean amplitude
            mean_amp = (np.abs(recon_values).mean() + np.abs(obs_values).mean()) / 2.0
            if mean_amp > 0:
                abs_diff_whole = abs_diff_cuda.use_whole_array()
                if self.elem_vec.use_gpu and CUPY_AVAILABLE:
                    if not isinstance(abs_diff_whole, cp.ndarray):
                        abs_diff_whole = cp.asarray(abs_diff_whole)
                    rel_diff_whole = abs_diff_whole / mean_amp
                    tolerance_whole = cp.full(
                        rel_diff_whole.shape, amplitude_tolerance, dtype=cp.float64
                    )
                    match_mask_whole = rel_diff_whole <= tolerance_whole
                    rel_diff_np = cp.asnumpy(rel_diff_whole)
                    match_mask_np = cp.asnumpy(match_mask_whole).astype(np.float64)
                else:
                    rel_diff_np = abs_diff_whole / mean_amp
                    match_mask_np = (rel_diff_np <= amplitude_tolerance).astype(
                        np.float64
                    )

                rel_diff_cuda = CudaArray(rel_diff_np, device="cpu", block_size=None)
                match_mask_cuda = CudaArray(
                    match_mask_np, device="cpu", block_size=None
                )

                amplitude_matches_result = self.reduction_vec.vectorize_reduction(
                    match_mask_cuda, "sum"
                )
                amplitude_matches = int(_to_float(amplitude_matches_result))

                # Cleanup
                for arr in [
                    recon_values_cuda,
                    obs_values_cuda,
                    diff_cuda,
                    abs_diff_cuda,
                    rel_diff_cuda,
                    match_mask_cuda,
                ]:
                    if arr.device == "cuda":
                        arr.swap_to_cpu()

        # Cleanup GPU memory
        for arr in [
            recon_cuda,
            obs_cuda,
            recon_threshold_cuda,
            obs_threshold_cuda,
            recon_mask_cuda,
            obs_mask_cuda,
        ]:
            if arr.device == "cuda":
                arr.swap_to_cpu()

        return {
            "structure_count": max(recon_count, obs_count),
            "position_matches": position_matches,
            "amplitude_matches": amplitude_matches,
        }

    def validate_amplitude(
        self, min_microK: float = 20.0, max_microK: float = 30.0
    ) -> Dict[str, Any]:
        """
        Validate temperature fluctuation amplitude (20-30 μK).

        Uses CUDA-accelerated operations for all calculations.

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
        if min_microK >= max_microK:
            raise ValueError("min_microK must be less than max_microK")

        # Convert maps to CudaArray
        recon_cuda = CudaArray(self.reconstructed_map, device="cpu")

        # Calculate mean amplitude using CUDA
        mean_result = self.reduction_vec.vectorize_reduction(recon_cuda, "mean")
        mean_amplitude = _to_float(mean_result)

        # Calculate standard deviation using CUDA
        std_result = self.reduction_vec.vectorize_reduction(recon_cuda, "std")
        std_amplitude = _to_float(std_result)

        # Calculate fraction of pixels in range
        # Count pixels where min_microK <= |value| <= max_microK
        recon_whole = recon_cuda.use_whole_array()
        if self.elem_vec.use_gpu and CUPY_AVAILABLE:
            if not isinstance(recon_whole, cp.ndarray):
                recon_whole = cp.asarray(recon_whole)
            abs_recon_whole = cp.abs(recon_whole)
            abs_recon_np = cp.asnumpy(abs_recon_whole)
        else:
            abs_recon_np = np.abs(recon_whole)
        abs_recon_cuda = CudaArray(abs_recon_np, device="cpu", block_size=None)

        min_cuda = CudaArray(
            np.full(recon_cuda.shape, min_microK, dtype=np.float64),
            device="cpu",
            block_size=None,
        )
        max_cuda = CudaArray(
            np.full(recon_cuda.shape, max_microK, dtype=np.float64),
            device="cpu",
            block_size=None,
        )

        # Check: min_microK <= |value| <= max_microK
        abs_recon_whole = abs_recon_cuda.use_whole_array()
        min_whole = min_cuda.use_whole_array()
        max_whole = max_cuda.use_whole_array()

        if self.elem_vec.use_gpu and CUPY_AVAILABLE:
            if not isinstance(abs_recon_whole, cp.ndarray):
                abs_recon_whole = cp.asarray(abs_recon_whole)
            if not isinstance(min_whole, cp.ndarray):
                min_whole = cp.asarray(min_whole)
            if not isinstance(max_whole, cp.ndarray):
                max_whole = cp.asarray(max_whole)
            ge_min_whole = abs_recon_whole >= min_whole
            le_max_whole = abs_recon_whole <= max_whole
            in_range_whole = ge_min_whole * le_max_whole
            ge_min_np = cp.asnumpy(ge_min_whole).astype(np.float64)
            le_max_np = cp.asnumpy(le_max_whole).astype(np.float64)
            in_range_np = cp.asnumpy(in_range_whole).astype(np.float64)
        else:
            ge_min_np = (abs_recon_whole >= min_whole).astype(np.float64)
            le_max_np = (abs_recon_whole <= max_whole).astype(np.float64)
            in_range_np = (ge_min_np * le_max_np).astype(np.float64)

        ge_min_cuda = CudaArray(ge_min_np, device="cpu", block_size=None)
        le_max_cuda = CudaArray(le_max_np, device="cpu", block_size=None)
        in_range_cuda = CudaArray(in_range_np, device="cpu", block_size=None)

        # Count pixels in range
        in_range_count_result = self.reduction_vec.vectorize_reduction(
            in_range_cuda, "sum"
        )
        in_range_count = int(_to_float(in_range_count_result))
        total_pixels = recon_cuda.shape[0]
        in_range_fraction = (
            float(in_range_count) / float(total_pixels) if total_pixels > 0 else 0.0
        )

        # Validation passed if significant fraction is in range
        validation_passed = in_range_fraction >= 0.5  # At least 50% in range

        # Cleanup GPU memory
        for arr in [
            recon_cuda,
            abs_recon_cuda,
            min_cuda,
            max_cuda,
            ge_min_cuda,
            le_max_cuda,
            in_range_cuda,
        ]:
            if arr.device == "cuda":
                arr.swap_to_cpu()

        return {
            "mean_amplitude": float(mean_amplitude),
            "std_amplitude": float(std_amplitude),
            "in_range_fraction": float(in_range_fraction),
            "validation_passed": bool(validation_passed),
        }

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
