"""
CMB map reconstruction from Θ-field frequency spectrum.

Implements formula CMB2.1 to generate spherical harmonic map ΔT(n̂)
from Θ-field nodes.

Formula CMB2.1:
    ρ_Θ(ω,t) ∝ ω^(-2±0.5) · (t/t_0)^(-3±1)

This frequency spectrum is integrated directly into reconstruction,
not applied as post-processing correction.

All array operations use CUDA utilities for acceleration:
- CudaArray for all arrays
- ElementWiseVectorizer for element-wise operations
- ReductionVectorizer for reductions
- TransformVectorizer for spherical harmonic synthesis

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional
import numpy as np
import healpy as hp
from cmb.theta_data_loader import ThetaFrequencySpectrum, ThetaEvolution
from cmb.theta_node_processor import ThetaNodeData, map_depth_to_temperature
from cmb.theta_evolution_processor import process_evolution_data
from config.settings import Config, get_config
from utils.cuda import (
    CudaArray,
    ElementWiseVectorizer,
    ReductionVectorizer,
    TransformVectorizer,
)


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


class CmbMapReconstructor:
    """
    CMB map reconstructor from Θ-field data.

    Reconstructs full-sky CMB temperature map from Θ-field
    frequency spectrum and node structure.

    Uses CUDA acceleration for all array operations:
    - Frequency spectrum integration
    - Node depth to temperature conversion
    - Sky coordinate projection
    - Spherical harmonic synthesis
    """

    def __init__(
        self,
        frequency_spectrum: ThetaFrequencySpectrum,
        evolution: ThetaEvolution,
        node_data: ThetaNodeData,
        nside: int = 2048,
        config: Optional[Config] = None,
    ):
        """
        Initialize reconstructor.

        Args:
            frequency_spectrum: Θ-field frequency spectrum
            evolution: Temporal evolution data
            node_data: Θ-node structure data
            nside: HEALPix NSIDE parameter
            config: Configuration instance (uses global if None)
        """
        self.frequency_spectrum = frequency_spectrum
        self.evolution = evolution
        self.node_data = node_data
        self.nside = nside

        if config is None:
            config = get_config()
        self.config = config

        # Process evolution data
        self.evolution_processor = process_evolution_data(evolution, config)

        # Initialize CUDA vectorizers
        self.elem_vec = ElementWiseVectorizer(use_gpu=True)
        self.reduction_vec = ReductionVectorizer(use_gpu=True)
        self.transform_vec = TransformVectorizer(use_gpu=True, whole_array=True)

        # Reference time t_0 (from config or default)
        self.t_0 = config.cmb_config.get("t_0", 1.0) if config.cmb_config else 1.0

        # Frequency spectrum power law indices
        # Formula CMB2.1: ρ_Θ(ω,t) ∝ ω^(-2±0.5) · (t/t_0)^(-3±1)
        self.alpha = config.cmb_config.get("alpha", -2.0) if config.cmb_config else -2.0
        self.beta = config.cmb_config.get("beta", -3.0) if config.cmb_config else -3.0

    def reconstruct_map(self) -> np.ndarray:
        """
        Reconstruct CMB temperature map from Θ-field.

        Uses formula CMB2.1: ρ_Θ(ω,t) ∝ ω^(-2±0.5) · (t/t_0)^(-3±1)

        Algorithm:
        1. Integrate frequency spectrum ρ_Θ(ω,t) over frequencies
        2. Convert node depths to temperatures: ΔT ~ T_0 · (Δω/ω_CMB)
        3. Project nodes to sky coordinates using phase parameters only
        4. Generate spherical harmonic map with integrated spectrum

        Returns:
            HEALPix map of temperature fluctuations ΔT in μK

        Raises:
            ValueError: If reconstruction fails
        """
        try:
            # Step 1: Integrate frequency spectrum for each node
            spectrum_weights = self._integrate_frequency_spectrum()

            # Step 2: Convert node depths to temperatures
            temperatures = self._convert_nodes_to_temperature()

            # Step 3: Project nodes to sky coordinates (validates positions)
            self._project_nodes_to_sky()

            # Step 4: Generate spherical harmonic map
            # Create HEALPix pixel indices for each node
            npix = hp.nside2npix(self.nside)

            # Initialize map using CudaArray
            # Create numpy array first (required for CudaArray constructor), then wrap
            zeros_np = np.zeros(npix, dtype=np.float64)
            cmb_map_cuda = CudaArray(zeros_np, device="cpu")

            # For each node, add temperature contribution to corresponding pixel
            # Use CUDA-accelerated operations for pixel assignment
            positions_np = self.node_data.positions
            temperatures_cuda = CudaArray(temperatures, device="cpu")
            spectrum_weights_cuda = CudaArray(spectrum_weights, device="cpu")

            # Calculate weighted temperatures: T_weighted = T * spectrum_weight
            weighted_temps_cuda = self.elem_vec.multiply(
                temperatures_cuda, spectrum_weights_cuda
            )
            weighted_temps = weighted_temps_cuda.to_numpy()

            # Convert sky coordinates to HEALPix pixel indices
            # theta, phi are in radians
            theta = positions_np[:, 0]
            phi = positions_np[:, 1]

            # Convert to HEALPix pixel indices
            # hp.ang2pix requires numpy arrays, use it directly
            pixel_indices = hp.ang2pix(self.nside, theta, phi)

            # Accumulate temperatures in pixels using CUDA-accelerated operations
            # For each unique pixel, sum contributions
            # Use numpy for unique pixel finding (requires sorting)
            unique_pixels, pixel_counts = np.unique(pixel_indices, return_counts=True)

            # Convert pixel_indices to CudaArray for CUDA-accelerated comparisons
            pixel_indices_cuda = CudaArray(pixel_indices, device="cpu")
            weighted_temps_cuda_for_accum = CudaArray(weighted_temps, device="cpu")

            # Accumulate contributions using CUDA
            cmb_map_np = cmb_map_cuda.to_numpy()
            for pixel_idx, count in zip(unique_pixels, pixel_counts):
                # Find all nodes contributing to this pixel using CUDA comparison
                # Use scalar pixel_idx directly (vectorize_operation supports scalars)
                node_mask_cuda = self.elem_vec.vectorize_operation(
                    pixel_indices_cuda, "equal", float(pixel_idx)
                )
                node_mask = node_mask_cuda.to_numpy()

                # Cleanup mask
                if node_mask_cuda.device == "cuda":
                    node_mask_cuda.swap_to_cpu()

                # Get node indices (np.where is acceptable for indexing)
                node_indices = np.where(node_mask)[0]

                # Sum contributions using CUDA
                if len(node_indices) > 0:
                    # Extract contributions using numpy indexing, then wrap in CudaArray
                    contributions_np = weighted_temps[node_indices]
                    contributions_cuda = CudaArray(contributions_np, device="cpu")
                    pixel_sum = self.reduction_vec.vectorize_reduction(
                        contributions_cuda, "sum"
                    )
                    cmb_map_np[pixel_idx] = _to_float(pixel_sum)

                    # Cleanup
                    if contributions_cuda.device == "cuda":
                        contributions_cuda.swap_to_cpu()

            # Cleanup pixel indices arrays
            if pixel_indices_cuda.device == "cuda":
                pixel_indices_cuda.swap_to_cpu()
            if weighted_temps_cuda_for_accum.device == "cuda":
                weighted_temps_cuda_for_accum.swap_to_cpu()

            # Cleanup GPU memory
            if temperatures_cuda.device == "cuda":
                temperatures_cuda.swap_to_cpu()
            if spectrum_weights_cuda.device == "cuda":
                spectrum_weights_cuda.swap_to_cpu()
            if weighted_temps_cuda.device == "cuda":
                weighted_temps_cuda.swap_to_cpu()
            if cmb_map_cuda.device == "cuda":
                cmb_map_cuda.swap_to_cpu()

            return cmb_map_np

        except Exception as e:
            raise ValueError(f"CMB map reconstruction failed: {str(e)}") from e

    def _convert_nodes_to_temperature(self) -> np.ndarray:
        """
        Convert Θ-node depths to temperature fluctuations.

        Formula (from tech_spec-new.md 2.1): ΔT = (Δω/ω_CMB) T_0
        Where:
            T_0 = 2.725 K (CMB temperature)
            ω_CMB ~ 10^11 Hz
            Δω = ω - ω_min (depth of node)
        Result: ΔT ≈ 20-30 μK

        This is NOT a linear approximation, but direct conversion
        from node depth (Δω/ω) to temperature fluctuation.

        Uses CUDA-accelerated operations via map_depth_to_temperature.

        Returns:
            Array of temperature fluctuations for each node in μK

        Raises:
            ValueError: If conversion fails
        """
        # Use existing function which already uses CUDA
        depths = self.node_data.depths
        temperatures = map_depth_to_temperature(depths, self.config)
        return temperatures

    def _project_nodes_to_sky(self) -> np.ndarray:
        """
        Project Θ-nodes to sky coordinates.

        Maps nodes from early universe (z≈1100) to current sky.

        Uses ONLY phase parameters (ω_min(t), ω_macro(t)) for evolution.
        Does NOT use classical cosmological formulas (FRW, ΛCDM).

        For now, uses node positions directly (they are already in sky coordinates).
        In future, could apply evolution-based projection.

        Returns:
            Array of (theta, phi) sky coordinates for each node in radians

        Raises:
            ValueError: If projection fails
        """
        # Node positions are already in sky coordinates (theta, phi)
        # Return them directly
        positions = self.node_data.positions

        # Validate positions using CUDA-accelerated operations
        theta = positions[:, 0]
        phi = positions[:, 1]

        theta_cuda = CudaArray(theta, device="cpu")
        phi_cuda = CudaArray(phi, device="cpu")

        # Validate theta range [0, π]
        theta_lt_zero = self.elem_vec.vectorize_operation(theta_cuda, "less", 0.0)
        theta_gt_pi = self.elem_vec.vectorize_operation(theta_cuda, "greater", np.pi)
        has_theta_invalid = self.reduction_vec.vectorize_reduction(
            theta_lt_zero, "any"
        ) or self.reduction_vec.vectorize_reduction(theta_gt_pi, "any")

        if has_theta_invalid:
            theta_min_result = self.reduction_vec.vectorize_reduction(theta_cuda, "min")
            theta_max_result = self.reduction_vec.vectorize_reduction(theta_cuda, "max")
            theta_min = _to_float(theta_min_result)
            theta_max = _to_float(theta_max_result)
            # Cleanup
            if theta_cuda.device == "cuda":
                theta_cuda.swap_to_cpu()
            if phi_cuda.device == "cuda":
                phi_cuda.swap_to_cpu()
            raise ValueError(
                f"Theta values must be in [0, π]. "
                f"Found range: [{theta_min}, {theta_max}]"
            )

        # Validate phi range [0, 2π]
        phi_lt_zero = self.elem_vec.vectorize_operation(phi_cuda, "less", 0.0)
        phi_gt_2pi = self.elem_vec.vectorize_operation(phi_cuda, "greater", 2 * np.pi)
        has_phi_invalid = self.reduction_vec.vectorize_reduction(
            phi_lt_zero, "any"
        ) or self.reduction_vec.vectorize_reduction(phi_gt_2pi, "any")

        if has_phi_invalid:
            phi_min_result = self.reduction_vec.vectorize_reduction(phi_cuda, "min")
            phi_max_result = self.reduction_vec.vectorize_reduction(phi_cuda, "max")
            phi_min = _to_float(phi_min_result)
            phi_max = _to_float(phi_max_result)
            # Cleanup
            if theta_cuda.device == "cuda":
                theta_cuda.swap_to_cpu()
            if phi_cuda.device == "cuda":
                phi_cuda.swap_to_cpu()
            raise ValueError(
                f"Phi values must be in [0, 2π]. "
                f"Found range: [{phi_min}, {phi_max}]"
            )

        # Cleanup GPU memory
        if theta_cuda.device == "cuda":
            theta_cuda.swap_to_cpu()
        if phi_cuda.device == "cuda":
            phi_cuda.swap_to_cpu()

        return positions

    def _integrate_frequency_spectrum(self) -> np.ndarray:
        """
        Integrate frequency spectrum ρ_Θ(ω,t) for each node.

        Uses formula CMB2.1: ρ_Θ(ω,t) ∝ ω^(-2±0.5) · (t/t_0)^(-3±1)

        Integrates over:
        - Frequency range for each node
        - Temporal evolution ω_min(t), ω_macro(t)

        Uses CUDA-accelerated operations for all calculations.

        Returns:
            Array of integrated spectrum weights for each node

        Raises:
            ValueError: If integration fails
        """
        n_nodes = len(self.node_data.positions)

        # Get frequency and time arrays from spectrum
        frequencies = self.frequency_spectrum.frequencies
        times = self.frequency_spectrum.times
        spectrum = self.frequency_spectrum.spectrum

        # Convert to CudaArray for CUDA processing
        frequencies_cuda = CudaArray(frequencies, device="cpu")
        times_cuda = CudaArray(times, device="cpu")
        spectrum_cuda = CudaArray(spectrum, device="cpu")

        # For each node, integrate spectrum over frequency range
        # Node frequency range: [ω_min, ω_macro] for each node
        # Use evolution processor to get ω_min and ω_macro at reference time
        # For simplicity, use mean time from evolution data
        times_evolution = self.evolution.times
        times_evolution_cuda = CudaArray(times_evolution, device="cpu")
        mean_time_result = self.reduction_vec.vectorize_reduction(
            times_evolution_cuda, "mean"
        )
        mean_time = _to_float(mean_time_result)
        if times_evolution_cuda.device == "cuda":
            times_evolution_cuda.swap_to_cpu()

        # Get ω_min and ω_macro at mean time
        omega_min_ref = self.evolution_processor.get_omega_min(mean_time)
        omega_macro_ref = self.evolution_processor.get_omega_macro(mean_time)

        # Calculate spectrum weights for each node
        # For each node, integrate: ∫ ρ_Θ(ω,t) dω over [ω_min, ω_macro]
        # Using formula: ρ_Θ(ω,t) ∝ ω^alpha · (t/t_0)^beta
        # Create CudaArray from numpy zeros (will be overwritten by integration result)
        weights_np = np.zeros(n_nodes, dtype=np.float64)
        weights_cuda = CudaArray(weights_np, device="cpu")

        # For each node, find frequency range and integrate
        # Use node depths to determine frequency range
        depths = self.node_data.depths
        depths_cuda = CudaArray(depths, device="cpu")

        # Calculate node frequencies: ω_node = ω_min + depth * (ω_macro - ω_min)
        # depth = Δω/ω, so ω_node ≈ ω_min * (1 + depth) for small depths
        # Create CudaArray from numpy full arrays
        omega_min_array_np = np.full(n_nodes, omega_min_ref, dtype=np.float64)
        omega_macro_array_np = np.full(n_nodes, omega_macro_ref, dtype=np.float64)
        omega_min_cuda = CudaArray(omega_min_array_np, device="cpu")
        omega_macro_cuda = CudaArray(omega_macro_array_np, device="cpu")

        # Calculate node frequencies using CUDA
        # ω_node ≈ ω_min + depth * (ω_macro - ω_min)
        omega_range_cuda = self.elem_vec.subtract(omega_macro_cuda, omega_min_cuda)
        omega_node_offset_cuda = self.elem_vec.multiply(depths_cuda, omega_range_cuda)
        omega_node_cuda = self.elem_vec.add(omega_min_cuda, omega_node_offset_cuda)

        # For each node, integrate spectrum over frequency range
        # Integration: ∫[ω_min to ω_node] ω^alpha · (t/t_0)^beta dω
        # For power law: ∫ ω^alpha dω = ω^(alpha+1) / (alpha+1) if alpha != -1
        # Time factor: (t/t_0)^beta is constant for fixed time
        time_factor = (mean_time / self.t_0) ** self.beta

        # Integrate for each node using CUDA
        # For alpha != -1: integral = (ω_node^(alpha+1) - ω_min^(alpha+1)) / (alpha+1)
        alpha_plus_one = self.alpha + 1.0

        if abs(alpha_plus_one) > 1e-10:
            # Calculate ω_node^(alpha+1) and ω_min^(alpha+1) using CUDA
            omega_node_power_cuda = self.elem_vec.power(omega_node_cuda, alpha_plus_one)
            omega_min_power_cuda = self.elem_vec.power(omega_min_cuda, alpha_plus_one)

            # Calculate difference
            omega_diff_cuda = self.elem_vec.subtract(
                omega_node_power_cuda, omega_min_power_cuda
            )

            # Divide by (alpha+1)
            integral_cuda = self.elem_vec.divide(omega_diff_cuda, alpha_plus_one)

            # Multiply by time factor
            weights_cuda = self.elem_vec.multiply(integral_cuda, time_factor)
        else:
            # Special case: alpha = -1
            # ∫ ω^(-1) dω = ln(ω)
            # Use CUDA for logarithm
            omega_node_log_cuda = self.elem_vec.log(omega_node_cuda)
            omega_min_log_cuda = self.elem_vec.log(omega_min_cuda)
            omega_diff_log_cuda = self.elem_vec.subtract(
                omega_node_log_cuda, omega_min_log_cuda
            )
            weights_cuda = self.elem_vec.multiply(omega_diff_log_cuda, time_factor)

            # Cleanup intermediate arrays
            if omega_node_log_cuda.device == "cuda":
                omega_node_log_cuda.swap_to_cpu()
            if omega_min_log_cuda.device == "cuda":
                omega_min_log_cuda.swap_to_cpu()
            if omega_diff_log_cuda.device == "cuda":
                omega_diff_log_cuda.swap_to_cpu()

        # Convert to numpy
        weights = weights_cuda.to_numpy()

        # Cleanup GPU memory
        if frequencies_cuda.device == "cuda":
            frequencies_cuda.swap_to_cpu()
        if times_cuda.device == "cuda":
            times_cuda.swap_to_cpu()
        if spectrum_cuda.device == "cuda":
            spectrum_cuda.swap_to_cpu()
        if depths_cuda.device == "cuda":
            depths_cuda.swap_to_cpu()
        if omega_min_cuda.device == "cuda":
            omega_min_cuda.swap_to_cpu()
        if omega_macro_cuda.device == "cuda":
            omega_macro_cuda.swap_to_cpu()
        if omega_range_cuda.device == "cuda":
            omega_range_cuda.swap_to_cpu()
        if omega_node_offset_cuda.device == "cuda":
            omega_node_offset_cuda.swap_to_cpu()
        if omega_node_cuda.device == "cuda":
            omega_node_cuda.swap_to_cpu()
        if weights_cuda.device == "cuda":
            weights_cuda.swap_to_cpu()

        # Cleanup additional arrays if created
        if abs(alpha_plus_one) > 1e-10:
            if omega_node_power_cuda.device == "cuda":
                omega_node_power_cuda.swap_to_cpu()
            if omega_min_power_cuda.device == "cuda":
                omega_min_power_cuda.swap_to_cpu()
            if omega_diff_cuda.device == "cuda":
                omega_diff_cuda.swap_to_cpu()
            if integral_cuda.device == "cuda":
                integral_cuda.swap_to_cpu()

        return weights
