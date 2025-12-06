"""
Θ-node data processing for CMB verification project.

Processes node geometry, depth, and temperature mapping from
data/theta/nodes/ directory.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional
import numpy as np
from dataclasses import dataclass
import logging
from config.settings import Config, get_config
from cmb.theta_node_loader import load_node_geometry, load_node_depths
from utils.cuda import CudaArray, ElementWiseVectorizer, ReductionVectorizer
from typing import Any

logger = logging.getLogger(__name__)


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


@dataclass
class ThetaNodeData:
    """
    Θ-node structure data.

    Attributes:
        positions: Node sky positions (theta, phi) in radians
        scales: Node scales in parsec (~300 pc at z≈1100)
        depths: Node depths (Δω/ω)
        temperatures: Temperature fluctuations (ΔT in μK)
        metadata: Additional metadata
    """

    positions: np.ndarray  # Shape: (N, 2) for (theta, phi)
    scales: np.ndarray  # Shape: (N,)
    depths: np.ndarray  # Shape: (N,)
    temperatures: np.ndarray  # Shape: (N,)
    metadata: dict


def map_depth_to_temperature(
    depths: np.ndarray, config: Optional[Config] = None
) -> np.ndarray:
    """
    Map node depth (Δω/ω) to temperature fluctuation (ΔT).

    Formula from theory (tech_spec-new.md 2.1, ALL.md JW-CMB.6, JW-CMB2.6):
        ΔT = (Δω/ω_CMB) T_0

    Where:
        T_0 = 2.725 K (CMB temperature)
        ω_CMB ~ 10^11 Hz
        Δω = ω - ω_min (depth of node)
        depths = Δω/ω (relative depth from load_node_depths)

    For CMB nodes (where ω ≈ ω_CMB):
        Δω/ω ≈ Δω/ω_CMB
        Therefore: ΔT ≈ (Δω/ω) T_0 = depths * T_0

    This is an approximation valid for CMB nodes in the frequency range ~10^11 Hz.
    For non-CMB nodes, the full formula would require: ΔT = depths * (ω/ω_CMB) * T_0.

    Result: ΔT ≈ 20-30 μK (as observed in ACT/SPT data)

    This is direct conversion from node depth, NOT linear approximation
    or classical thermodynamic formula.

    See ALL.md JW-CMB.6 and JW-CMB2.6 for derivation.

    Uses CUDA acceleration with ElementWiseVectorizer for vectorized
    multiplication operations.

    Args:
        depths: Node depths (Δω/ω) - dimensionless relative depth
        config: Configuration instance (uses global if None)

    Returns:
        Array of temperature fluctuations in μK

    Raises:
        ValueError: If depths are invalid
    """
    if config is None:
        config = get_config()

    # Validate depths using CUDA-accelerated reduction
    depths = np.asarray(depths, dtype=float)
    depths_cuda = CudaArray(depths, device="cpu")

    # Check if any negative values using CUDA-accelerated comparison
    elem_vec = ElementWiseVectorizer(use_gpu=True)
    reduction_vec = ReductionVectorizer(use_gpu=True)

    # Compare depths < 0
    depths_lt_zero = elem_vec.vectorize_operation(depths_cuda, "less", 0.0)
    has_negative = reduction_vec.vectorize_reduction(depths_lt_zero, "any")

    if has_negative:
        raise ValueError("All depths must be non-negative")

    # Formula from theory (tech_spec-new.md 2.1, ALL.md JW-CMB.6, JW-CMB2.6):
    # ΔT = (Δω/ω_CMB) T_0
    #
    # Where:
    #   - depths = Δω/ω (relative depth from load_node_depths)
    #   - For CMB nodes: ω ≈ ω_CMB (nodes are in CMB frequency range ~10^11 Hz)
    #   - Therefore: Δω/ω ≈ Δω/ω_CMB for CMB nodes
    #
    # Strict formula: ΔT = (Δω/ω_CMB) T_0 = (Δω/ω) * (ω/ω_CMB) T_0
    # For CMB nodes (ω ≈ ω_CMB): ΔT ≈ (Δω/ω) T_0 = depths * T_0
    #
    # This is an approximation valid for CMB nodes where ω ≈ ω_CMB.
    # For non-CMB nodes, would need to multiply by (ω/ω_CMB) factor.
    #
    # Use ElementWiseVectorizer for CUDA-accelerated multiplication
    elem_vec = ElementWiseVectorizer(use_gpu=True)

    # Apply formula: ΔT ≈ depths * T_0 (valid for CMB nodes where ω ≈ ω_CMB)
    # This gives temperature in Kelvin
    temperatures_K_cuda = elem_vec.multiply(depths_cuda, config.constants.T_0)

    # Then multiply by 1e6 to convert to microKelvin
    temperatures_microK_cuda = elem_vec.multiply(temperatures_K_cuda, 1e6)

    # Convert back to numpy array
    temperatures_microK = temperatures_microK_cuda.to_numpy()

    # Cleanup GPU memory if used
    if temperatures_K_cuda.device == "cuda":
        temperatures_K_cuda.swap_to_cpu()
    if temperatures_microK_cuda.device == "cuda":
        temperatures_microK_cuda.swap_to_cpu()

    return temperatures_microK


def process_node_data(
    geometry_path: Optional[Path] = None, depth_path: Optional[Path] = None
) -> ThetaNodeData:
    """
    Process complete Θ-node data.

    Loads geometry and depth, maps to temperature, and creates
    ThetaNodeData structure.

    Args:
        geometry_path: Path to geometry data file
        depth_path: Path to depth data file

    Returns:
        ThetaNodeData instance with all processed data

    Raises:
        FileNotFoundError: If data files don't exist
        ValueError: If data processing fails
    """
    # Load geometry
    positions, scales = load_node_geometry(geometry_path)
    n_nodes = len(positions)

    # Load depths
    depths = load_node_depths(depth_path)

    # Validate consistency
    if len(depths) != n_nodes:
        raise ValueError(
            f"Inconsistent data: geometry has {n_nodes} nodes, "
            f"but depths has {len(depths)} values"
        )

    # Map depths to temperatures
    temperatures = map_depth_to_temperature(depths)

    # Create metadata using CUDA-accelerated reductions
    config = get_config()
    reduction_vec = ReductionVectorizer(use_gpu=True)

    # Convert arrays to CudaArray for CUDA processing
    scales_cuda = CudaArray(scales, device="cpu")
    depths_cuda = CudaArray(depths, device="cpu")
    temperatures_cuda = CudaArray(temperatures, device="cpu")

    # Compute statistics using CUDA-accelerated reductions
    scale_mean_result = reduction_vec.vectorize_reduction(scales_cuda, "mean")
    scale_std_result = reduction_vec.vectorize_reduction(scales_cuda, "std")
    depth_mean_result = reduction_vec.vectorize_reduction(depths_cuda, "mean")
    depth_std_result = reduction_vec.vectorize_reduction(depths_cuda, "std")
    temperature_mean_result = reduction_vec.vectorize_reduction(
        temperatures_cuda, "mean"
    )
    temperature_std_result = reduction_vec.vectorize_reduction(temperatures_cuda, "std")

    # Cleanup GPU memory if used
    if scales_cuda.device == "cuda":
        scales_cuda.swap_to_cpu()
    if depths_cuda.device == "cuda":
        depths_cuda.swap_to_cpu()
    if temperatures_cuda.device == "cuda":
        temperatures_cuda.swap_to_cpu()

    # Convert results to float
    scale_mean = _to_float(scale_mean_result)
    scale_std = _to_float(scale_std_result)
    depth_mean = _to_float(depth_mean_result)
    depth_std = _to_float(depth_std_result)
    temperature_mean = _to_float(temperature_mean_result)
    temperature_std = _to_float(temperature_std_result)

    metadata = {
        "n_nodes": n_nodes,
        "z_CMB": config.constants.z_CMB,
        "T_0": config.constants.T_0,
        "omega_CMB": config.constants.omega_CMB,
        "scale_mean_pc": scale_mean,
        "scale_std_pc": scale_std,
        "depth_mean": depth_mean,
        "depth_std": depth_std,
        "temperature_mean_microK": temperature_mean,
        "temperature_std_microK": temperature_std,
    }

    return ThetaNodeData(
        positions=positions,
        scales=scales,
        depths=depths,
        temperatures=temperatures,
        metadata=metadata,
    )


def validate_node_data(node_data: ThetaNodeData) -> bool:
    """
    Validate Θ-node data.

    Args:
        node_data: ThetaNodeData to validate

    Returns:
        True if valid

    Raises:
        ValueError: If data is invalid
    """
    # Check array shapes
    n_nodes = len(node_data.positions)

    if node_data.positions.shape != (n_nodes, 2):
        raise ValueError(
            f"Positions must have shape ({n_nodes}, 2), "
            f"got {node_data.positions.shape}"
        )

    if node_data.scales.shape != (n_nodes,):
        raise ValueError(
            f"Scales must have shape ({n_nodes},), " f"got {node_data.scales.shape}"
        )

    if node_data.depths.shape != (n_nodes,):
        raise ValueError(
            f"Depths must have shape ({n_nodes},), " f"got {node_data.depths.shape}"
        )

    if node_data.temperatures.shape != (n_nodes,):
        raise ValueError(
            f"Temperatures must have shape ({n_nodes},), "
            f"got {node_data.temperatures.shape}"
        )

    # Validate positions using CUDA-accelerated operations
    theta = node_data.positions[:, 0]
    phi = node_data.positions[:, 1]

    # Convert to CudaArray for CUDA processing
    theta_cuda = CudaArray(theta, device="cpu")
    phi_cuda = CudaArray(phi, device="cpu")
    scales_cuda = CudaArray(node_data.scales, device="cpu")
    depths_cuda = CudaArray(node_data.depths, device="cpu")
    temperatures_cuda = CudaArray(node_data.temperatures, device="cpu")

    elem_vec = ElementWiseVectorizer(use_gpu=True)
    reduction_vec = ReductionVectorizer(use_gpu=True)

    # Validate theta range [0, π]
    theta_lt_zero = elem_vec.vectorize_operation(theta_cuda, "less", 0.0)
    theta_gt_pi = elem_vec.vectorize_operation(theta_cuda, "greater", np.pi)
    has_theta_invalid = reduction_vec.vectorize_reduction(
        theta_lt_zero, "any"
    ) or reduction_vec.vectorize_reduction(theta_gt_pi, "any")

    if has_theta_invalid:
        theta_min_result = reduction_vec.vectorize_reduction(theta_cuda, "min")
        theta_max_result = reduction_vec.vectorize_reduction(theta_cuda, "max")
        theta_min = _to_float(theta_min_result)
        theta_max = _to_float(theta_max_result)
        raise ValueError(
            f"Theta values must be in [0, π]. "
            f"Found range: [{theta_min}, {theta_max}]"
        )

    # Validate phi range [0, 2π]
    phi_lt_zero = elem_vec.vectorize_operation(phi_cuda, "less", 0.0)
    phi_gt_2pi = elem_vec.vectorize_operation(phi_cuda, "greater", 2 * np.pi)
    has_phi_invalid = reduction_vec.vectorize_reduction(
        phi_lt_zero, "any"
    ) or reduction_vec.vectorize_reduction(phi_gt_2pi, "any")

    if has_phi_invalid:
        phi_min_result = reduction_vec.vectorize_reduction(phi_cuda, "min")
        phi_max_result = reduction_vec.vectorize_reduction(phi_cuda, "max")
        phi_min = _to_float(phi_min_result)
        phi_max = _to_float(phi_max_result)
        raise ValueError(
            f"Phi values must be in [0, 2π]. " f"Found range: [{phi_min}, {phi_max}]"
        )

    # Validate scales (must be positive)
    scales_le_zero = elem_vec.vectorize_operation(scales_cuda, "less_equal", 0.0)
    has_scale_invalid = reduction_vec.vectorize_reduction(scales_le_zero, "any")
    if has_scale_invalid:
        raise ValueError("All scales must be positive")

    # Validate depths (must be non-negative)
    depths_lt_zero = elem_vec.vectorize_operation(depths_cuda, "less", 0.0)
    has_depth_invalid = reduction_vec.vectorize_reduction(depths_lt_zero, "any")
    if has_depth_invalid:
        raise ValueError("All depths must be non-negative")

    # Validate temperatures (must be non-negative)
    config = get_config()
    temps_lt_zero = elem_vec.vectorize_operation(temperatures_cuda, "less", 0.0)
    has_temp_invalid = reduction_vec.vectorize_reduction(temps_lt_zero, "any")
    if has_temp_invalid:
        raise ValueError("All temperatures must be non-negative")

    # Check temperature range (20-30 μK expected) using CUDA reductions
    temp_min_result = reduction_vec.vectorize_reduction(temperatures_cuda, "min")
    temp_max_result = reduction_vec.vectorize_reduction(temperatures_cuda, "max")
    temp_min = _to_float(temp_min_result)
    temp_max = _to_float(temp_max_result)

    # Cleanup GPU memory if used
    for arr in [theta_cuda, phi_cuda, scales_cuda, depths_cuda, temperatures_cuda]:
        if arr.device == "cuda":
            arr.swap_to_cpu()

    if (
        temp_min < config.constants.delta_T_min * 0.5
        or temp_max > config.constants.delta_T_max * 2.0
    ):
        logger.warning(
            f"Temperature range [{temp_min:.2f}, {temp_max:.2f}] μK "
            f"is outside expected range "
            f"[{config.constants.delta_T_min}, {config.constants.delta_T_max}] μK"
        )

    # Validate metadata
    if not isinstance(node_data.metadata, dict):
        raise ValueError("Metadata must be a dictionary")

    return True
