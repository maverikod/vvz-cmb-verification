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

logger = logging.getLogger(__name__)


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

    Formula (from tech_spec-new.md 2.1): ΔT = (Δω/ω_CMB) T_0
    Where:
        T_0 = 2.725 K (CMB temperature)
        ω_CMB ~ 10^11 Hz
        Δω = ω - ω_min (depth of node)
    Result: ΔT ≈ 20-30 μK

    This is direct conversion from node depth, NOT linear approximation
    or classical thermodynamic formula.

    See ALL.md JW-CMB.6 and JW-CMB2.6 for derivation.

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

    # Validate depths
    depths = np.asarray(depths, dtype=float)
    if np.any(depths < 0):
        raise ValueError("All depths must be non-negative")

    # Formula: ΔT = (Δω/ω_CMB) T_0
    # depths is already Δω/ω, so we need to convert to ΔT
    # ΔT = depths * T_0 (in Kelvin)
    # Then convert to microKelvin
    temperatures_K = depths * config.constants.T_0
    temperatures_microK = temperatures_K * 1e6

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

    # Create metadata
    config = get_config()
    metadata = {
        "n_nodes": n_nodes,
        "z_CMB": config.constants.z_CMB,
        "T_0": config.constants.T_0,
        "omega_CMB": config.constants.omega_CMB,
        "scale_mean_pc": float(np.mean(scales)),
        "scale_std_pc": float(np.std(scales)),
        "depth_mean": float(np.mean(depths)),
        "depth_std": float(np.std(depths)),
        "temperature_mean_microK": float(np.mean(temperatures)),
        "temperature_std_microK": float(np.std(temperatures)),
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

    # Validate positions
    theta = node_data.positions[:, 0]
    phi = node_data.positions[:, 1]

    if np.any(theta < 0) or np.any(theta > np.pi):
        raise ValueError(
            f"Theta values must be in [0, π]. "
            f"Found range: [{np.min(theta)}, {np.max(theta)}]"
        )

    if np.any(phi < 0) or np.any(phi > 2 * np.pi):
        raise ValueError(
            f"Phi values must be in [0, 2π]. "
            f"Found range: [{np.min(phi)}, {np.max(phi)}]"
        )

    # Validate scales
    if np.any(node_data.scales <= 0):
        raise ValueError("All scales must be positive")

    # Validate depths
    if np.any(node_data.depths < 0):
        raise ValueError("All depths must be non-negative")

    # Validate temperatures
    config = get_config()
    if np.any(node_data.temperatures < 0):
        raise ValueError("All temperatures must be non-negative")

    # Check temperature range (20-30 μK expected)
    temp_min = np.min(node_data.temperatures)
    temp_max = np.max(node_data.temperatures)

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
