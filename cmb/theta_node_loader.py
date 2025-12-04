"""
Θ-node data loading utilities for CMB verification project.

Loads node geometry and depth data from data/theta/nodes/ directory.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import logging
from config.settings import get_config
from utils.io.data_loader import load_csv_data, load_json_data

logger = logging.getLogger(__name__)


def load_node_geometry(
    data_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Θ-node geometry data.

    Args:
        data_path: Path to node geometry data file.
                   If None, uses path from data index or default location.

    Returns:
        Tuple of (positions, scales)
        - positions: (N, 2) array of (theta, phi) in radians
        - scales: (N,) array of scales in parsec

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    if data_path is None:
        # Try to find from data index
        try:
            # Look for node geometry files in theta directory
            config = get_config()
            theta_dir = config.paths.data_theta / "nodes"

            # Try common file names
            for filename in [
                "node_geometry.csv",
                "nodes_geometry.csv",
                "node_geometry.json",
                "nodes_geometry.json",
            ]:
                candidate_path = theta_dir / filename
                if candidate_path.exists():
                    data_path = candidate_path
                    break

            if data_path is None:
                # Try to find any CSV or JSON in nodes directory
                if theta_dir.exists():
                    csv_files = list(theta_dir.glob("*.csv"))
                    json_files = list(theta_dir.glob("*.json"))
                    if csv_files:
                        data_path = csv_files[0]
                    elif json_files:
                        data_path = json_files[0]

            if data_path is None:
                raise FileNotFoundError(
                    f"Node geometry file not found. "
                    f"Expected in {theta_dir} or specify data_path."
                )
        except Exception as e:
            logger.warning(f"Could not find node geometry from index: {e}")
            # Fallback to default location
            config = get_config()
            data_path = config.paths.data_theta / "nodes" / "node_geometry.csv"

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Node geometry file not found: {data_path}")

    # Determine file format and load
    if data_path.suffix.lower() == ".csv":
        data = load_csv_data(data_path)

        # Expected columns: theta, phi, scale (or similar)
        # Try common column names
        theta_key = None
        phi_key = None
        scale_key = None

        for key in data.keys():
            key_lower = key.lower()
            if theta_key is None and ("theta" in key_lower or "colat" in key_lower):
                theta_key = key
            elif phi_key is None and "phi" in key_lower:
                phi_key = key
            elif scale_key is None and ("scale" in key_lower or "size" in key_lower):
                scale_key = key

        if theta_key is None or phi_key is None:
            raise ValueError(
                f"CSV file must contain 'theta' and 'phi' columns. "
                f"Found columns: {list(data.keys())}"
            )

        if scale_key is None:
            raise ValueError(
                f"CSV file must contain 'scale' column. "
                f"Found columns: {list(data.keys())}"
            )

        # Extract data
        theta = np.asarray(data[theta_key], dtype=float)
        phi = np.asarray(data[phi_key], dtype=float)
        scales = np.asarray(data[scale_key], dtype=float)

        # Convert to radians if needed (check if values are > 2π, likely degrees)
        if np.max(theta) > 2 * np.pi:
            theta = np.deg2rad(theta)
        if np.max(phi) > 2 * np.pi:
            phi = np.deg2rad(phi)

        # Validate ranges
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

        # Validate array lengths
        n_nodes = len(theta)
        if len(phi) != n_nodes or len(scales) != n_nodes:
            raise ValueError(
                f"Inconsistent array lengths: theta={len(theta)}, "
                f"phi={len(phi)}, scale={len(scales)}"
            )

        # Validate scales (should be ~300 pc)
        if np.any(scales <= 0):
            raise ValueError("All scales must be positive")

        # Check if scales are in arcmin (2-5 arcmin) and convert to pc
        config = get_config()
        if np.max(scales) < 10:  # Likely in arcmin
            scales = scales * config.constants.arcmin_to_pc_at_z1100

        positions = np.column_stack([theta, phi])

    elif data_path.suffix.lower() == ".json":
        data = load_json_data(data_path)

        # Expected structure: {"positions": [[theta, phi], ...], "scales": [...]}
        # or {"theta": [...], "phi": [...], "scale": [...]}
        if "positions" in data and "scales" in data:
            positions_list = data["positions"]
            scales = np.asarray(data["scales"], dtype=float)

            positions = np.asarray(positions_list, dtype=float)
            if positions.shape[1] != 2:
                raise ValueError(
                    f"Positions must have shape (N, 2), got {positions.shape}"
                )
        elif "theta" in data and "phi" in data and "scale" in data:
            theta = np.asarray(data["theta"], dtype=float)
            phi = np.asarray(data["phi"], dtype=float)
            scales = np.asarray(data["scale"], dtype=float)

            # Convert to radians if needed
            if np.max(theta) > 2 * np.pi:
                theta = np.deg2rad(theta)
            if np.max(phi) > 2 * np.pi:
                phi = np.deg2rad(phi)

            positions = np.column_stack([theta, phi])
        else:
            raise ValueError(
                f"JSON file must contain 'positions' and 'scales' or "
                f"'theta', 'phi', and 'scale' keys. "
                f"Found keys: {list(data.keys())}"
            )

        # Validate ranges
        theta = positions[:, 0]
        phi = positions[:, 1]
        if np.any(theta < 0) or np.any(theta > np.pi):
            raise ValueError("Theta values must be in [0, π]")
        if np.any(phi < 0) or np.any(phi > 2 * np.pi):
            raise ValueError("Phi values must be in [0, 2π]")

        # Validate array lengths
        n_nodes = len(positions)
        if len(scales) != n_nodes:
            raise ValueError(
                f"Inconsistent array lengths: positions={n_nodes}, "
                f"scales={len(scales)}"
            )

        # Validate scales
        if np.any(scales <= 0):
            raise ValueError("All scales must be positive")

        # Check if scales are in arcmin and convert to pc
        config = get_config()
        if np.max(scales) < 10:  # Likely in arcmin
            scales = scales * config.constants.arcmin_to_pc_at_z1100
    else:
        raise ValueError(
            f"Unsupported file format: {data_path.suffix}. "
            f"Supported formats: .csv, .json"
        )

    return positions, scales


def load_node_depths(data_path: Optional[Path] = None) -> np.ndarray:
    """
    Load Θ-node depth data (Δω/ω).

    Args:
        data_path: Path to node depth data file.
                   If None, uses path from data index or default location.

    Returns:
        Array of node depths (Δω/ω)

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    if data_path is None:
        # Try to find from data index
        try:
            config = get_config()
            theta_dir = config.paths.data_theta / "nodes"

            # Try common file names
            for filename in [
                "node_depths.csv",
                "nodes_depths.csv",
                "node_depths.json",
                "nodes_depths.json",
                "depths.csv",
                "depths.json",
            ]:
                candidate_path = theta_dir / filename
                if candidate_path.exists():
                    data_path = candidate_path
                    break

            if data_path is None:
                # Try to find any CSV or JSON with "depth" in name
                if theta_dir.exists():
                    depth_files = list(theta_dir.glob("*depth*.csv")) + list(
                        theta_dir.glob("*depth*.json")
                    )
                    if depth_files:
                        data_path = depth_files[0]

            if data_path is None:
                raise FileNotFoundError(
                    f"Node depth file not found. "
                    f"Expected in {theta_dir} or specify data_path."
                )
        except Exception as e:
            logger.warning(f"Could not find node depths from index: {e}")
            # Fallback to default location
            config = get_config()
            data_path = config.paths.data_theta / "nodes" / "node_depths.csv"

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Node depth file not found: {data_path}")

    # Determine file format and load
    if data_path.suffix.lower() == ".csv":
        data = load_csv_data(data_path)

        # Find depth column
        depth_key = None
        for key in data.keys():
            key_lower = key.lower()
            if "depth" in key_lower or "delta" in key_lower or "domega" in key_lower:
                depth_key = key
                break

        if depth_key is None:
            # Try first numeric column
            for key in data.keys():
                if isinstance(data[key], np.ndarray) and np.issubdtype(
                    data[key].dtype, np.floating
                ):
                    depth_key = key
                    break

        if depth_key is None:
            raise ValueError(
                f"CSV file must contain a depth column. "
                f"Found columns: {list(data.keys())}"
            )

        depths = np.asarray(data[depth_key], dtype=float)

    elif data_path.suffix.lower() == ".json":
        data = load_json_data(data_path)

        # Try common keys
        if "depths" in data:
            depths = np.asarray(data["depths"], dtype=float)
        elif "depth" in data:
            depths = np.asarray(data["depth"], dtype=float)
        elif "delta_omega" in data:
            depths = np.asarray(data["delta_omega"], dtype=float)
        else:
            # Try first array-like value
            for key, value in data.items():
                if isinstance(value, (list, np.ndarray)):
                    depths = np.asarray(value, dtype=float)
                    break
            else:
                raise ValueError(
                    f"JSON file must contain depth data. "
                    f"Found keys: {list(data.keys())}"
                )
    else:
        raise ValueError(
            f"Unsupported file format: {data_path.suffix}. "
            f"Supported formats: .csv, .json"
        )

    # Validate depths
    if np.any(depths < 0):
        raise ValueError("All depths must be non-negative")

    return depths
