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
from utils.cuda import CudaArray, ElementWiseVectorizer, ReductionVectorizer

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
        # Use CUDA-accelerated reduction for max check
        theta_cuda = CudaArray(theta, device="cpu")
        phi_cuda = CudaArray(phi, device="cpu")
        reduction_vec = ReductionVectorizer(use_gpu=True)

        theta_max = reduction_vec.vectorize_reduction(theta_cuda, "max")
        phi_max = reduction_vec.vectorize_reduction(phi_cuda, "max")

        if theta_max > 2 * np.pi:
            # Use CUDA-accelerated conversion
            elem_vec = ElementWiseVectorizer(use_gpu=True)
            theta_deg_cuda = CudaArray(np.deg2rad(theta), device="cpu")
            theta = theta_deg_cuda.to_numpy()
            if theta_deg_cuda.device == "cuda":
                theta_deg_cuda.swap_to_cpu()
        if phi_max > 2 * np.pi:
            # Use CUDA-accelerated conversion
            elem_vec = ElementWiseVectorizer(use_gpu=True)
            phi_deg_cuda = CudaArray(np.deg2rad(phi), device="cpu")
            phi = phi_deg_cuda.to_numpy()
            if phi_deg_cuda.device == "cuda":
                phi_deg_cuda.swap_to_cpu()

        # Recreate CUDA arrays after potential conversion
        theta_cuda = CudaArray(theta, device="cpu")
        phi_cuda = CudaArray(phi, device="cpu")

        # Validate ranges using CUDA-accelerated operations
        elem_vec = ElementWiseVectorizer(use_gpu=True)

        theta_lt_zero = elem_vec.vectorize_operation(theta_cuda, "less", 0.0)
        theta_gt_pi = elem_vec.vectorize_operation(theta_cuda, "greater", np.pi)
        has_theta_invalid = reduction_vec.vectorize_reduction(
            theta_lt_zero, "any"
        ) or reduction_vec.vectorize_reduction(theta_gt_pi, "any")

        if has_theta_invalid:
            theta_min = reduction_vec.vectorize_reduction(theta_cuda, "min")
            theta_max_val = reduction_vec.vectorize_reduction(theta_cuda, "max")
            raise ValueError(
                f"Theta values must be in [0, π]. "
                f"Found range: [{theta_min}, {theta_max_val}]"
            )

        phi_lt_zero = elem_vec.vectorize_operation(phi_cuda, "less", 0.0)
        phi_gt_2pi = elem_vec.vectorize_operation(phi_cuda, "greater", 2 * np.pi)
        has_phi_invalid = reduction_vec.vectorize_reduction(
            phi_lt_zero, "any"
        ) or reduction_vec.vectorize_reduction(phi_gt_2pi, "any")

        if has_phi_invalid:
            phi_min = reduction_vec.vectorize_reduction(phi_cuda, "min")
            phi_max_val = reduction_vec.vectorize_reduction(phi_cuda, "max")
            raise ValueError(
                f"Phi values must be in [0, 2π]. "
                f"Found range: [{phi_min}, {phi_max_val}]"
            )

        # Cleanup GPU memory
        if theta_cuda.device == "cuda":
            theta_cuda.swap_to_cpu()
        if phi_cuda.device == "cuda":
            phi_cuda.swap_to_cpu()

        # Validate array lengths
        n_nodes = len(theta)
        if len(phi) != n_nodes or len(scales) != n_nodes:
            raise ValueError(
                f"Inconsistent array lengths: theta={len(theta)}, "
                f"phi={len(phi)}, scale={len(scales)}"
            )

        # Validate scales (should be ~300 pc) using CUDA
        scales_cuda = CudaArray(scales, device="cpu")
        elem_vec = ElementWiseVectorizer(use_gpu=True)
        reduction_vec = ReductionVectorizer(use_gpu=True)

        scales_le_zero = elem_vec.vectorize_operation(scales_cuda, "less_equal", 0.0)
        has_scale_invalid = reduction_vec.vectorize_reduction(scales_le_zero, "any")
        if has_scale_invalid:
            raise ValueError("All scales must be positive")

        # Check if scales are in arcmin (2-5 arcmin) and convert to pc
        config = get_config()
        scales_max = reduction_vec.vectorize_reduction(scales_cuda, "max")
        if scales_max < 10:  # Likely in arcmin
            # Use CUDA-accelerated multiplication
            scales_cuda = elem_vec.multiply(
                scales_cuda, config.constants.arcmin_to_pc_at_z1100
            )
            scales = scales_cuda.to_numpy()

        # Cleanup GPU memory
        if scales_cuda.device == "cuda":
            scales_cuda.swap_to_cpu()

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

            # Convert to radians if needed using CUDA
            theta_cuda = CudaArray(theta, device="cpu")
            phi_cuda = CudaArray(phi, device="cpu")
            reduction_vec = ReductionVectorizer(use_gpu=True)

            theta_max = reduction_vec.vectorize_reduction(theta_cuda, "max")
            phi_max = reduction_vec.vectorize_reduction(phi_cuda, "max")

            if theta_max > 2 * np.pi:
                theta_deg_cuda = CudaArray(np.deg2rad(theta), device="cpu")
                theta = theta_deg_cuda.to_numpy()
                if theta_deg_cuda.device == "cuda":
                    theta_deg_cuda.swap_to_cpu()
                theta_cuda = CudaArray(theta, device="cpu")
            if phi_max > 2 * np.pi:
                phi_deg_cuda = CudaArray(np.deg2rad(phi), device="cpu")
                phi = phi_deg_cuda.to_numpy()
                if phi_deg_cuda.device == "cuda":
                    phi_deg_cuda.swap_to_cpu()
                phi_cuda = CudaArray(phi, device="cpu")

            positions = np.column_stack([theta, phi])
        else:
            raise ValueError(
                f"JSON file must contain 'positions' and 'scales' or "
                f"'theta', 'phi', and 'scale' keys. "
                f"Found keys: {list(data.keys())}"
            )

        # Validate ranges using CUDA
        theta = positions[:, 0]
        phi = positions[:, 1]
        theta_cuda = CudaArray(theta, device="cpu")
        phi_cuda = CudaArray(phi, device="cpu")
        elem_vec = ElementWiseVectorizer(use_gpu=True)
        reduction_vec = ReductionVectorizer(use_gpu=True)

        theta_lt_zero = elem_vec.vectorize_operation(theta_cuda, "less", 0.0)
        theta_gt_pi = elem_vec.vectorize_operation(theta_cuda, "greater", np.pi)
        has_theta_invalid = reduction_vec.vectorize_reduction(
            theta_lt_zero, "any"
        ) or reduction_vec.vectorize_reduction(theta_gt_pi, "any")
        if has_theta_invalid:
            raise ValueError("Theta values must be in [0, π]")

        phi_lt_zero = elem_vec.vectorize_operation(phi_cuda, "less", 0.0)
        phi_gt_2pi = elem_vec.vectorize_operation(phi_cuda, "greater", 2 * np.pi)
        has_phi_invalid = reduction_vec.vectorize_reduction(
            phi_lt_zero, "any"
        ) or reduction_vec.vectorize_reduction(phi_gt_2pi, "any")
        if has_phi_invalid:
            raise ValueError("Phi values must be in [0, 2π]")

        # Cleanup GPU memory
        if theta_cuda.device == "cuda":
            theta_cuda.swap_to_cpu()
        if phi_cuda.device == "cuda":
            phi_cuda.swap_to_cpu()

        # Validate array lengths
        n_nodes = len(positions)
        if len(scales) != n_nodes:
            raise ValueError(
                f"Inconsistent array lengths: positions={n_nodes}, "
                f"scales={len(scales)}"
            )

        # Validate scales using CUDA
        scales_cuda = CudaArray(scales, device="cpu")
        elem_vec = ElementWiseVectorizer(use_gpu=True)
        reduction_vec = ReductionVectorizer(use_gpu=True)

        scales_le_zero = elem_vec.vectorize_operation(scales_cuda, "less_equal", 0.0)
        has_scale_invalid = reduction_vec.vectorize_reduction(scales_le_zero, "any")
        if has_scale_invalid:
            raise ValueError("All scales must be positive")

        # Check if scales are in arcmin and convert to pc
        config = get_config()
        scales_max = reduction_vec.vectorize_reduction(scales_cuda, "max")
        if scales_max < 10:  # Likely in arcmin
            scales_cuda = elem_vec.multiply(
                scales_cuda, config.constants.arcmin_to_pc_at_z1100
            )
            scales = scales_cuda.to_numpy()

        # Cleanup GPU memory
        if scales_cuda.device == "cuda":
            scales_cuda.swap_to_cpu()
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

    # Validate depths using CUDA
    depths_cuda = CudaArray(depths, device="cpu")
    elem_vec = ElementWiseVectorizer(use_gpu=True)
    reduction_vec = ReductionVectorizer(use_gpu=True)

    depths_lt_zero = elem_vec.vectorize_operation(depths_cuda, "less", 0.0)
    has_depth_invalid = reduction_vec.vectorize_reduction(depths_lt_zero, "any")
    if has_depth_invalid:
        raise ValueError("All depths must be non-negative")

    # Cleanup GPU memory
    if depths_cuda.device == "cuda":
        depths_cuda.swap_to_cpu()

    return depths
