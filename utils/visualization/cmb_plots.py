"""
CMB map visualization utilities.

Provides functions for plotting HEALPix CMB maps and temperature fluctuations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from utils.cuda import CudaArray, ReductionVectorizer


def plot_healpix_map(
    map_data: np.ndarray,
    title: str = "CMB Map",
    output_path: Optional[Path] = None,
    projection: str = "mollweide",
    unit: str = "μK",
) -> None:
    """
    Plot HEALPix CMB map.

    Args:
        map_data: HEALPix map array
        title: Plot title
        output_path: Path to save plot (optional)
        projection: Map projection ('mollweide', 'orthographic', etc.)
        unit: Temperature unit for colorbar

    Raises:
        ValueError: If map data is invalid
    """
    if not isinstance(map_data, np.ndarray):
        raise ValueError("Map data must be a numpy array")

    if map_data.size == 0:
        raise ValueError("Map data is empty")

    try:
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        fig.add_subplot(111, projection=projection)

        # Plot HEALPix map
        hp.mollview(
            map_data,
            title=title,
            unit=unit,
            fig=fig.number,
            return_projected_map=False,
        )

        plt.tight_layout()

        # Save if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    except Exception as e:
        raise ValueError(f"Failed to plot HEALPix map: {e}") from e


def plot_temperature_fluctuations(
    map_data: np.ndarray,
    title: str = "Temperature Fluctuations",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot temperature fluctuation map.

    Args:
        map_data: HEALPix temperature map
        title: Plot title
        output_path: Path to save plot (optional)
    """
    if not isinstance(map_data, np.ndarray):
        raise ValueError("Map data must be a numpy array")

    # Remove monopole and dipole if needed
    # Use CUDA-accelerated reduction for mean calculation
    map_data_cuda = CudaArray(map_data, device="cpu")
    reduction_vec = ReductionVectorizer(use_gpu=True)
    mean_result = reduction_vec.vectorize_reduction(map_data_cuda, "mean")

    # Convert to float
    if isinstance(mean_result, CudaArray):
        mean_value = float(mean_result.to_numpy().item())
    else:
        mean_value = float(mean_result)

    # Calculate fluctuations using CUDA-accelerated subtraction
    # Use scalar operand for subtraction (ElementWiseVectorizer supports scalars)
    from utils.cuda import ElementWiseVectorizer

    elem_vec = ElementWiseVectorizer(use_gpu=True)
    map_fluctuations_cuda = elem_vec.subtract(map_data_cuda, mean_value)
    map_fluctuations = map_fluctuations_cuda.to_numpy()

    # Cleanup GPU memory
    if map_data_cuda.device == "cuda":
        map_data_cuda.swap_to_cpu()
    if map_fluctuations_cuda.device == "cuda":
        map_fluctuations_cuda.swap_to_cpu()

    try:
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        hp.mollview(
            map_fluctuations,
            title=title,
            unit="μK",
            fig=fig.number,
        )

        plt.tight_layout()

        # Save if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    except Exception as e:
        raise ValueError(f"Failed to plot temperature fluctuations: {e}") from e


def plot_map_comparison(
    map1: np.ndarray,
    map2: np.ndarray,
    title: str = "Map Comparison",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot comparison of two CMB maps.

    Args:
        map1: First HEALPix map
        map2: Second HEALPix map
        title: Plot title
        output_path: Path to save plot (optional)
    """
    if not isinstance(map1, np.ndarray) or not isinstance(map2, np.ndarray):
        raise ValueError("Both maps must be numpy arrays")

    if map1.size != map2.size:
        raise ValueError(f"Map sizes must match: {map1.size} vs {map2.size}")

    try:
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 6))

        # Plot first map
        fig.add_subplot(131)
        hp.mollview(map1, title="Map 1", unit="μK", fig=fig.number, sub=(1, 3, 1))

        # Plot second map
        fig.add_subplot(132)
        hp.mollview(map2, title="Map 2", unit="μK", fig=fig.number, sub=(1, 3, 2))

        # Plot difference
        diff_map = map1 - map2
        fig.add_subplot(133)
        hp.mollview(
            diff_map, title="Difference", unit="μK", fig=fig.number, sub=(1, 3, 3)
        )

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        # Save if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    except Exception as e:
        raise ValueError(f"Failed to plot map comparison: {e}") from e


def plot_node_overlay(
    map_data: np.ndarray,
    node_positions: np.ndarray,
    title: str = "CMB Map with Nodes",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot CMB map with node positions overlaid.

    Args:
        map_data: HEALPix map array
        node_positions: Array of (theta, phi) node positions
            Shape: (N, 2) where N is number of nodes
            theta: colatitude in radians (0 to π)
            phi: azimuth in radians (0 to 2π)
        title: Plot title
        output_path: Path to save plot (optional)
    """
    if not isinstance(map_data, np.ndarray):
        raise ValueError("Map data must be a numpy array")

    if not isinstance(node_positions, np.ndarray):
        raise ValueError("Node positions must be a numpy array")

    if node_positions.shape[1] != 2:
        raise ValueError(
            f"Node positions must have shape (N, 2), got {node_positions.shape}"
        )

    try:
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        hp.mollview(map_data, title=title, unit="μK", fig=fig.number)

        # Convert (theta, phi) to (ra, dec) for plotting
        # theta is colatitude, phi is azimuth
        # For HEALPix: theta = colatitude, phi = longitude
        ra = node_positions[:, 1]  # phi is longitude
        dec = np.pi / 2 - node_positions[:, 0]  # dec = π/2 - theta

        # Convert to degrees for plotting
        ra_deg = np.degrees(ra)
        dec_deg = np.degrees(dec)

        # Overlay node positions
        # Note: This is a simplified overlay - may need adjustment
        # based on actual HEALPix coordinate system
        ax = plt.gca()
        ax.scatter(ra_deg, dec_deg, c="red", s=20, marker="x", label="Nodes")

        plt.legend()
        plt.tight_layout()

        # Save if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    except Exception as e:
        raise ValueError(f"Failed to plot node overlay: {e}") from e
