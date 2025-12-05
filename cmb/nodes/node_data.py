"""
Θ-node data structures.

Defines data classes for node classification and node map data.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Tuple, List
import numpy as np
from dataclasses import dataclass


@dataclass
class NodeClassification:
    """
    Node classification data.

    Attributes:
        node_id: Node identifier
        depth: Node depth (Δω/ω)
        area: Node area (spatial extent)
        curvature: Local curvature at node
        position: Node position (x, y, z or sky coordinates)
    """

    node_id: int
    depth: float
    area: float
    curvature: float
    position: Tuple[float, ...]


@dataclass
class ThetaNodeMap:
    """
    Θ-node map data.

    Attributes:
        omega_min_map: Map of ω_min(x) values
        node_mask: Binary mask of node locations
        node_classifications: List of node classifications
        grid_shape: Shape of input grid
        metadata: Additional metadata
    """

    omega_min_map: np.ndarray
    node_mask: np.ndarray
    node_classifications: List[NodeClassification]
    grid_shape: Tuple[int, ...]
    metadata: dict
