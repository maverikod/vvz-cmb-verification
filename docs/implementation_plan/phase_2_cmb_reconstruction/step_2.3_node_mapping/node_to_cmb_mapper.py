"""
Node-to-CMB map mapping module.

Maps Θ-nodes from early universe to current sky coordinates
and creates node catalog with CMB positions.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
import healpy as hp
from dataclasses import dataclass
from cmb.theta_node_processor import ThetaNodeData
from cmb.reconstruction.cmb_map_reconstructor import CmbMapReconstructor
from config.settings import Config


@dataclass
class NodeCmbMapping:
    """
    Node-to-CMB mapping data.

    Attributes:
        node_ids: Array of node identifiers
        sky_positions: Array of (theta, phi) sky coordinates
        pixel_indices: Array of HEALPix pixel indices
        cmb_values: CMB map values at node positions
        node_depths: Node depths (Δω/ω)
        node_temperatures: Node temperatures (ΔT in μK)
    """

    node_ids: np.ndarray
    sky_positions: np.ndarray  # Shape: (N, 2) for (theta, phi)
    pixel_indices: np.ndarray
    cmb_values: np.ndarray
    node_depths: np.ndarray
    node_temperatures: np.ndarray


class NodeToCmbMapper:
    """
    Maps Θ-nodes to CMB map positions.

    Handles coordinate transformation from early universe
    (z≈1100) to current sky coordinates.
    """

    def __init__(self, node_data: ThetaNodeData, cmb_map: np.ndarray, nside: int):
        """
        Initialize mapper.

        Args:
            node_data: Θ-node data
            cmb_map: Reconstructed CMB map
            nside: HEALPix NSIDE parameter
        """
        self.node_data = node_data
        self.cmb_map = cmb_map
        self.nside = nside

    def map_nodes_to_sky(self) -> np.ndarray:
        """
        Map nodes from early universe to sky coordinates.

        Handles cosmological projection from z≈1100 to z=0.

        Returns:
            Array of (theta, phi) sky coordinates

        Raises:
            ValueError: If mapping fails
        """
        pass

    def create_node_catalog(self) -> NodeCmbMapping:
        """
        Create node catalog with CMB positions.

        Returns:
            NodeCmbMapping instance with complete mapping

        Raises:
            ValueError: If catalog creation fails
        """
        pass

    def get_nodes_at_position(
        self, theta: float, phi: float, radius: float = 0.01
    ) -> List[int]:
        """
        Get nodes near given sky position.

        Args:
            theta: Sky theta coordinate (radians)
            phi: Sky phi coordinate (radians)
            radius: Search radius (radians)

        Returns:
            List of node IDs near position
        """
        pass

    def get_cmb_value_at_node(self, node_id: int) -> float:
        """
        Get CMB map value at node position.

        Args:
            node_id: Node identifier

        Returns:
            CMB map value at node position

        Raises:
            ValueError: If node_id is invalid
        """
        pass
