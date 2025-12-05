"""
Θ-node map generation module.

Implements Module C from tech_spec-new.md: generates Θ-node map
from field ω(x) by finding nodes as points of local minima.

Nodes are topological structures of Θ-field. Matter does NOT influence Θ-field.
Nodes are NOT created by matter - matter "sits" in nodes.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Tuple, List
import numpy as np

from utils.cuda import (
    CudaArray,
    GridVectorizer,
    ElementWiseVectorizer,
    ReductionVectorizer,
)
from config.settings import Config

from cmb.nodes.node_data import NodeClassification, ThetaNodeMap
from cmb.nodes.node_properties import NodePropertiesCalculator
from cmb.nodes.node_map_creator import NodeMapCreator
from cmb.nodes.node_map_validator import NodeMapValidator


class ThetaNodeMapGenerator:
    """
    Θ-node map generator.

    Finds nodes as points of local minima ω_min(x) and classifies them
    by depth, area, and local curvature.

    Implements Module C from tech_spec-new.md.

    This is a facade class that coordinates:
    - NodePropertiesCalculator: calculates node properties
    - NodeMapCreator: creates node maps
    - NodeMapValidator: validates node maps
    """

    def __init__(
        self,
        omega_field: np.ndarray,
        grid_spacing: Optional[float] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize node map generator.

        Args:
            omega_field: Field ω(x) on grid
            grid_spacing: Grid spacing (if None, uses config)
            config: Configuration instance (uses global if None)
        """
        self.omega_field = omega_field
        self.grid_spacing = grid_spacing
        self.config = config or Config.load()

        # Initialize CUDA arrays
        self.omega_field_cuda = CudaArray(omega_field, device="cpu")

        # Initialize vectorizers
        self.grid_vec = GridVectorizer(use_gpu=True)
        self.elem_vec = ElementWiseVectorizer(use_gpu=True)
        self.reduction_vec = ReductionVectorizer(use_gpu=True)

        # Initialize helper classes
        self.properties_calc = NodePropertiesCalculator(
            self.omega_field_cuda,
            self.grid_vec,
            self.reduction_vec,
            self.grid_spacing,
        )
        self.map_creator = NodeMapCreator(
            self.omega_field_cuda,
            self.grid_vec,
            self.elem_vec,
            self.omega_field.shape,
        )
        self.validator = NodeMapValidator()

    def find_nodes(self, neighborhood_size: int = 5) -> np.ndarray:
        """
        Find nodes as points of local minima.

        Node definition: x_node = {x : ω(x) = ω_min(x)}

        Uses local minimum detection to find points where
        ω(x) = ω_min(x) locally.

        Args:
            neighborhood_size: Size of neighborhood for local minimum detection

        Returns:
            Array of node positions (indices or coordinates)

        Raises:
            ValueError: If node detection fails
        """
        try:
            # Swap to GPU for processing
            self.omega_field_cuda.swap_to_gpu()

            # Find local minima using GridVectorizer
            node_mask_cuda = self.grid_vec.vectorize_grid_operation(
                self.omega_field_cuda,
                "local_minima",
                neighborhood_size=neighborhood_size,
            )

            # Convert to numpy for finding indices
            node_mask = node_mask_cuda.to_numpy()

            # Find node positions using numpy where (for indexing)
            node_indices = np.where(node_mask)

            # Convert to array of positions
            if len(node_indices) == 1:
                # 1D case
                node_positions = np.column_stack([node_indices[0]])
            else:
                # Multi-dimensional case
                node_positions = np.column_stack(node_indices)

            # Cleanup GPU memory
            if node_mask_cuda.device == "cuda":
                node_mask_cuda.swap_to_cpu()
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()

            return node_positions

        except Exception as e:
            # Cleanup on error
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()
            raise ValueError(f"Node detection failed: {str(e)}") from e

    def classify_nodes(self, node_positions: np.ndarray) -> List[NodeClassification]:
        """
        Classify nodes by depth, area, and local curvature.

        Uses vectorized batch processing for all nodes simultaneously.

        Args:
            node_positions: Array of node positions

        Returns:
            List of NodeClassification instances

        Raises:
            ValueError: If classification fails
        """
        if len(node_positions) == 0:
            return []

        try:
            # Swap to GPU for processing
            self.omega_field_cuda.swap_to_gpu()

            # Calculate all properties in batch using vectorized operations
            depths, areas, curvatures = (
                self.properties_calc.calculate_node_properties_batch(node_positions)
            )

            # Create classifications from batch results
            # Vectorize position conversion using numpy
            n_nodes = len(node_positions)
            classifications = []
            # Convert positions to tuples using vectorized numpy operations
            if len(node_positions.shape) == 2:
                # Multi-dimensional: use numpy array operations (vectorized)
                # Convert to float array first, then create tuples
                positions_float = node_positions.astype(float)
                positions_tuples = [tuple(positions_float[i]) for i in range(n_nodes)]
            else:
                # 1D: use numpy array operations (vectorized)
                positions_float = node_positions.astype(float)
                positions_tuples = [
                    (float(positions_float[i]),) for i in range(n_nodes)
                ]

            # Create classifications (object creation still requires iteration)
            # But use vectorized numpy arrays for data access
            depths_array = np.asarray(depths, dtype=float)
            areas_array = np.asarray(areas, dtype=float)
            curvatures_array = np.asarray(curvatures, dtype=float)

            for node_id in range(n_nodes):
                classification = NodeClassification(
                    node_id=node_id,
                    depth=float(depths_array[node_id]),
                    area=float(areas_array[node_id]),
                    curvature=float(curvatures_array[node_id]),
                    position=positions_tuples[node_id],
                )
                classifications.append(classification)

            # Cleanup GPU memory
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()

            return classifications

        except Exception as e:
            # Cleanup on error
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()
            raise ValueError(f"Node classification failed: {str(e)}") from e

    def calculate_node_depth(self, node_position: Tuple[int, ...]) -> float:
        """
        Calculate node depth (Δω/ω).

        Formula: Δω = ω - ω_min
        where ω_min is local minimum at node position.

        Args:
            node_position: Node position (grid indices)

        Returns:
            Node depth (Δω/ω)
        """
        return self.properties_calc.calculate_node_depth(node_position)

    def calculate_node_area(self, node_position: Tuple[int, ...]) -> float:
        """
        Calculate node area (spatial extent).

        Args:
            node_position: Node position (grid indices)

        Returns:
            Node area in grid units or physical units
        """
        return self.properties_calc.calculate_node_area(node_position)

    def calculate_local_curvature(self, node_position: Tuple[int, ...]) -> float:
        """
        Calculate local curvature at node position.

        Args:
            node_position: Node position (grid indices)

        Returns:
            Local curvature value
        """
        return self.properties_calc.calculate_local_curvature(node_position)

    def generate_node_map(self, neighborhood_size: int = 5) -> ThetaNodeMap:
        """
        Generate complete Θ-node map.

        Finds nodes, classifies them, and creates maps.

        Args:
            neighborhood_size: Size of neighborhood for local minimum detection

        Returns:
            ThetaNodeMap instance with:
            - ω_min(x) map
            - Node mask
            - Node classifications

        Raises:
            ValueError: If map generation fails
        """
        try:
            # Find nodes
            node_positions = self.find_nodes(neighborhood_size=neighborhood_size)

            # Create node mask
            node_mask = self.map_creator.create_node_mask(node_positions)

            # Create omega_min map
            omega_min_map = self.map_creator.create_omega_min_map(node_positions)

            # Classify nodes
            node_classifications = self.classify_nodes(node_positions)

            # Create metadata
            metadata = {
                "num_nodes": len(node_positions),
                "neighborhood_size": neighborhood_size,
                "grid_spacing": self.grid_spacing,
            }

            # Create ThetaNodeMap
            node_map = ThetaNodeMap(
                omega_min_map=omega_min_map,
                node_mask=node_mask,
                node_classifications=node_classifications,
                grid_shape=self.omega_field.shape,
                metadata=metadata,
            )

            # Validate map
            self.validator.validate_node_map(node_map)

            return node_map

        except Exception as e:
            raise ValueError(f"Node map generation failed: {str(e)}") from e

    def create_omega_min_map(self, node_positions: np.ndarray) -> np.ndarray:
        """
        Create map of ω_min(x) values.

        Args:
            node_positions: Array of node positions

        Returns:
            Map of ω_min(x) values at each grid point
        """
        return self.map_creator.create_omega_min_map(node_positions)

    def create_node_mask(self, node_positions: np.ndarray) -> np.ndarray:
        """
        Create binary mask of node locations.

        Args:
            node_positions: Array of node positions

        Returns:
            Binary mask (1 at nodes, 0 elsewhere)
        """
        return self.map_creator.create_node_mask(node_positions)

    def validate_node_map(self, node_map: ThetaNodeMap) -> bool:
        """
        Validate generated node map.

        Args:
            node_map: ThetaNodeMap to validate

        Returns:
            True if map is valid

        Raises:
            ValueError: If map is invalid
        """
        return self.validator.validate_node_map(node_map)
