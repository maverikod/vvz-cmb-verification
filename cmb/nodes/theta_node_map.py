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
from dataclasses import dataclass

from utils.cuda import (
    CudaArray,
    GridVectorizer,
    ElementWiseVectorizer,
    ReductionVectorizer,
)
from config.settings import Config


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


class ThetaNodeMapGenerator:
    """
    Θ-node map generator.

    Finds nodes as points of local minima ω_min(x) and classifies them
    by depth, area, and local curvature.

    Implements Module C from tech_spec-new.md.
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

        Args:
            node_positions: Array of node positions

        Returns:
            List of NodeClassification instances

        Raises:
            ValueError: If classification fails
        """
        if len(node_positions) == 0:
            return []

        classifications = []

        try:
            # Swap to GPU for processing
            self.omega_field_cuda.swap_to_gpu()

            # Process all nodes
            for node_id, position in enumerate(node_positions):
                # Convert position to tuple of ints for indexing
                pos_tuple = tuple(int(p) for p in position)

                # Calculate properties
                depth = self.calculate_node_depth(pos_tuple)
                area = self.calculate_node_area(pos_tuple)
                curvature = self.calculate_local_curvature(pos_tuple)

                # Create classification
                classification = NodeClassification(
                    node_id=node_id,
                    depth=depth,
                    area=area,
                    curvature=curvature,
                    position=tuple(float(p) for p in position),
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
        try:
            # Swap to GPU
            self.omega_field_cuda.swap_to_gpu()

            # Get omega value at node position
            omega_np = self.omega_field_cuda.to_numpy()
            omega_at_node = float(omega_np[node_position])

            # Find local minimum in neighborhood
            # Use GridVectorizer to find minimum in neighborhood
            # Create small neighborhood around node
            neighborhood_size = 5
            half_size = neighborhood_size // 2

            # Get slice bounds
            slices = []
            for i, pos in enumerate(node_position):
                start = max(0, int(pos) - half_size)
                end = min(omega_np.shape[i], int(pos) + half_size + 1)
                slices.append(slice(start, end))

            # Extract neighborhood
            neighborhood = omega_np[tuple(slices)]
            neighborhood_cuda = CudaArray(neighborhood, device="cpu")
            neighborhood_cuda.swap_to_gpu()

            # Find minimum in neighborhood using ReductionVectorizer
            omega_min = self.reduction_vec.vectorize_reduction(neighborhood_cuda, "min")

            # Convert to float
            if isinstance(omega_min, CudaArray):
                omega_min_val = float(omega_min.to_numpy().item())
            else:
                omega_min_val = float(omega_min)

            # Calculate depth: Δω/ω = (ω - ω_min) / ω
            if omega_at_node > 0:
                depth = (omega_at_node - omega_min_val) / omega_at_node
            else:
                depth = 0.0

            # Cleanup
            if neighborhood_cuda.device == "cuda":
                neighborhood_cuda.swap_to_cpu()
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()

            return depth

        except Exception as e:
            # Cleanup on error
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()
            raise ValueError(f"Depth calculation failed: {str(e)}") from e

    def calculate_node_area(self, node_position: Tuple[int, ...]) -> float:
        """
        Calculate node area (spatial extent).

        Args:
            node_position: Node position (grid indices)

        Returns:
            Node area in grid units or physical units
        """
        try:
            # Swap to GPU
            self.omega_field_cuda.swap_to_gpu()

            # Find connected region around node
            # Use local minimum mask to find connected area
            omega_np = self.omega_field_cuda.to_numpy()

            # Create mask for local minima in neighborhood
            neighborhood_size = 5
            half_size = neighborhood_size // 2

            # Get slice bounds
            slices = []
            for i, pos in enumerate(node_position):
                start = max(0, int(pos) - half_size)
                end = min(omega_np.shape[i], int(pos) + half_size + 1)
                slices.append(slice(start, end))

            # Extract neighborhood
            neighborhood = omega_np[tuple(slices)]
            neighborhood_cuda = CudaArray(neighborhood, device="cpu")
            neighborhood_cuda.swap_to_gpu()

            # Find local minima in neighborhood
            node_mask_cuda = self.grid_vec.vectorize_grid_operation(
                neighborhood_cuda, "local_minima", neighborhood_size=3
            )

            # Count connected pixels (area)
            node_mask = node_mask_cuda.to_numpy()
            area = float(np.sum(node_mask))

            # Convert to physical units if grid_spacing is provided
            if self.grid_spacing is not None:
                area = area * (self.grid_spacing ** len(node_position))

            # Cleanup
            if node_mask_cuda.device == "cuda":
                node_mask_cuda.swap_to_cpu()
            if neighborhood_cuda.device == "cuda":
                neighborhood_cuda.swap_to_cpu()
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()

            return area

        except Exception as e:
            # Cleanup on error
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()
            raise ValueError(f"Area calculation failed: {str(e)}") from e

    def calculate_local_curvature(self, node_position: Tuple[int, ...]) -> float:
        """
        Calculate local curvature at node position.

        Args:
            node_position: Node position (grid indices)

        Returns:
            Local curvature value
        """
        try:
            # Swap to GPU
            self.omega_field_cuda.swap_to_gpu()

            # Get neighborhood around node
            omega_np = self.omega_field_cuda.to_numpy()
            neighborhood_size = 5
            half_size = neighborhood_size // 2

            # Get slice bounds
            slices = []
            for i, pos in enumerate(node_position):
                start = max(0, int(pos) - half_size)
                end = min(omega_np.shape[i], int(pos) + half_size + 1)
                slices.append(slice(start, end))

            # Extract neighborhood
            neighborhood = omega_np[tuple(slices)]
            neighborhood_cuda = CudaArray(neighborhood, device="cpu")
            neighborhood_cuda.swap_to_gpu()

            # Calculate curvature using GridVectorizer
            curvature_cuda = self.grid_vec.vectorize_grid_operation(
                neighborhood_cuda, "curvature"
            )

            # Get curvature at center (node position)
            curvature_np = curvature_cuda.to_numpy()
            center_idx = tuple(half_size for _ in node_position)
            curvature_value = float(curvature_np[center_idx])

            # Cleanup
            if curvature_cuda.device == "cuda":
                curvature_cuda.swap_to_cpu()
            if neighborhood_cuda.device == "cuda":
                neighborhood_cuda.swap_to_cpu()
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()

            return curvature_value

        except Exception as e:
            # Cleanup on error
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()
            raise ValueError(f"Curvature calculation failed: {str(e)}") from e

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
            node_mask = self.create_node_mask(node_positions)

            # Create omega_min map
            omega_min_map = self.create_omega_min_map(node_positions)

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
            self.validate_node_map(node_map)

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
        try:
            # Swap to GPU
            self.omega_field_cuda.swap_to_gpu()

            # Use GridVectorizer to apply minimum filter
            local_minima_values_cuda = self.grid_vec.vectorize_grid_operation(
                self.omega_field_cuda,
                "minimum_filter",
                size=5,
                mode="constant",
                cval=np.inf,
            )

            # Convert to numpy for indexing
            omega_np = self.omega_field_cuda.to_numpy()
            omega_min_map = local_minima_values_cuda.to_numpy().copy()

            # Set node positions to actual omega values
            # using vectorized operations
            if len(node_positions) > 0:
                # Create node mask using numpy (fast for sparse indexing)
                node_mask = np.zeros(self.omega_field.shape, dtype=bool)

                if len(node_positions.shape) == 2:
                    # Multi-dimensional case - use advanced indexing
                    # Convert positions to integer tuples and filter valid ones
                    valid_positions = []
                    shape_tuple: Tuple[int, ...] = node_mask.shape
                    for pos in node_positions:
                        pos_tuple = tuple(int(p) for p in pos)
                        if all(0 <= p < s for p, s in zip(pos_tuple, shape_tuple)):
                            valid_positions.append(pos_tuple)
                    # Use advanced indexing for vectorized mask setting
                    if valid_positions:
                        indices = tuple(
                            np.array([p[i] for p in valid_positions])
                            for i in range(len(valid_positions[0]))
                        )
                        node_mask[indices] = True
                else:
                    # 1D case - use vectorized indexing
                    pos_ints = np.array([int(pos) for pos in node_positions])
                    valid_mask = (pos_ints >= 0) & (pos_ints < len(node_mask))
                    node_mask[pos_ints[valid_mask]] = True

                # Use ElementWiseVectorizer for vectorized combination
                # Create masks and values as CudaArrays
                node_mask_cuda = CudaArray(node_mask.astype(np.float32), device="cpu")
                node_mask_cuda.swap_to_gpu()

                omega_np_cuda = CudaArray(omega_np, device="cpu")
                omega_np_cuda.swap_to_gpu()

                local_minima_cuda = local_minima_values_cuda

                # Create complement mask
                ones_cuda = CudaArray(
                    np.ones_like(node_mask, dtype=np.float32), device="cpu"
                )
                ones_cuda.swap_to_gpu()
                non_node_mask_cuda = self.elem_vec.subtract(ones_cuda, node_mask_cuda)

                # Combine: omega_min_map = node_mask * omega
                # + non_node_mask * local_minima
                node_values = self.elem_vec.multiply(node_mask_cuda, omega_np_cuda)
                non_node_values = self.elem_vec.multiply(
                    non_node_mask_cuda, local_minima_cuda
                )
                omega_min_map_cuda = self.elem_vec.add(node_values, non_node_values)
                omega_min_map = omega_min_map_cuda.to_numpy()

                # Cleanup temporary arrays
                for arr in [
                    node_mask_cuda,
                    omega_np_cuda,
                    ones_cuda,
                    non_node_mask_cuda,
                    node_values,
                    non_node_values,
                ]:
                    if arr.device == "cuda":
                        arr.swap_to_cpu()

            # Cleanup
            if local_minima_values_cuda.device == "cuda":
                local_minima_values_cuda.swap_to_cpu()
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()

            return omega_min_map

        except Exception as e:
            # Cleanup on error
            if self.omega_field_cuda.device == "cuda":
                self.omega_field_cuda.swap_to_cpu()
            raise ValueError(f"Omega min map creation failed: {str(e)}") from e

    def create_node_mask(self, node_positions: np.ndarray) -> np.ndarray:
        """
        Create binary mask of node locations.

        Args:
            node_positions: Array of node positions

        Returns:
            Binary mask (1 at nodes, 0 elsewhere)
        """
        try:
            # Create mask
            node_mask = np.zeros(self.omega_field.shape, dtype=bool)

            if len(node_positions) == 0:
                # Convert to int32 using CudaArray for consistency
                mask_cuda = CudaArray(node_mask.astype(np.float32), device="cpu")
                return mask_cuda.to_numpy().astype(np.int32)

            # Set node positions to 1 using vectorized operations
            if len(node_positions.shape) == 2:
                # Multi-dimensional case - use advanced indexing
                # Convert positions to integer arrays and filter valid ones
                pos_arrays = [
                    np.array([int(p[i]) for p in node_positions])
                    for i in range(node_positions.shape[1])
                ]

                # Create validity mask for bounds checking
                valid_mask = np.ones(len(node_positions), dtype=bool)
                for i, pos_arr in enumerate(pos_arrays):
                    valid_mask &= (pos_arr >= 0) & (pos_arr < node_mask.shape[i])

                # Apply advanced indexing with valid positions only
                if np.any(valid_mask):
                    valid_indices = tuple(pos_arr[valid_mask] for pos_arr in pos_arrays)
                    node_mask[valid_indices] = True
            else:
                # 1D case - use vectorized indexing
                pos_ints = np.array([int(pos) for pos in node_positions])
                valid_mask = (pos_ints >= 0) & (pos_ints < len(node_mask))
                node_mask[pos_ints[valid_mask]] = True

            # Convert to int32 using CudaArray for consistency
            mask_cuda = CudaArray(node_mask.astype(np.float32), device="cpu")
            return mask_cuda.to_numpy().astype(np.int32)

        except Exception as e:
            raise ValueError(f"Node mask creation failed: {str(e)}") from e

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
        # Check shapes
        if node_map.omega_min_map.shape != node_map.node_mask.shape:
            raise ValueError(
                f"Shape mismatch: omega_min_map "
                f"{node_map.omega_min_map.shape} "
                f"!= node_mask {node_map.node_mask.shape}"
            )

        if node_map.omega_min_map.shape != node_map.grid_shape:
            raise ValueError(
                f"Shape mismatch: omega_min_map "
                f"{node_map.omega_min_map.shape} "
                f"!= grid_shape {node_map.grid_shape}"
            )

        # Check node classifications
        expected_nodes = node_map.metadata.get("num_nodes", 0)
        if len(node_map.node_classifications) != expected_nodes:
            raise ValueError(
                f"Classification count mismatch: "
                f"{len(node_map.node_classifications)} != {expected_nodes}"
            )

        # Check node mask values (should be 0 or 1)
        mask_valid = np.all((node_map.node_mask == 0) | (node_map.node_mask == 1))
        if not mask_valid:
            raise ValueError("Node mask contains values other than 0 and 1")

        # Check that all nodes have valid classifications
        for classification in node_map.node_classifications:
            if classification.depth < 0:
                raise ValueError(f"Node {classification.node_id} has negative depth")
            if classification.area < 0:
                raise ValueError(f"Node {classification.node_id} has negative area")

        return True
