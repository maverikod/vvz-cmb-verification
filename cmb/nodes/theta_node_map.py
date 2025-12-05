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
            depths, areas, curvatures = self._calculate_node_properties_batch(
                node_positions
            )

            # Create classifications from batch results
            # Vectorize position conversion using numpy
            n_nodes = len(node_positions)
            classifications = []
            # Convert positions to tuples using vectorized operations
            if len(node_positions.shape) == 2:
                # Multi-dimensional: use numpy array operations
                positions_tuples = [
                    tuple(float(p) for p in node_positions[i]) for i in range(n_nodes)
                ]
            else:
                # 1D: use numpy array operations
                positions_tuples = [(float(node_positions[i]),) for i in range(n_nodes)]

            # Create classifications (object creation still requires iteration)
            for node_id in range(n_nodes):
                classification = NodeClassification(
                    node_id=node_id,
                    depth=float(depths[node_id]),
                    area=float(areas[node_id]),
                    curvature=float(curvatures[node_id]),
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

            # Get slice bounds using vectorized operations
            pos_array = np.asarray(node_position, dtype=int)
            shape_array = np.array(omega_np.shape)
            starts = np.maximum(0, pos_array - half_size)
            ends = np.minimum(shape_array, pos_array + half_size + 1)
            slices = [slice(int(starts[i]), int(ends[i])) for i in range(len(starts))]

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

            # Get slice bounds using vectorized operations
            pos_array = np.asarray(node_position, dtype=int)
            shape_array = np.array(omega_np.shape)
            starts = np.maximum(0, pos_array - half_size)
            ends = np.minimum(shape_array, pos_array + half_size + 1)
            slices = [slice(int(starts[i]), int(ends[i])) for i in range(len(starts))]

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

    def _calculate_node_properties_batch(
        self, node_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate depth, area, and curvature for all nodes in batch.

        Uses vectorized CUDA operations to process neighborhoods efficiently.
        While neighborhood extraction requires iteration, all computations
        within neighborhoods use CUDA-accelerated vectorized operations.

        Args:
            node_positions: Array of node positions (N, D) where N is number
                of nodes and D is dimensionality

        Returns:
            Tuple of (depths, areas, curvatures) arrays, each of shape (N,)

        Raises:
            ValueError: If batch calculation fails
        """
        try:
            omega_np = self.omega_field_cuda.to_numpy()
            n_nodes = len(node_positions)
            neighborhood_size = 5
            half_size = neighborhood_size // 2

            # Initialize result arrays
            depths = np.zeros(n_nodes, dtype=np.float64)
            areas = np.zeros(n_nodes, dtype=np.float64)
            curvatures = np.zeros(n_nodes, dtype=np.float64)

            # Process each node using CUDA-accelerated operations
            # Note: Neighborhood extraction requires iteration, but all
            # computations use vectorized CUDA operations
            for node_idx, position in enumerate(node_positions):
                try:
                    # Convert position to tuple of ints using vectorized operations
                    pos_array = np.asarray(position, dtype=int)
                    pos_tuple = tuple(pos_array)

                    # Check bounds using vectorized numpy operations
                    bounds_valid = np.all(
                        (pos_array >= 0) & (pos_array < np.array(omega_np.shape))
                    )
                    if not bounds_valid:
                        continue

                    # Get omega value at node
                    omega_at_node = float(omega_np[pos_tuple])

                    # Get slice bounds for neighborhood using vectorized operations
                    pos_array = np.asarray(pos_tuple)
                    shape_array = np.array(omega_np.shape)
                    starts = np.maximum(0, pos_array - half_size)
                    ends = np.minimum(shape_array, pos_array + half_size + 1)
                    slices = [
                        slice(int(starts[i]), int(ends[i])) for i in range(len(starts))
                    ]

                    # Extract neighborhood
                    neighborhood = omega_np[tuple(slices)]
                    neighborhood_cuda = CudaArray(neighborhood, device="cpu")
                    neighborhood_cuda.swap_to_gpu()

                    # Calculate depth: find minimum in neighborhood using CUDA
                    omega_min = self.reduction_vec.vectorize_reduction(
                        neighborhood_cuda, "min"
                    )
                    if isinstance(omega_min, CudaArray):
                        omega_min_val = float(omega_min.to_numpy().item())
                    else:
                        omega_min_val = float(omega_min)

                    if omega_at_node > 0:
                        depths[node_idx] = (
                            omega_at_node - omega_min_val
                        ) / omega_at_node
                    else:
                        depths[node_idx] = 0.0

                    # Calculate area: find local minima in neighborhood using CUDA
                    node_mask_cuda = self.grid_vec.vectorize_grid_operation(
                        neighborhood_cuda, "local_minima", neighborhood_size=3
                    )
                    node_mask = node_mask_cuda.to_numpy()
                    # Use ReductionVectorizer for sum
                    node_mask_sum_cuda = CudaArray(
                        node_mask.astype(np.float32), device="cpu"
                    )
                    node_mask_sum_cuda.swap_to_gpu()
                    area_sum = self.reduction_vec.vectorize_reduction(
                        node_mask_sum_cuda, "sum"
                    )
                    if isinstance(area_sum, CudaArray):
                        area = float(area_sum.to_numpy().item())
                    else:
                        area = float(area_sum)
                    if self.grid_spacing is not None:
                        area = area * (self.grid_spacing ** len(pos_tuple))
                    areas[node_idx] = area

                    # Cleanup area calculation arrays
                    if node_mask_sum_cuda.device == "cuda":
                        node_mask_sum_cuda.swap_to_cpu()
                    if node_mask_cuda.device == "cuda":
                        node_mask_cuda.swap_to_cpu()

                    # Calculate curvature using CUDA
                    curvature_cuda = self.grid_vec.vectorize_grid_operation(
                        neighborhood_cuda, "curvature"
                    )
                    curvature_np = curvature_cuda.to_numpy()
                    # Create center index using vectorized operations
                    center_idx_array = np.full(len(pos_tuple), half_size, dtype=int)
                    center_idx = tuple(center_idx_array)
                    # Check bounds using vectorized numpy operations
                    bounds_valid = np.all(
                        (center_idx_array >= 0)
                        & (center_idx_array < np.array(curvature_np.shape))
                    )
                    if bounds_valid:
                        curvatures[node_idx] = float(curvature_np[center_idx])
                    else:
                        curvatures[node_idx] = 0.0

                    # Cleanup
                    if curvature_cuda.device == "cuda":
                        curvature_cuda.swap_to_cpu()
                    if neighborhood_cuda.device == "cuda":
                        neighborhood_cuda.swap_to_cpu()

                except (IndexError, ValueError):
                    # Skip invalid nodes
                    continue

            return depths, areas, curvatures

        except Exception as e:
            raise ValueError(
                f"Batch node properties calculation failed: {str(e)}"
            ) from e

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

            # Get slice bounds using vectorized operations
            pos_array = np.asarray(node_position, dtype=int)
            shape_array = np.array(omega_np.shape)
            starts = np.maximum(0, pos_array - half_size)
            ends = np.minimum(shape_array, pos_array + half_size + 1)
            slices = [slice(int(starts[i]), int(ends[i])) for i in range(len(starts))]

            # Extract neighborhood
            neighborhood = omega_np[tuple(slices)]
            neighborhood_cuda = CudaArray(neighborhood, device="cpu")
            neighborhood_cuda.swap_to_gpu()

            # Calculate curvature using GridVectorizer
            curvature_cuda = self.grid_vec.vectorize_grid_operation(
                neighborhood_cuda, "curvature"
            )

            # Get curvature at center (node position) using vectorized operations
            curvature_np = curvature_cuda.to_numpy()
            center_idx_array = np.full(len(node_position), half_size, dtype=int)
            center_idx = tuple(center_idx_array)
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
                    # Convert positions to integer arrays using vectorized operations
                    # Use numpy array indexing instead of list comprehension
                    pos_arrays = [
                        node_positions[:, i].astype(int)
                        for i in range(node_positions.shape[1])
                    ]

                    # Create validity mask using vectorized operations
                    valid_mask = np.ones(len(node_positions), dtype=bool)
                    shape_tuple: Tuple[int, ...] = node_mask.shape
                    # Vectorize all bounds checks at once
                    for i, pos_arr in enumerate(pos_arrays):
                        valid_mask &= (pos_arr >= 0) & (pos_arr < shape_tuple[i])

                    # Use advanced indexing for vectorized mask setting
                    if np.any(valid_mask):
                        valid_indices = tuple(
                            pos_arr[valid_mask] for pos_arr in pos_arrays
                        )
                        node_mask[valid_indices] = True
                else:
                    # 1D case - use vectorized indexing
                    pos_ints = node_positions.astype(int)
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
                # Convert positions to integer arrays using vectorized operations
                pos_arrays = [
                    node_positions[:, i].astype(int)
                    for i in range(node_positions.shape[1])
                ]

                # Create validity mask for bounds checking using vectorized operations
                valid_mask = np.ones(len(node_positions), dtype=bool)
                for i, pos_arr in enumerate(pos_arrays):
                    valid_mask &= (pos_arr >= 0) & (pos_arr < node_mask.shape[i])

                # Apply advanced indexing with valid positions only
                if np.any(valid_mask):
                    valid_indices = tuple(pos_arr[valid_mask] for pos_arr in pos_arrays)
                    node_mask[valid_indices] = True
            else:
                # 1D case - use vectorized indexing
                pos_ints = node_positions.astype(int)
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

        # Check that all nodes have valid classifications using vectorized operations
        if len(node_map.node_classifications) > 0:
            # Extract depths and areas using vectorized operations
            depths_array = np.array([c.depth for c in node_map.node_classifications])
            areas_array = np.array([c.area for c in node_map.node_classifications])

            # Use CUDA-accelerated comparisons
            depths_cuda = CudaArray(depths_array, device="cpu")
            areas_cuda = CudaArray(areas_array, device="cpu")

            # Check for negative values using ElementWiseVectorizer
            elem_vec = ElementWiseVectorizer(use_gpu=True)
            reduction_vec = ReductionVectorizer(use_gpu=True)

            # Check depths < 0
            depths_lt_zero = elem_vec.vectorize_operation(depths_cuda, "less", 0.0)
            has_negative_depth = reduction_vec.vectorize_reduction(
                depths_lt_zero, "any"
            )

            # Check areas < 0
            areas_lt_zero = elem_vec.vectorize_operation(areas_cuda, "less", 0.0)
            has_negative_area = reduction_vec.vectorize_reduction(areas_lt_zero, "any")

            # Cleanup GPU memory
            if depths_lt_zero.device == "cuda":
                depths_lt_zero.swap_to_cpu()
            if areas_lt_zero.device == "cuda":
                areas_lt_zero.swap_to_cpu()
            if depths_cuda.device == "cuda":
                depths_cuda.swap_to_cpu()
            if areas_cuda.device == "cuda":
                areas_cuda.swap_to_cpu()

            # Find invalid nodes for error messages
            if has_negative_depth:
                invalid_depths = np.where(depths_array < 0)[0]
                node_ids = [
                    node_map.node_classifications[i].node_id for i in invalid_depths
                ]
                raise ValueError(
                    f"Nodes {node_ids} have negative depth: "
                    f"{depths_array[invalid_depths]}"
                )

            if has_negative_area:
                invalid_areas = np.where(areas_array < 0)[0]
                node_ids = [
                    node_map.node_classifications[i].node_id for i in invalid_areas
                ]
                raise ValueError(
                    f"Nodes {node_ids} have negative area: {areas_array[invalid_areas]}"
                )

        return True
