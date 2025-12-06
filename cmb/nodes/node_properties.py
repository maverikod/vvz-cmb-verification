"""
Θ-node properties calculator.

Calculates node depth, area, and curvature using CUDA-accelerated operations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Tuple, Optional
import numpy as np

from utils.cuda import (
    CudaArray,
    GridVectorizer,
    ReductionVectorizer,
)


class NodePropertiesCalculator:
    """
    Calculator for node properties (depth, area, curvature).

    Uses CUDA-accelerated operations for efficient computation.
    """

    def __init__(
        self,
        omega_field_cuda: CudaArray,
        grid_vec: GridVectorizer,
        reduction_vec: ReductionVectorizer,
        grid_spacing: Optional[float] = None,
    ):
        """
        Initialize properties calculator.

        Args:
            omega_field_cuda: CUDA array of omega field
            grid_vec: Grid vectorizer for grid operations
            reduction_vec: Reduction vectorizer for reductions
            grid_spacing: Grid spacing for area conversion
        """
        self.omega_field_cuda = omega_field_cuda
        self.grid_vec = grid_vec
        self.reduction_vec = reduction_vec
        self.grid_spacing = grid_spacing

    def calculate_node_depth(self, node_position: Tuple[int, ...]) -> float:
        """
        Calculate node depth (Δω/ω).

        Formula: Δω = ω - ω_min
        where ω_min is local minimum at node position.

        Args:
            node_position: Node position (grid indices)

        Returns:
            Node depth (Δω/ω)

        Raises:
            ValueError: If depth calculation fails
        """
        try:
            # Swap to GPU
            self.omega_field_cuda.swap_to_gpu()

            # Get omega value at node position
            omega_np = self.omega_field_cuda.to_numpy()
            omega_at_node = float(omega_np[node_position])

            # Find local minimum in neighborhood
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

        Raises:
            ValueError: If area calculation fails
        """
        try:
            # Swap to GPU
            self.omega_field_cuda.swap_to_gpu()

            # Find connected region around node
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

            # Count connected pixels (area) using CUDA-accelerated reduction
            node_mask_sum_cuda = CudaArray(
                node_mask_cuda.to_numpy().astype(np.float32), device="cpu"
            )
            node_mask_sum_cuda.swap_to_gpu()
            area_sum = self.reduction_vec.vectorize_reduction(node_mask_sum_cuda, "sum")
            if isinstance(area_sum, CudaArray):
                area = float(area_sum.to_numpy().item())
            else:
                area = float(area_sum)
            # Cleanup
            if node_mask_sum_cuda.device == "cuda":
                node_mask_sum_cuda.swap_to_cpu()

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

        Raises:
            ValueError: If curvature calculation fails
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

    def calculate_node_properties_batch(
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

            # Initialize result arrays using CudaArray
            depths_cuda = CudaArray(np.zeros(n_nodes, dtype=np.float64), device="cpu")
            areas_cuda = CudaArray(np.zeros(n_nodes, dtype=np.float64), device="cpu")
            curvatures_cuda = CudaArray(
                np.zeros(n_nodes, dtype=np.float64), device="cpu"
            )
            depths = depths_cuda.to_numpy()
            areas = areas_cuda.to_numpy()
            curvatures = curvatures_cuda.to_numpy()

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
                    pos_array = np.asarray(pos_tuple, dtype=int)
                    shape_array = np.array(omega_np.shape, dtype=int)
                    starts = np.maximum(0, pos_array - half_size)
                    ends = np.minimum(shape_array, pos_array + half_size + 1)
                    # Convert to slices using vectorized operations
                    slices = tuple(
                        slice(int(starts[i]), int(ends[i])) for i in range(len(starts))
                    )

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

                    # Calculate depth: for scalar operations, use numpy directly
                    # (CUDA overhead for scalars is not beneficial)
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
                    # Apply grid spacing: for scalar operations, use numpy directly
                    # (CUDA overhead for scalars is not beneficial)
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
