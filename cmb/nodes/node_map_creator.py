"""
Θ-node map creator.

Creates omega_min map and node mask using CUDA-accelerated operations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Tuple, TYPE_CHECKING
import numpy as np

from utils.cuda import CudaArray, ElementWiseVectorizer, ReductionVectorizer

if TYPE_CHECKING:
    from utils.cuda import GridVectorizer


class NodeMapCreator:
    """
    Creator for node maps (omega_min map and node mask).

    Uses CUDA-accelerated operations for efficient map creation.
    """

    def __init__(
        self,
        omega_field_cuda: CudaArray,
        grid_vec: "GridVectorizer",
        elem_vec: ElementWiseVectorizer,
        omega_field_shape: Tuple[int, ...],
    ):
        """
        Initialize map creator.

        Args:
            omega_field_cuda: CUDA array of omega field
            grid_vec: Grid vectorizer for grid operations
            elem_vec: Element-wise vectorizer for element operations
            omega_field_shape: Shape of omega field
        """
        self.omega_field_cuda = omega_field_cuda
        self.grid_vec = grid_vec
        self.elem_vec = elem_vec
        self.omega_field_shape = omega_field_shape

    def create_omega_min_map(self, node_positions: np.ndarray) -> np.ndarray:
        """
        Create map of ω_min(x) values.

        Args:
            node_positions: Array of node positions

        Returns:
            Map of ω_min(x) values at each grid point

        Raises:
            ValueError: If map creation fails
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
                # Create node mask using CudaArray
                node_mask_cuda_init = CudaArray(
                    np.zeros(self.omega_field_shape, dtype=bool), device="cpu"
                )
                node_mask = node_mask_cuda_init.to_numpy()

                if len(node_positions.shape) == 2:
                    # Multi-dimensional case - use advanced indexing
                    # Convert positions to integer arrays using vectorized ops
                    pos_arrays = [
                        node_positions[:, i].astype(int)
                        for i in range(node_positions.shape[1])
                    ]

                    # Create validity mask using fully vectorized operations
                    valid_mask = np.ones(len(node_positions), dtype=bool)
                    shape_tuple: Tuple[int, ...] = node_mask.shape

                    # Vectorize all bounds checks using ElementWiseVectorizer
                    # Convert shape to array for vectorized comparison
                    shape_array = np.array(shape_tuple)

                    # Process each dimension using vectorized operations
                    for i, pos_arr in enumerate(pos_arrays):
                        # Use numpy vectorized operations (already optimized)
                        valid_mask &= (pos_arr >= 0) & (pos_arr < shape_array[i])

                    # Use advanced indexing for vectorized mask setting
                    # Use CUDA for any() check
                    valid_mask_cuda = CudaArray(valid_mask.astype(float), device="cpu")
                    reduction_vec = ReductionVectorizer(use_gpu=True)
                    has_valid = reduction_vec.vectorize_reduction(
                        valid_mask_cuda, "any"
                    )
                    if has_valid:
                        valid_indices = tuple(
                            pos_arr[valid_mask] for pos_arr in pos_arrays
                        )
                        node_mask[valid_indices] = True
                    # Cleanup GPU memory
                    if valid_mask_cuda.device == "cuda":
                        valid_mask_cuda.swap_to_cpu()
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

        Raises:
            ValueError: If mask creation fails
        """
        try:
            # Create mask using CudaArray
            node_mask_cuda_init = CudaArray(
                np.zeros(self.omega_field_shape, dtype=bool), device="cpu"
            )
            node_mask = node_mask_cuda_init.to_numpy()

            if len(node_positions) == 0:
                # Convert to int32 using CudaArray for consistency
                mask_cuda = CudaArray(node_mask.astype(np.float32), device="cpu")
                return mask_cuda.to_numpy().astype(np.int32)

            # Set node positions to 1 using vectorized operations
            if len(node_positions.shape) == 2:
                # Multi-dimensional case - use advanced indexing
                # Convert positions to integer arrays using vectorized ops
                pos_arrays = [
                    node_positions[:, i].astype(int)
                    for i in range(node_positions.shape[1])
                ]

                # Create validity mask for bounds checking using vectorized ops
                # Use numpy vectorized operations for all dimensions
                valid_mask = np.ones(len(node_positions), dtype=bool)
                shape_array: np.ndarray = np.array(node_mask.shape)

                # Process each dimension using vectorized operations
                for i, pos_arr in enumerate(pos_arrays):
                    # Use numpy vectorized operations (already optimized)
                    valid_mask &= (pos_arr >= 0) & (pos_arr < shape_array[i])

                # Apply advanced indexing with valid positions only
                # Use CUDA for any() check
                valid_mask_cuda = CudaArray(valid_mask.astype(float), device="cpu")
                reduction_vec = ReductionVectorizer(use_gpu=True)
                has_valid = reduction_vec.vectorize_reduction(valid_mask_cuda, "any")
                if has_valid:
                    valid_indices = tuple(pos_arr[valid_mask] for pos_arr in pos_arrays)
                    node_mask[valid_indices] = True
                # Cleanup GPU memory
                if valid_mask_cuda.device == "cuda":
                    valid_mask_cuda.swap_to_cpu()
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
