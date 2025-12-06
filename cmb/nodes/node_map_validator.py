"""
Θ-node map validator.

Validates generated node maps for consistency and correctness.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np

from utils.cuda import CudaArray, ElementWiseVectorizer, ReductionVectorizer

from cmb.nodes.node_data import ThetaNodeMap


class NodeMapValidator:
    """
    Validator for node maps.

    Validates node maps for shape consistency, classification correctness,
    and data validity using CUDA-accelerated operations.
    """

    def __init__(self):
        """Initialize validator."""
        pass

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

        # Check node mask values (should be 0 or 1) using CUDA-accelerated operations
        mask_cuda = CudaArray(node_map.node_mask.astype(np.float32), device="cpu")
        mask_cuda.swap_to_gpu()

        elem_vec = ElementWiseVectorizer(use_gpu=True)
        reduction_vec = ReductionVectorizer(use_gpu=True)

        # Check that all values are either 0 or 1
        # We check: all values are <= 1 and >= 0, and all values are integers
        # For binary mask: check that (mask == 0) OR (mask == 1) for all elements
        # Since CUDA doesn't have direct OR, we check: NOT((mask != 0) AND (mask != 1))
        # Which is: NOT((mask > 0) AND (mask < 1)) for values between 0 and 1
        # But simpler: check that mask <= 1 and mask >= 0, and mask is integer
        # For binary: check mask * (1 - mask) == 0 for all (only true for 0 and 1)
        ones_cuda = CudaArray(
            np.ones_like(node_map.node_mask, dtype=np.float32), device="cpu"
        )
        ones_cuda.swap_to_gpu()
        mask_complement = elem_vec.subtract(ones_cuda, mask_cuda)
        mask_product = elem_vec.multiply(mask_cuda, mask_complement)
        # For binary mask: mask * (1 - mask) should be 0 for all elements
        # (only true when mask is 0 or 1)
        mask_product_ne_zero = elem_vec.vectorize_operation(
            mask_product, "not_equal", 0.0
        )
        has_invalid = reduction_vec.vectorize_reduction(mask_product_ne_zero, "any")

        # Cleanup GPU memory
        if mask_cuda.device == "cuda":
            mask_cuda.swap_to_cpu()
        if ones_cuda.device == "cuda":
            ones_cuda.swap_to_cpu()
        if mask_complement.device == "cuda":
            mask_complement.swap_to_cpu()
        if mask_product.device == "cuda":
            mask_product.swap_to_cpu()
        if mask_product_ne_zero.device == "cuda":
            mask_product_ne_zero.swap_to_cpu()

        if has_invalid:
            raise ValueError("Node mask contains values other than 0 and 1")

        # Check that all nodes have valid classifications using vectorized ops
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
                    f"Nodes {node_ids} have negative area: "
                    f"{areas_array[invalid_areas]}"
                )

        return True
