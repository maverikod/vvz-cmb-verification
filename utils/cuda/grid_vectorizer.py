"""
Grid vectorizer for CUDA acceleration.

Provides vectorized grid-based operations (local minima, gradients, etc.).

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.cuda.array_model import CudaArray

import numpy as np

from utils.cuda.vectorizer import BaseVectorizer

# Try to import CuPy
try:
    import cupy as cp
    from cupyx.scipy.ndimage import (
        minimum_filter,
        maximum_filter,
        gaussian_filter,
    )

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    minimum_filter = None
    maximum_filter = None
    gaussian_filter = None


class GridVectorizer(BaseVectorizer):
    """
    Vectorizer for grid-based operations.

    Supports:
    - Local minimum/maximum detection
    - Gradient calculations
    - Laplacian calculations
    - Curvature calculations
    - Neighborhood operations
    """

    def __init__(
        self,
        use_gpu: bool = True,
        block_size: Optional[int] = None,
        whole_array: bool = False,
    ):
        """
        Initialize grid vectorizer.

        Args:
            use_gpu: Use GPU acceleration if available
            block_size: Block size (None = auto)
            whole_array: Use whole array mode (default: False)
        """
        super().__init__(
            use_gpu=use_gpu, block_size=block_size, whole_array=whole_array
        )

    def vectorize_grid_operation(
        self,
        array: "CudaArray",  # noqa: F821
        operation: str,
        *args: Any,
        **kwargs: Any,
    ) -> "CudaArray":  # noqa: F821
        """
        Apply grid-based operation.

        Args:
            array: Input grid array
            operation: Operation name ("local_minima", "gradient", "curvature", etc.)
            *args: Operation-specific arguments
            **kwargs: Operation-specific keyword arguments

        Returns:
            Result CudaArray

        Raises:
            ValueError: If operation not supported
        """
        # Store operation type for use in processing
        self._current_operation = operation
        self._operation_args = args
        self._operation_kwargs = kwargs

        return self.vectorize(array, *args, **kwargs)

    def _process_block(
        self, block: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process single block with grid operation.

        Args:
            block: Block data
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed block
        """
        operation = getattr(self, "_current_operation", None)
        if operation is None:
            raise ValueError(
                "Operation not set. Use vectorize_grid_operation() method."
            )

        # Apply operation
        if operation == "local_minima":
            neighborhood_size = kwargs.get("neighborhood_size", 3)
            return self._local_minima(block, neighborhood_size)
        elif operation == "local_maxima":
            neighborhood_size = kwargs.get("neighborhood_size", 3)
            return self._local_maxima(block, neighborhood_size)
        elif operation == "gradient":
            return self._gradient(block)
        elif operation == "laplacian":
            return self._laplacian(block)
        elif operation == "curvature":
            return self._curvature(block)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _process_whole_array(
        self, array: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process whole array with grid operation.

        Args:
            array: Whole array data
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed array
        """
        # For grid operations, whole array is same as block processing
        return self._process_block(array, *args, **kwargs)

    def _local_minima(self, grid: np.ndarray, neighborhood_size: int = 3) -> np.ndarray:
        """
        Find local minima in grid.

        Args:
            grid: Input grid
            neighborhood_size: Size of neighborhood (default: 3)

        Returns:
            Boolean array indicating local minima
        """
        if self.use_gpu and CUPY_AVAILABLE and minimum_filter is not None:
            if not isinstance(grid, cp.ndarray):
                grid = cp.asarray(grid)
            # Find minimum in neighborhood
            neighborhood_min = minimum_filter(grid, size=neighborhood_size)
            # Points where grid equals neighborhood minimum are local minima
            is_minimum = grid == neighborhood_min
            return is_minimum
        else:
            # CPU implementation using scipy
            from scipy.ndimage import minimum_filter as scipy_min_filter

            neighborhood_min = scipy_min_filter(grid, size=neighborhood_size)
            is_minimum = grid == neighborhood_min
            return is_minimum

    def _local_maxima(self, grid: np.ndarray, neighborhood_size: int = 3) -> np.ndarray:
        """
        Find local maxima in grid.

        Args:
            grid: Input grid
            neighborhood_size: Size of neighborhood (default: 3)

        Returns:
            Boolean array indicating local maxima
        """
        if self.use_gpu and CUPY_AVAILABLE and maximum_filter is not None:
            if not isinstance(grid, cp.ndarray):
                grid = cp.asarray(grid)
            # Find maximum in neighborhood
            neighborhood_max = maximum_filter(grid, size=neighborhood_size)
            # Points where grid equals neighborhood maximum are local maxima
            is_maximum = grid == neighborhood_max
            return is_maximum
        else:
            # CPU implementation using scipy
            from scipy.ndimage import maximum_filter as scipy_max_filter

            neighborhood_max = scipy_max_filter(grid, size=neighborhood_size)
            is_maximum = grid == neighborhood_max
            return is_maximum

    def _gradient(self, grid: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of grid.

        Args:
            grid: Input grid

        Returns:
            Gradient array (first component for 1D, tuple converted to array)
        """
        if self.use_gpu and CUPY_AVAILABLE:
            if not isinstance(grid, cp.ndarray):
                grid = cp.asarray(grid)
            # Use CuPy gradient
            gradients = cp.gradient(grid)
            # Return first component if tuple
            if isinstance(gradients, tuple):
                return gradients[0]
            return gradients
        else:
            # Use NumPy gradient
            gradients = np.gradient(grid)
            # Return first component if tuple
            if isinstance(gradients, tuple):
                return gradients[0]
            return gradients

    def _laplacian(self, grid: np.ndarray) -> np.ndarray:
        """
        Calculate Laplacian of grid.

        Args:
            grid: Input grid

        Returns:
            Laplacian array
        """
        # Laplacian = sum of second derivatives
        gradients = self._gradient(grid)
        laplacian = np.zeros_like(grid)

        if self.use_gpu and CUPY_AVAILABLE:
            for grad in gradients:
                grad_grad = cp.gradient(grad)
                if isinstance(grad_grad, tuple):
                    laplacian += grad_grad[0]
                else:
                    laplacian += grad_grad
        else:
            for grad in gradients:
                grad_grad = np.gradient(grad)
                if isinstance(grad_grad, tuple):
                    laplacian += grad_grad[0]
                else:
                    laplacian += grad_grad

        return laplacian

    def _curvature(self, grid: np.ndarray) -> np.ndarray:
        """
        Calculate curvature of grid.

        Args:
            grid: Input grid

        Returns:
            Curvature array
        """
        # Curvature calculation (simplified)
        # For 2D: curvature = |gradient|^2 / (1 + |gradient|^2)^(3/2)
        gradients = self._gradient(grid)

        if self.use_gpu and CUPY_AVAILABLE:
            grad_mag_sq = sum(g * g for g in gradients)
            curvature = grad_mag_sq / cp.power(1 + grad_mag_sq, 1.5)
        else:
            grad_mag_sq = sum(g * g for g in gradients)
            curvature = grad_mag_sq / np.power(1 + grad_mag_sq, 1.5)

        return curvature
