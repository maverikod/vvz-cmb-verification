"""
Grid vectorizer for CUDA acceleration.

Provides vectorized grid-based operations (local minima, gradients, etc.).

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Any, TYPE_CHECKING

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
        # Simplified Laplacian calculation
        # For 2D: use second-order finite differences
        if len(grid.shape) == 1:
            # 1D: second derivative
            laplacian = np.zeros_like(grid)
            laplacian[1:-1] = grid[2:] - 2 * grid[1:-1] + grid[:-2]
        else:
            # 2D: sum of second derivatives in each direction
            laplacian = np.zeros_like(grid)
            # Second derivative in first dimension
            if grid.shape[0] > 2:
                laplacian[1:-1, :] += grid[2:, :] - 2 * grid[1:-1, :] + grid[:-2, :]
            # Second derivative in second dimension
            if grid.shape[1] > 2:
                laplacian[:, 1:-1] += grid[:, 2:] - 2 * grid[:, 1:-1] + grid[:, :-2]

        return laplacian

    def _curvature(self, grid: np.ndarray) -> np.ndarray:
        """
        Calculate curvature of grid.

        Args:
            grid: Input grid

        Returns:
            Curvature array
        """
        # Simplified curvature calculation
        # Use gradient magnitude squared
        if len(grid.shape) == 1:
            # 1D: use first derivative squared
            grad = np.gradient(grid)
            if isinstance(grad, tuple):
                grad = grad[0]
            grad_mag_sq = grad**2
        else:
            # 2D: use gradient magnitude
            grad_x = np.gradient(grid, axis=0)
            if isinstance(grad_x, tuple):
                grad_x = grad_x[0]
            grad_y = np.gradient(grid, axis=1)
            if isinstance(grad_y, tuple):
                grad_y = grad_y[0]
            grad_mag_sq = grad_x**2 + grad_y**2

        # Curvature = grad_mag_sq / (1 + grad_mag_sq)^(3/2)
        if self.use_gpu and CUPY_AVAILABLE:
            curvature = grad_mag_sq / cp.power(1 + grad_mag_sq, 1.5)
        else:
            curvature = grad_mag_sq / np.power(1 + grad_mag_sq, 1.5)

        return curvature
