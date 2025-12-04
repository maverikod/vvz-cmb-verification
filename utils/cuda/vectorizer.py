"""
Base vectorizer class for CUDA acceleration.

Provides abstract base class for vectorizing operations on arrays.
Subclasses implement specific operation types.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.cuda.array_model import CudaArray

import numpy as np

from utils.cuda.array_model import CudaUnavailableError

# Try to import CuPy
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class BaseVectorizer(ABC):
    """
    Base class for vectorizing operations on arrays.

    Subclasses implement specific operation types:
    - ElementWiseVectorizer: Element-wise operations (+, -, *, /, etc.)
    - TransformVectorizer: Transform operations (FFT, spherical harmonics)
    - ReductionVectorizer: Reduction operations (sum, mean, max, etc.)
    - CorrelationVectorizer: Correlation operations
    - GridVectorizer: Grid operations (minima, gradients)

    Attributes:
        _use_gpu: Use GPU acceleration if available
        _block_size: Block size for processing (None = auto)
        _whole_array: Use whole array mode (no blocking)
        _cuda_available: CUDA availability flag
    """

    def __init__(
        self,
        use_gpu: bool = True,
        block_size: Optional[int] = None,
        whole_array: bool = False,
    ):
        """
        Initialize vectorizer.

        Args:
            use_gpu: Use GPU acceleration if available
            block_size: Block size (None = auto-detect from array)
            whole_array: Use whole array mode (no blocking)
        """
        self._use_gpu = use_gpu
        self._block_size = block_size
        self._whole_array = whole_array
        self._cuda_available = CUPY_AVAILABLE and (cp is not None)

        # If whole_array mode, disable blocking
        if self._whole_array:
            self._block_size = None

    @property
    def use_gpu(self) -> bool:
        """Check if GPU is being used."""
        return self._use_gpu and self._cuda_available

    @property
    def whole_array(self) -> bool:
        """Check if whole array mode is enabled."""
        return self._whole_array

    def vectorize(self, array: "CudaArray", *args: Any, **kwargs: Any) -> "CudaArray":  # noqa: F821
        """
        Vectorize operation on array.

        Args:
            array: Input CudaArray
            *args: Operation-specific arguments
            **kwargs: Operation-specific keyword arguments

        Returns:
            Result CudaArray

        Raises:
            CudaUnavailableError: If GPU requested but unavailable
        """
        # Check GPU availability
        if self._use_gpu and not self._cuda_available:
            if self._use_gpu:
                # Fallback to CPU with warning
                import warnings

                warnings.warn(
                    "GPU requested but unavailable. Falling back to CPU.",
                    UserWarning,
                )
            self._use_gpu = False

        # Use whole array mode if enabled
        if self._whole_array:
            whole_array_data = array.use_whole_array()
            result_data = self._process_whole_array(whole_array_data, *args, **kwargs)
            # Create new CudaArray with result
            from utils.cuda.array_model import CudaArray

            result_numpy = self._to_numpy(result_data)
            return CudaArray(result_numpy, block_size=None, device=array.device)
        else:
            # Use block processing
            return self._process_blocks(array, *args, **kwargs)

    def batch(
        self, arrays: List["CudaArray"], *args: Any, **kwargs: Any  # noqa: F821
    ) -> List["CudaArray"]:  # noqa: F821
        """
        Batch process multiple arrays.

        Args:
            arrays: List of input CudaArrays
            *args: Operation-specific arguments
            **kwargs: Operation-specific keyword arguments

        Returns:
            List of result CudaArrays
        """
        results = []
        for array in arrays:
            result = self.vectorize(array, *args, **kwargs)
            results.append(result)
        return results

    def _process_blocks(
        self, array: "CudaArray", *args: Any, **kwargs: Any  # noqa: F821
    ) -> "CudaArray":  # noqa: F821
        """
        Process array in blocks.

        Args:
            array: Input CudaArray
            *args: Operation-specific arguments
            **kwargs: Operation-specific keyword arguments

        Returns:
            Processed CudaArray
        """
        # Determine block size
        block_size = self._block_size
        if block_size is None:
            block_size = array.block_size

        # Create output array
        from utils.cuda.array_model import CudaArray

        output_data = np.zeros_like(array.to_numpy())
        output_array = CudaArray(output_data, block_size=block_size, device="cpu")

        # Move to GPU if using GPU
        if self.use_gpu:
            try:
                array.swap_to_gpu()
                output_array.swap_to_gpu()
            except CudaUnavailableError:
                # Fallback to CPU
                pass

        # Process each block
        for block_idx in range(array.num_blocks):
            block = array.get_block(block_idx)
            processed_block = self._process_block(block, *args, **kwargs)
            output_array.set_block(block_idx, processed_block)

        return output_array

    @abstractmethod
    def _process_block(
        self, block: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process single block (to be implemented by subclasses).

        Args:
            block: Block data
            *args: Operation-specific arguments
            **kwargs: Operation-specific keyword arguments

        Returns:
            Processed block
        """
        raise NotImplementedError("Subclasses must implement _process_block")

    @abstractmethod
    def _process_whole_array(
        self, array: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process whole array (for FFT-like operations).

        Args:
            array: Whole array data
            *args: Operation-specific arguments
            **kwargs: Operation-specific keyword arguments

        Returns:
            Processed array
        """
        raise NotImplementedError("Subclasses must implement _process_whole_array")

    def _to_numpy(self, data: Any) -> np.ndarray:
        """
        Convert data to numpy array (from CuPy if needed).

        Args:
            data: Input data (numpy or CuPy array)

        Returns:
            Numpy array
        """
        if self._cuda_available and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return np.asarray(data)
