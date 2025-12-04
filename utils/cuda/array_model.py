"""
Block-based array model for CUDA acceleration.

Provides efficient block-based array storage with CPU-GPU memory management,
swap capabilities, and whole-array mode for FFT operations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Tuple, Callable, Union
import numpy as np

# Try to import CuPy, but make it optional
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class CudaUnavailableError(Exception):
    """Raised when CUDA is requested but unavailable."""

    pass


class CudaMemoryError(Exception):
    """Raised when CUDA memory operations fail."""

    pass


class CudaArray:
    """
    Block-based array model for CUDA acceleration.

    Supports:
    - Block-based processing for memory efficiency
    - CPU-GPU memory swap
    - Whole-array mode (for FFT operations)
    - Automatic block size calculation

    Attributes:
        _data: Original data on CPU (numpy array)
        _gpu_data: GPU data (CuPy array, None if not on GPU)
        _block_size: Size of blocks for processing (None = auto)
        _device: Current device ("cpu" or "cuda")
        _shape: Array shape
        _dtype: Data type
        _num_blocks: Number of blocks
        _block_shape: Shape of each block
    """

    def __init__(
        self,
        data: np.ndarray,
        block_size: Optional[int] = None,
        device: str = "cpu",
    ):
        """
        Initialize array model.

        Args:
            data: Input numpy array
            block_size: Size of blocks for processing (None = auto-detect)
            device: Initial device ("cpu" or "cuda")

        Raises:
            ValueError: If data is not a numpy array
            CudaUnavailableError: If device="cuda" but CUDA unavailable
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(f"data must be numpy array, got {type(data)}")

        self._data = np.ascontiguousarray(data)
        self._gpu_data: Optional[Union[cp.ndarray, np.ndarray]] = None
        self._device = device
        self._shape = self._data.shape
        self._dtype = self._data.dtype

        # Calculate block size if not provided
        if block_size is None:
            self._block_size = self._calculate_block_size()
        else:
            self._block_size = block_size

        # Calculate number of blocks and block shape
        self._num_blocks = self._calculate_num_blocks()
        self._block_shape = self._calculate_block_shape()

        # Move to GPU if requested
        if device == "cuda":
            self.swap_to_gpu()

    def _calculate_block_size(self) -> int:
        """
        Calculate optimal block size based on available memory.

        Returns:
            Optimal block size in elements
        """
        if self._device == "cuda" and CUPY_AVAILABLE:
            try:
                # Get available GPU memory
                mempool = cp.get_default_memory_pool()
                free_mem = mempool.free_bytes()

                # Reserve 50% for operations (need input + output)
                available = free_mem * 0.5

                # Calculate block size (bytes per element)
                bytes_per_element = self._data.itemsize

                # Maximum elements per block
                max_elements = int(
                    available // (bytes_per_element * 2)
                )  # *2 for input+output

                # Use at least 4 blocks, at most available memory
                block_size = min(max_elements, self._data.size // 4)
                return max(block_size, 1024)  # Minimum 1024 elements
            except Exception:
                # Fallback to CPU block size if GPU calculation fails
                return self._data.size // 4
        else:
            # CPU: use 4 blocks
            return max(self._data.size // 4, 1024)

    def _calculate_num_blocks(self) -> int:
        """
        Calculate number of blocks needed.

        Returns:
            Number of blocks
        """
        if self._block_size >= self._data.size:
            return 1
        return (self._data.size + self._block_size - 1) // self._block_size

    def _calculate_block_shape(self) -> Tuple[int, ...]:
        """
        Calculate shape of each block.

        Returns:
            Block shape tuple
        """
        if self._num_blocks == 1:
            return self._shape

        # For 1D arrays, block shape is just (block_size,)
        if len(self._shape) == 1:
            return (self._block_size,)

        # For multi-dimensional arrays, calculate block shape
        # This is simplified - assumes first dimension is split
        block_first_dim = min(self._block_size, self._shape[0])
        return (block_first_dim,) + self._shape[1:]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get array shape."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Get data type."""
        return self._dtype

    @property
    def device(self) -> str:
        """Get current device."""
        return self._device

    @property
    def num_blocks(self) -> int:
        """Get number of blocks."""
        return self._num_blocks

    @property
    def block_size(self) -> int:
        """Get block size."""
        return self._block_size

    def swap_to_gpu(self) -> None:
        """
        Move array to GPU memory.

        Raises:
            CudaUnavailableError: If GPU unavailable
            CudaMemoryError: If out of GPU memory
        """
        if not CUPY_AVAILABLE:
            raise CudaUnavailableError(
                "CuPy not available. Install with: pip install cupy"
            )

        try:
            if self._gpu_data is None:
                self._gpu_data = cp.asarray(self._data)
            self._device = "cuda"
        except cp.cuda.memory.OutOfMemoryError as e:
            raise CudaMemoryError(f"Out of GPU memory: {e}") from e

    def swap_to_cpu(self) -> None:
        """
        Move array to CPU memory.

        Copies GPU data to CPU if on GPU.
        """
        if self._device == "cuda" and self._gpu_data is not None:
            self._data = cp.asnumpy(self._gpu_data)
            # Optionally free GPU memory
            if CUPY_AVAILABLE:
                del self._gpu_data
                cp.get_default_memory_pool().free_all_blocks()
            self._gpu_data = None
        self._device = "cpu"

    def get_block(self, block_idx: int) -> np.ndarray:
        """
        Get specific block for processing.

        Args:
            block_idx: Block index (0 to num_blocks-1)

        Returns:
            Block data (on current device)

        Raises:
            IndexError: If block_idx out of range
        """
        if block_idx < 0 or block_idx >= self._num_blocks:
            raise IndexError(
                f"block_idx {block_idx} out of range [0, {self._num_blocks})"
            )

        # Calculate block slice
        start_idx = block_idx * self._block_size
        end_idx = min(start_idx + self._block_size, self._data.size)

        # Get data from current device
        if self._device == "cuda" and self._gpu_data is not None:
            # Flatten for indexing, then reshape
            flat_gpu = self._gpu_data.flatten()
            block_flat = flat_gpu[start_idx:end_idx]
            # Reshape to original shape if possible
            if len(self._shape) == 1:
                return block_flat
            # For multi-dim, need to calculate proper shape
            block_size_actual = end_idx - start_idx
            return block_flat[:block_size_actual].reshape(-1, *self._shape[1:])
        else:
            # CPU: get from numpy array
            flat_data = self._data.flatten()
            block_flat = flat_data[start_idx:end_idx]
            if len(self._shape) == 1:
                return block_flat
            block_size_actual = end_idx - start_idx
            return block_flat[:block_size_actual].reshape(-1, *self._shape[1:])

    def set_block(self, block_idx: int, block_data: np.ndarray) -> None:
        """
        Set block data after processing.

        Args:
            block_idx: Block index
            block_data: Processed block data

        Raises:
            IndexError: If block_idx out of range
            ValueError: If block_data shape doesn't match expected
        """
        if block_idx < 0 or block_idx >= self._num_blocks:
            raise IndexError(
                f"block_idx {block_idx} out of range [0, {self._num_blocks})"
            )

        # Calculate block slice
        start_idx = block_idx * self._block_size
        end_idx = min(start_idx + self._block_size, self._data.size)
        block_size_actual = end_idx - start_idx

        # Validate block size
        if block_data.size != block_size_actual:
            raise ValueError(
                f"block_data size {block_data.size} doesn't match "
                f"expected {block_size_actual}"
            )

        # Set data on current device
        if self._device == "cuda" and self._gpu_data is not None:
            # Convert to CuPy if needed
            if not isinstance(block_data, cp.ndarray):
                block_data = cp.asarray(block_data)
            # Flatten and set (need to copy to ensure modification)
            flat_gpu = self._gpu_data.flatten()
            flat_gpu[start_idx:end_idx] = block_data.flatten()[:block_size_actual].copy()
            # Update _gpu_data reference
            self._gpu_data = flat_gpu.reshape(self._shape)
        else:
            # CPU: set in numpy array
            flat_data = self._data.flatten()
            flat_data[start_idx:end_idx] = block_data.flatten()[:block_size_actual].copy()
            # Update _data reference
            self._data = flat_data.reshape(self._shape)

    def process_blocks(
        self, operation: Callable[[np.ndarray], np.ndarray], use_gpu: bool = True
    ) -> "CudaArray":
        """
        Process array in blocks using operation.

        Args:
            operation: Operation function (takes block, returns processed block)
            use_gpu: Use GPU if available

        Returns:
            New CudaArray with processed data
        """
        # Create output array with same shape
        output_data = np.zeros_like(self._data)
        output_array = CudaArray(output_data, block_size=self._block_size, device="cpu")

        # Process on GPU if requested and available
        if use_gpu and CUPY_AVAILABLE and self._device == "cuda":
            output_array.swap_to_gpu()

        # Process each block
        for block_idx in range(self._num_blocks):
            block = self.get_block(block_idx)
            processed_block = operation(block)
            output_array.set_block(block_idx, processed_block)

        return output_array

    def use_whole_array(self) -> np.ndarray:
        """
        Get whole array as contiguous block (for FFT).

        Returns:
            Whole array (on current device)

        Note:
            This bypasses block processing - use only for operations
            requiring whole array (e.g., FFT).
        """
        if self._device == "cuda" and self._gpu_data is not None:
            return self._gpu_data
        else:
            # Ensure contiguous on CPU
            return np.ascontiguousarray(self._data)

    def to_numpy(self) -> np.ndarray:
        """
        Convert to numpy array (always on CPU).

        Returns:
            Numpy array on CPU
        """
        if self._device == "cuda" and self._gpu_data is not None:
            return cp.asnumpy(self._gpu_data)
        return np.ascontiguousarray(self._data)

    def __array__(self) -> np.ndarray:
        """Allow conversion to numpy array."""
        return self.to_numpy()
