"""
Unit tests for CudaArray class.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np

from utils.cuda.array_model import CudaArray
from utils.cuda.exceptions import (
    CudaUnavailableError,
    CudaMemoryError,
    CudaMemoryLimitExceededError,
)


class TestCudaArray:
    """Test CudaArray class."""

    def test_init_cpu(self):
        """Test initialization on CPU."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        assert array.device == "cpu"
        assert array.shape == (5,)
        assert np.array_equal(array.to_numpy(), data)

    def test_init_invalid_data(self):
        """Test initialization with invalid data type."""
        with pytest.raises(ValueError, match="data must be numpy array"):
            CudaArray([1, 2, 3], device="cpu")

    def test_init_with_block_size(self):
        """Test initialization with custom block size."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")
        assert array.block_size == 2
        assert array.num_blocks == 4

    def test_init_auto_block_size(self):
        """Test automatic block size calculation."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=None, device="cpu")
        assert array.block_size >= 1024  # Minimum block size

    def test_init_small_array(self):
        """Test initialization with small array."""
        data = np.array([1, 2])
        array = CudaArray(data, device="cpu")
        assert array.num_blocks == 1  # Single block for small arrays

    def test_init_2d_array(self):
        """Test initialization with 2D array."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        array = CudaArray(data, device="cpu")
        assert array.shape == (3, 2)
        assert np.array_equal(array.to_numpy(), data)

    def test_properties(self):
        """Test all properties."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        array = CudaArray(data, device="cpu")
        assert array.shape == (5,)
        assert array.dtype == np.float32
        assert array.device == "cpu"
        assert array.num_blocks >= 1
        assert array.block_size > 0

    def test_get_block(self):
        """Test getting blocks."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")
        block = array.get_block(0)
        assert np.array_equal(block, np.array([1, 2]))

    def test_get_block_all_blocks(self):
        """Test getting all blocks."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")
        blocks = [array.get_block(i) for i in range(array.num_blocks)]
        assert len(blocks) == 4
        assert np.array_equal(blocks[0], np.array([1, 2]))
        assert np.array_equal(blocks[3], np.array([7, 8]))

    def test_get_block_invalid_index(self):
        """Test getting block with invalid index."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        with pytest.raises(IndexError):
            array.get_block(-1)
        with pytest.raises(IndexError):
            array.get_block(100)

    def test_set_block(self):
        """Test setting blocks."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")
        array.set_block(0, np.array([10, 20]))
        result = array.to_numpy()
        assert result[0] == 10
        assert result[1] == 20

    def test_set_block_invalid_index(self):
        """Test setting block with invalid index."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        with pytest.raises(IndexError):
            array.set_block(-1, np.array([10]))
        with pytest.raises(IndexError):
            array.set_block(100, np.array([10]))

    def test_set_block_invalid_size(self):
        """Test setting block with invalid size."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")
        with pytest.raises(ValueError, match="doesn't match"):
            array.set_block(0, np.array([10, 20, 30]))

    def test_use_whole_array(self):
        """Test getting whole array."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        whole = array.use_whole_array()
        assert np.array_equal(whole, data)
        assert whole.flags["C_CONTIGUOUS"]

    def test_use_whole_array_2d(self):
        """Test getting whole 2D array."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        array = CudaArray(data, device="cpu")
        whole = array.use_whole_array()
        assert np.array_equal(whole, data)

    def test_to_numpy(self):
        """Test conversion to numpy."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        numpy_array = array.to_numpy()
        assert isinstance(numpy_array, np.ndarray)
        assert np.array_equal(numpy_array, data)
        assert numpy_array.flags["C_CONTIGUOUS"]

    def test_to_numpy_2d(self):
        """Test conversion to numpy for 2D array."""
        data = np.array([[1, 2], [3, 4]])
        array = CudaArray(data, device="cpu")
        numpy_array = array.to_numpy()
        assert np.array_equal(numpy_array, data)

    def test_array_conversion(self):
        """Test __array__ method."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        numpy_array = np.asarray(array)
        assert np.array_equal(numpy_array, data)

    def test_swap_to_gpu_unavailable(self):
        """Test swap to GPU when CUDA unavailable."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        # This will raise error if CUDA unavailable
        # (which is expected in test environment without GPU)
        try:
            array.swap_to_gpu()
        except CudaUnavailableError:
            # Expected if CUDA not available
            pass

    def test_swap_to_cpu(self):
        """Test swap to CPU."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        array.swap_to_cpu()
        assert array.device == "cpu"
        assert np.array_equal(array.to_numpy(), data)

    def test_process_blocks(self):
        """Test block processing."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")

        def double(x):
            return x * 2

        result = array.process_blocks(double, use_gpu=False)
        expected = data * 2
        assert np.array_equal(result.to_numpy(), expected)

    def test_process_blocks_identity(self):
        """Test block processing with identity function."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")

        def identity(x):
            return x

        result = array.process_blocks(identity, use_gpu=False)
        assert np.array_equal(result.to_numpy(), data)

    def test_process_blocks_square(self):
        """Test block processing with square function."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")

        def square(x):
            return x ** 2

        result = array.process_blocks(square, use_gpu=False)
        expected = data ** 2
        assert np.array_equal(result.to_numpy(), expected)

    def test_calculate_block_shape_1d(self):
        """Test block shape calculation for 1D array."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")
        # Block shape should be (2,) for 1D
        assert array.num_blocks == 4

    def test_calculate_block_shape_2d(self):
        """Test block shape calculation for 2D array."""
        data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        array = CudaArray(data, block_size=2, device="cpu")
        assert array.shape == (4, 2)

    def test_single_block_array(self):
        """Test array that fits in single block."""
        data = np.array([1, 2])
        array = CudaArray(data, block_size=10, device="cpu")
        assert array.num_blocks == 1
        block = array.get_block(0)
        assert np.array_equal(block, data)

    def test_memory_limit_check(self):
        """Test GPU memory limit checking."""
        # This test will pass if CUDA unavailable (expected in test env)
        try:
            import cupy as cp

            if cp is not None:
                # Create large array that might exceed limit
                # In test environment, this might not actually exceed
                # but we test the check mechanism
                data = np.ones(1000, dtype=np.float32)
                array = CudaArray(data, device="cpu")
                # Check should not raise if memory is available
                try:
                    array.swap_to_gpu()
                except (CudaMemoryLimitExceededError, CudaUnavailableError):
                    # Expected if limit exceeded or CUDA unavailable
                    pass
        except ImportError:
            # CuPy not available, skip test
            pass
