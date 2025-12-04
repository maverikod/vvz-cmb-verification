"""
Unit tests for CudaArray class.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np

from utils.cuda.array_model import CudaArray, CudaUnavailableError


class TestCudaArray:
    """Test CudaArray class."""

    def test_init_cpu(self):
        """Test initialization on CPU."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        assert array.device == "cpu"
        assert array.shape == (5,)
        assert np.array_equal(array.to_numpy(), data)

    def test_init_with_block_size(self):
        """Test initialization with custom block size."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")
        assert array.block_size == 2
        assert array.num_blocks == 4

    def test_get_block(self):
        """Test getting blocks."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")
        block = array.get_block(0)
        assert np.array_equal(block, np.array([1, 2]))

    def test_set_block(self):
        """Test setting blocks."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")
        array.set_block(0, np.array([10, 20]))
        result = array.to_numpy()
        assert result[0] == 10
        assert result[1] == 20

    def test_use_whole_array(self):
        """Test getting whole array."""
        data = np.array([1, 2, 3, 4, 5])
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

    def test_process_blocks(self):
        """Test block processing."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")

        def double(x):
            return x * 2

        result = array.process_blocks(double, use_gpu=False)
        expected = data * 2
        assert np.array_equal(result.to_numpy(), expected)
