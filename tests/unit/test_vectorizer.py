"""
Unit tests for BaseVectorizer class.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np

from utils.cuda.array_model import CudaArray
from utils.cuda.vectorizer import BaseVectorizer
from utils.cuda.elementwise_vectorizer import ElementWiseVectorizer


class TestBaseVectorizer:
    """Test BaseVectorizer class."""

    def test_init(self):
        """Test initialization."""
        # Can't instantiate abstract class directly
        # Test through concrete implementation
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        assert not vectorizer.use_gpu
        assert not vectorizer.whole_array

    def test_init_with_options(self):
        """Test initialization with options."""
        vectorizer = ElementWiseVectorizer(
            use_gpu=True, block_size=1000, whole_array=True
        )
        assert vectorizer.whole_array

    def test_use_gpu_property(self):
        """Test use_gpu property."""
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        assert not vectorizer.use_gpu

    def test_whole_array_property(self):
        """Test whole_array property."""
        vectorizer = ElementWiseVectorizer(whole_array=True)
        assert vectorizer.whole_array

    def test_vectorize_whole_array_mode(self):
        """Test vectorize in whole array mode."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False, whole_array=True)
        result = vectorizer.multiply(array, 2)
        expected = data * 2
        assert np.allclose(result.to_numpy(), expected)

    def test_vectorize_block_mode(self):
        """Test vectorize in block mode."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        array = CudaArray(data, block_size=2, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False, whole_array=False)
        result = vectorizer.multiply(array, 2)
        expected = data * 2
        assert np.allclose(result.to_numpy(), expected)

    def test_batch_processing(self):
        """Test batch processing."""
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])
        array1 = CudaArray(data1, device="cpu")
        array2 = CudaArray(data2, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        # Use vectorize_operation for each
        results = [
            vectorizer.multiply(array1, 2),
            vectorizer.multiply(array2, 2),
        ]
        assert len(results) == 2
        assert np.allclose(results[0].to_numpy(), data1 * 2)
        assert np.allclose(results[1].to_numpy(), data2 * 2)

    def test_gpu_fallback(self):
        """Test GPU fallback to CPU."""
        vectorizer = ElementWiseVectorizer(use_gpu=True)
        # Should fallback to CPU if GPU unavailable
        data = np.array([1, 2, 3])
        array = CudaArray(data, device="cpu")
        result = vectorizer.multiply(array, 2)
        expected = data * 2
        assert np.allclose(result.to_numpy(), expected)
