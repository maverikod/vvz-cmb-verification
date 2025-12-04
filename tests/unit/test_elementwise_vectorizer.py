"""
Unit tests for ElementWiseVectorizer class.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np

from utils.cuda.array_model import CudaArray
from utils.cuda.elementwise_vectorizer import ElementWiseVectorizer


class TestElementWiseVectorizer:
    """Test ElementWiseVectorizer class."""

    def test_multiply(self):
        """Test multiplication operation."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.multiply(array, 2)
        expected = data * 2
        assert np.allclose(result.to_numpy(), expected)

    def test_add(self):
        """Test addition operation."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.add(array, 10)
        expected = data + 10
        assert np.allclose(result.to_numpy(), expected)

    def test_subtract(self):
        """Test subtraction operation."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.subtract(array, 1)
        expected = data - 1
        assert np.allclose(result.to_numpy(), expected)

    def test_power(self):
        """Test power operation."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.power(array, 2)
        expected = data ** 2
        assert np.allclose(result.to_numpy(), expected)

    def test_sqrt(self):
        """Test square root operation."""
        data = np.array([1, 4, 9, 16, 25])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.sqrt(array)
        expected = np.sqrt(data)
        assert np.allclose(result.to_numpy(), expected)

    def test_sin(self):
        """Test sine operation."""
        data = np.array([0, np.pi / 2, np.pi])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.sin(array)
        expected = np.sin(data)
        assert np.allclose(result.to_numpy(), expected)
