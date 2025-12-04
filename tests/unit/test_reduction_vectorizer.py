"""
Unit tests for ReductionVectorizer class.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np

from utils.cuda.array_model import CudaArray
from utils.cuda.reduction_vectorizer import ReductionVectorizer


class TestReductionVectorizer:
    """Test ReductionVectorizer class."""

    def test_sum(self):
        """Test sum reduction."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        vectorizer = ReductionVectorizer(use_gpu=False)
        result = vectorizer.vectorize_reduction(array, "sum")
        expected = np.sum(data)
        assert result == expected

    def test_mean(self):
        """Test mean reduction."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        vectorizer = ReductionVectorizer(use_gpu=False)
        result = vectorizer.vectorize_reduction(array, "mean")
        expected = np.mean(data)
        assert np.allclose(result, expected)

    def test_max(self):
        """Test max reduction."""
        data = np.array([1, 5, 3, 2, 4])
        array = CudaArray(data, device="cpu")
        vectorizer = ReductionVectorizer(use_gpu=False)
        result = vectorizer.vectorize_reduction(array, "max")
        expected = np.max(data)
        assert result == expected

    def test_min(self):
        """Test min reduction."""
        data = np.array([1, 5, 3, 2, 4])
        array = CudaArray(data, device="cpu")
        vectorizer = ReductionVectorizer(use_gpu=False)
        result = vectorizer.vectorize_reduction(array, "min")
        expected = np.min(data)
        assert result == expected
