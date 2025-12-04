"""
Unit tests for CorrelationVectorizer class.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np

from utils.cuda.array_model import CudaArray
from utils.cuda.correlation_vectorizer import CorrelationVectorizer


class TestCorrelationVectorizer:
    """Test CorrelationVectorizer class."""

    def test_init(self):
        """Test initialization."""
        vectorizer = CorrelationVectorizer(use_gpu=False)
        assert vectorizer.whole_array  # Always True for correlation

    def test_fft_correlation(self):
        """Test FFT-based correlation."""
        data1 = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        data2 = np.array([2, 3, 4, 5, 6], dtype=np.float64)
        array1 = CudaArray(data1, device="cpu")
        array2 = CudaArray(data2, device="cpu")
        vectorizer = CorrelationVectorizer(use_gpu=False)
        result = vectorizer.vectorize_correlation(array1, array2, method="fft")
        # Result should be correlation array
        assert result.shape == array1.shape

    def test_fft_correlation_different_sizes(self):
        """Test FFT correlation with different array sizes."""
        data1 = np.array([1, 2, 3], dtype=np.float64)
        data2 = np.array([2, 3, 4, 5, 6], dtype=np.float64)
        array1 = CudaArray(data1, device="cpu")
        array2 = CudaArray(data2, device="cpu")
        vectorizer = CorrelationVectorizer(use_gpu=False)
        result = vectorizer.vectorize_correlation(array1, array2, method="fft")
        # Should handle different sizes
        assert result is not None

    def test_direct_correlation(self):
        """Test direct correlation."""
        data1 = np.array([1, 2, 3], dtype=np.float64)
        data2 = np.array([2, 3, 4], dtype=np.float64)
        array1 = CudaArray(data1, device="cpu")
        array2 = CudaArray(data2, device="cpu")
        vectorizer = CorrelationVectorizer(use_gpu=False)
        result = vectorizer.vectorize_correlation(array1, array2, method="direct")
        # Direct correlation returns scalar
        assert result is not None

    def test_invalid_method(self):
        """Test invalid correlation method."""
        data1 = np.array([1, 2, 3])
        data2 = np.array([2, 3, 4])
        array1 = CudaArray(data1, device="cpu")
        array2 = CudaArray(data2, device="cpu")
        vectorizer = CorrelationVectorizer(use_gpu=False)
        with pytest.raises(ValueError, match="Unsupported method"):
            vectorizer.vectorize_correlation(array1, array2, method="invalid")

    def test_process_whole_array_error(self):
        """Test that _process_whole_array raises NotImplementedError."""
        vectorizer = CorrelationVectorizer(use_gpu=False)
        data = np.array([1, 2, 3])
        with pytest.raises(NotImplementedError):
            vectorizer._process_whole_array(data)
