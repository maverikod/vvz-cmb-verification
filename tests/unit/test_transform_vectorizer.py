"""
Unit tests for TransformVectorizer class.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np

from utils.cuda.array_model import CudaArray
from utils.cuda.transform_vectorizer import TransformVectorizer


class TestTransformVectorizer:
    """Test TransformVectorizer class."""

    def test_init(self):
        """Test initialization."""
        vectorizer = TransformVectorizer(use_gpu=False)
        assert vectorizer.whole_array  # Always True for transforms

    def test_fft(self):
        """Test FFT operation."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        array = CudaArray(data, device="cpu")
        vectorizer = TransformVectorizer(use_gpu=False)
        result = vectorizer.vectorize_transform(array, "fft")
        expected = np.fft.fft(data)
        assert np.allclose(result.to_numpy(), expected)

    def test_ifft(self):
        """Test inverse FFT operation."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.complex128)
        array = CudaArray(data, device="cpu")
        vectorizer = TransformVectorizer(use_gpu=False)
        # First do FFT
        fft_result = vectorizer.vectorize_transform(array, "fft")
        # Then do IFFT
        result = vectorizer.vectorize_transform(fft_result, "ifft")
        # Should get back original (within numerical precision)
        assert np.allclose(result.to_numpy(), data, rtol=1e-10)

    def test_rfft(self):
        """Test real FFT operation."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        array = CudaArray(data, device="cpu")
        vectorizer = TransformVectorizer(use_gpu=False)
        result = vectorizer.vectorize_transform(array, "rfft")
        expected = np.fft.rfft(data)
        assert np.allclose(result.to_numpy(), expected)

    def test_irfft(self):
        """Test inverse real FFT operation."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        array = CudaArray(data, device="cpu")
        vectorizer = TransformVectorizer(use_gpu=False)
        # First do RFFT
        rfft_result = vectorizer.vectorize_transform(array, "rfft")
        # Then do IRFFT
        result = vectorizer.vectorize_transform(rfft_result, "irfft", n=len(data))
        # Should get back original (within numerical precision)
        assert np.allclose(result.to_numpy(), data, rtol=1e-10)

    def test_invalid_transform(self):
        """Test invalid transform."""
        data = np.array([1, 2, 3])
        array = CudaArray(data, device="cpu")
        vectorizer = TransformVectorizer(use_gpu=False)
        with pytest.raises(ValueError, match="Unsupported transform"):
            vectorizer.vectorize_transform(array, "invalid_transform")

    def test_2d_fft(self):
        """Test FFT on 2D array."""
        data = np.array([[1, 2], [3, 4]], dtype=np.complex128)
        array = CudaArray(data, device="cpu")
        vectorizer = TransformVectorizer(use_gpu=False)
        result = vectorizer.vectorize_transform(array, "fft")
        expected = np.fft.fft(data)
        assert np.allclose(result.to_numpy(), expected)
