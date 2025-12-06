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

    def test_init(self):
        """Test initialization."""
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        assert not vectorizer.use_gpu
        assert not vectorizer.whole_array

    def test_init_with_options(self):
        """Test initialization with options."""
        vectorizer = ElementWiseVectorizer(
            use_gpu=True, block_size=1000, whole_array=True
        )
        assert vectorizer.whole_array

    def test_multiply(self):
        """Test multiplication operation."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.multiply(array, 2)
        expected = data * 2
        assert np.allclose(result.to_numpy(), expected)

    def test_multiply_with_array(self):
        """Test multiplication with array operand."""
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([2, 3, 4, 5, 6])
        array1 = CudaArray(data1, device="cpu")
        array2 = CudaArray(data2, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.multiply(array1, data2)
        expected = data1 * data2
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

    def test_divide(self):
        """Test division operation."""
        data = np.array([2, 4, 6, 8, 10])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.divide(array, 2)
        expected = data / 2
        assert np.allclose(result.to_numpy(), expected)

    def test_power(self):
        """Test power operation."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.power(array, 2)
        expected = data**2
        assert np.allclose(result.to_numpy(), expected)

    def test_sqrt(self):
        """Test square root operation."""
        data = np.array([1, 4, 9, 16, 25])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.sqrt(array)
        expected = np.sqrt(data)
        assert np.allclose(result.to_numpy(), expected)

    def test_abs(self):
        """Test absolute value operation."""
        data = np.array([-1, -2, 0, 2, 3])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.abs(array)
        expected = np.abs(data)
        assert np.allclose(result.to_numpy(), expected)

    def test_sin(self):
        """Test sine operation."""
        data = np.array([0, np.pi / 2, np.pi])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.sin(array)
        expected = np.sin(data)
        assert np.allclose(result.to_numpy(), expected)

    def test_cos(self):
        """Test cosine operation."""
        data = np.array([0, np.pi / 2, np.pi])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.cos(array)
        expected = np.cos(data)
        assert np.allclose(result.to_numpy(), expected)

    def test_exp(self):
        """Test exponential operation."""
        data = np.array([0, 1, 2], dtype=np.float64)
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.exp(array)
        expected = np.exp(data)
        assert np.allclose(result.to_numpy(), expected)

    def test_log(self):
        """Test natural logarithm operation."""
        data = np.array([1, np.e, np.e**2])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.log(array)
        expected = np.log(data)
        assert np.allclose(result.to_numpy(), expected)

    def test_vectorize_operation_invalid(self):
        """Test invalid operation."""
        data = np.array([1, 2, 3])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        with pytest.raises(ValueError, match="Unsupported operation"):
            vectorizer.vectorize_operation(array, "invalid_op", None)

    def test_vectorize_operation_unary(self):
        """Test unary operation via vectorize_operation."""
        data = np.array([1, 4, 9])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.vectorize_operation(array, "sqrt", None)
        expected = np.sqrt(data)
        assert np.allclose(result.to_numpy(), expected)

    def test_vectorize_operation_binary(self):
        """Test binary operation via vectorize_operation."""
        data = np.array([1, 2, 3])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.vectorize_operation(array, "multiply", 2)
        expected = data * 2
        assert np.allclose(result.to_numpy(), expected)

    def test_whole_array_mode(self):
        """Test whole array mode."""
        data = np.array([1, 2, 3, 4, 5])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False, whole_array=True)
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
        results = vectorizer.batch([array1, array2], "multiply", 2)
        assert len(results) == 2
        assert np.allclose(results[0].to_numpy(), data1 * 2)
        assert np.allclose(results[1].to_numpy(), data2 * 2)

    def test_2d_array_operations(self):
        """Test operations on 2D arrays."""
        data = np.array([[1, 2], [3, 4]])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.multiply(array, 2)
        expected = data * 2
        assert np.allclose(result.to_numpy(), expected)

    def test_float_operations(self):
        """Test operations with float operands."""
        data = np.array([1.5, 2.5, 3.5])
        array = CudaArray(data, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.add(array, 0.5)
        expected = data + 0.5
        assert np.allclose(result.to_numpy(), expected)

    def test_operation_with_cuda_array_operand(self):
        """Test operation with CudaArray as operand."""
        data1 = np.array([1, 2, 3])
        data2 = np.array([2, 3, 4])
        array1 = CudaArray(data1, device="cpu")
        array2 = CudaArray(data2, device="cpu")
        vectorizer = ElementWiseVectorizer(use_gpu=False)
        result = vectorizer.add(array1, array2)
        expected = data1 + data2
        assert np.allclose(result.to_numpy(), expected)
