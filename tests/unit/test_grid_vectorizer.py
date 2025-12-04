"""
Unit tests for GridVectorizer class.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np

from utils.cuda.array_model import CudaArray
from utils.cuda.grid_vectorizer import GridVectorizer


class TestGridVectorizer:
    """Test GridVectorizer class."""

    def test_init(self):
        """Test initialization."""
        vectorizer = GridVectorizer(use_gpu=False)
        assert not vectorizer.whole_array

    def test_local_minima(self):
        """Test local minimum detection."""
        # Create test grid with known minima
        data = np.array([[5, 4, 5], [4, 1, 4], [5, 4, 5]], dtype=np.float64)
        array = CudaArray(data, device="cpu")
        vectorizer = GridVectorizer(use_gpu=False)
        result = vectorizer.vectorize_grid_operation(
            array, "local_minima", neighborhood_size=3
        )
        # Center should be minimum
        result_np = result.to_numpy()
        assert result_np[1, 1] == True  # Center is minimum

    def test_local_maxima(self):
        """Test local maximum detection."""
        # Create test grid with known maxima
        data = np.array([[1, 2, 1], [2, 5, 2], [1, 2, 1]], dtype=np.float64)
        array = CudaArray(data, device="cpu")
        vectorizer = GridVectorizer(use_gpu=False)
        result = vectorizer.vectorize_grid_operation(
            array, "local_maxima", neighborhood_size=3
        )
        # Center should be maximum
        result_np = result.to_numpy()
        assert result_np[1, 1] == True  # Center is maximum

    def test_gradient(self):
        """Test gradient calculation."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        array = CudaArray(data, device="cpu")
        vectorizer = GridVectorizer(use_gpu=False)
        result = vectorizer.vectorize_grid_operation(array, "gradient")
        # Gradient should be calculated
        assert result is not None

    def test_laplacian(self):
        """Test Laplacian calculation."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        array = CudaArray(data, device="cpu")
        vectorizer = GridVectorizer(use_gpu=False)
        result = vectorizer.vectorize_grid_operation(array, "laplacian")
        # Laplacian should be calculated
        assert result is not None

    def test_curvature(self):
        """Test curvature calculation."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        array = CudaArray(data, device="cpu")
        vectorizer = GridVectorizer(use_gpu=False)
        result = vectorizer.vectorize_grid_operation(array, "curvature")
        # Curvature should be calculated
        assert result is not None

    def test_invalid_operation(self):
        """Test invalid grid operation."""
        data = np.array([[1, 2], [3, 4]])
        array = CudaArray(data, device="cpu")
        vectorizer = GridVectorizer(use_gpu=False)
        with pytest.raises(ValueError, match="Unsupported operation"):
            vectorizer.vectorize_grid_operation(array, "invalid_op")

    def test_1d_local_minima(self):
        """Test local minima on 1D array."""
        data = np.array([5, 4, 1, 4, 5], dtype=np.float64)
        array = CudaArray(data, device="cpu")
        vectorizer = GridVectorizer(use_gpu=False)
        result = vectorizer.vectorize_grid_operation(
            array, "local_minima", neighborhood_size=3
        )
        # Center should be minimum
        result_np = result.to_numpy()
        assert result_np[2] == True  # Center is minimum
