"""
Correlation vectorizer for CUDA acceleration.

Provides vectorized correlation and convolution operations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.cuda.array_model import CudaArray

import numpy as np

from utils.cuda.vectorizer import BaseVectorizer
from utils.cuda.transform_vectorizer import TransformVectorizer

# Try to import CuPy
try:
    import cupy as cp
    from cupyx.scipy import fft as cupy_fft

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cupy_fft = None


class CorrelationVectorizer(BaseVectorizer):
    """
    Vectorizer for correlation and convolution operations.

    Supports:
    - Cross-correlation
    - Auto-correlation
    - Correlation functions
    - Convolution

    Uses FFT-based correlation for efficiency on large arrays.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        block_size: Optional[int] = None,
        whole_array: bool = True,  # FFT-based correlation requires whole array
    ):
        """
        Initialize correlation vectorizer.

        Args:
            use_gpu: Use GPU acceleration if available
            block_size: Block size (ignored for FFT-based correlation)
            whole_array: Use whole array mode (always True for FFT-based)
        """
        super().__init__(use_gpu=use_gpu, block_size=None, whole_array=True)
        self._transform_vectorizer = TransformVectorizer(use_gpu=use_gpu)

    def vectorize_correlation(
        self,
        array1: "CudaArray",  # noqa: F821
        array2: "CudaArray",  # noqa: F821
        method: str = "fft",
        *args: Any,
        **kwargs: Any,
    ) -> "CudaArray":  # noqa: F821
        """
        Calculate correlation between two arrays.

        Args:
            array1: First array
            array2: Second array
            method: Method ("fft" or "direct")
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Correlation result

        Raises:
            ValueError: If method not supported
        """
        if method == "fft":
            return self._fft_correlation(array1, array2, *args, **kwargs)
        elif method == "direct":
            return self._direct_correlation(array1, array2, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def _fft_correlation(
        self,
        array1: "CudaArray",
        array2: "CudaArray",
        *args: Any,
        **kwargs: Any,  # noqa: F821
    ) -> "CudaArray":  # noqa: F821
        """
        FFT-based cross-correlation.

        Args:
            array1: First array
            array2: Second array
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Correlation result
        """
        # Get whole arrays
        a1 = array1.use_whole_array()
        a2 = array2.use_whole_array()

        # Ensure same shape (pad if needed)
        if a1.shape != a2.shape:
            size = max(a1.size, a2.size)
            if self.use_gpu and CUPY_AVAILABLE:
                a1_padded = cp.zeros(size, dtype=a1.dtype)
                a2_padded = cp.zeros(size, dtype=a2.dtype)
                a1_padded[: a1.size] = a1.flatten()
                a2_padded[: a2.size] = a2.flatten()
            else:
                a1_padded = np.zeros(size, dtype=a1.dtype)
                a2_padded = np.zeros(size, dtype=a2.dtype)
                a1_padded[: a1.size] = a1.flatten()
                a2_padded[: a2.size] = a2.flatten()
        else:
            a1_padded = a1.flatten()
            a2_padded = a2.flatten()

        # FFT
        if self.use_gpu and CUPY_AVAILABLE:
            A1 = cupy_fft.fft(a1_padded)
            A2 = cupy_fft.fft(a2_padded)
            # Cross-correlation in frequency domain
            C = A1 * cp.conj(A2)
            # Inverse FFT
            correlation = cupy_fft.ifft(C)
            correlation = cp.real(correlation)
        else:
            A1 = np.fft.fft(a1_padded)
            A2 = np.fft.fft(a2_padded)
            # Cross-correlation in frequency domain
            C = A1 * np.conj(A2)
            # Inverse FFT
            correlation = np.fft.ifft(C)
            correlation = np.real(correlation)

        # Reshape to original shape if possible
        if len(array1.shape) > 1:
            correlation = correlation.reshape(array1.shape)

        # Create new CudaArray
        from utils.cuda.array_model import CudaArray

        # Convert to numpy if needed
        if self.use_gpu and CUPY_AVAILABLE and isinstance(correlation, cp.ndarray):
            result_numpy = cp.asnumpy(correlation)
        else:
            result_numpy = np.asarray(correlation)
        return CudaArray(result_numpy, block_size=None, device=array1.device)

    def _direct_correlation(
        self,
        array1: "CudaArray",
        array2: "CudaArray",
        *args: Any,
        **kwargs: Any,  # noqa: F821
    ) -> "CudaArray":  # noqa: F821
        """
        Direct correlation (slower but more memory-efficient).

        Args:
            array1: First array
            array2: Second array
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Correlation result
        """
        # For direct correlation, use element-wise operations
        # This is a simplified implementation
        # Full implementation would use convolution

        a1 = array1.use_whole_array()
        a2 = array2.use_whole_array()

        # Simple correlation: element-wise multiply and sum
        if self.use_gpu and CUPY_AVAILABLE:
            if not isinstance(a1, cp.ndarray):
                a1 = cp.asarray(a1)
            if not isinstance(a2, cp.ndarray):
                a2 = cp.asarray(a2)
            correlation = cp.sum(a1 * a2)
        else:
            correlation = np.sum(a1 * a2)

        # Create new CudaArray
        from utils.cuda.array_model import CudaArray

        # Convert to numpy if needed
        if self.use_gpu and CUPY_AVAILABLE and isinstance(correlation, cp.ndarray):
            result_numpy = cp.asnumpy(correlation)
        else:
            result_numpy = np.asarray(correlation)
        return CudaArray(result_numpy, block_size=None, device=array1.device)

    def _process_block(
        self, block: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process single block (not used for correlation, but required by base class).

        Args:
            block: Block data
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed block
        """
        # Correlation requires whole arrays
        return self._process_whole_array(block, *args, **kwargs)

    def _process_whole_array(
        self, array: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process whole array (not directly used, correlation uses vectorize_correlation).

        Args:
            array: Whole array data
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed array
        """
        # This method is not directly used for correlation
        # Correlation requires two arrays, use vectorize_correlation instead
        raise NotImplementedError(
            "Use vectorize_correlation() method for correlation operations"
        )
