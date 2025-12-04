"""
Transform vectorizer for CUDA acceleration.

Provides vectorized transform operations requiring whole arrays
(FFT, spherical harmonics).

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.cuda.array_model import CudaArray

import numpy as np

from utils.cuda.vectorizer import BaseVectorizer

# Try to import CuPy
try:
    import cupy as cp
    from cupyx.scipy import fft as cupy_fft

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cupy_fft = None


class TransformVectorizer(BaseVectorizer):
    """
    Vectorizer for transform operations requiring whole arrays.

    Supports:
    - FFT / IFFT
    - Real FFT / Inverse real FFT
    - Spherical harmonic transforms (via healpy)

    Note: Always uses whole_array=True mode (transforms require contiguous data).
    """

    def __init__(
        self,
        use_gpu: bool = True,
        block_size: Optional[int] = None,
        whole_array: bool = True,  # Always True for transforms
    ):
        """
        Initialize transform vectorizer.

        Args:
            use_gpu: Use GPU acceleration if available
            block_size: Block size (ignored, transforms use whole array)
            whole_array: Use whole array mode (always True for transforms)
        """
        # Force whole_array=True for transforms
        super().__init__(use_gpu=use_gpu, block_size=None, whole_array=True)

    def vectorize_transform(
        self,
        array: "CudaArray",  # noqa: F821
        transform: str,
        *args: Any,
        **kwargs: Any,
    ) -> "CudaArray":  # noqa: F821
        """
        Apply transform operation.

        Args:
            array: Input CudaArray
            transform: Transform name ("fft", "ifft", "rfft", "irfft",
                "sph_harm_synthesis")
            *args: Transform-specific arguments
            **kwargs: Transform-specific keyword arguments

        Returns:
            Result CudaArray

        Raises:
            ValueError: If transform not supported
        """
        # Store transform type for use in _process_whole_array
        self._current_transform = transform
        self._transform_args = args
        self._transform_kwargs = kwargs

        return self.vectorize(array, *args, **kwargs)

    def _process_block(
        self, block: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process single block (not used for transforms, but required by base class).

        Args:
            block: Block data
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed block (delegates to whole array processing)
        """
        # For transforms, always process as whole array
        return self._process_whole_array(block, *args, **kwargs)

    def _process_whole_array(
        self, array: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process whole array with transform operation.

        Args:
            array: Whole array data
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed array
        """
        transform = getattr(self, "_current_transform", None)
        if transform is None:
            # Try to get from args/kwargs if not set
            transform = kwargs.get("transform", None)
            if transform is None and len(args) > 0:
                transform = args[0]

        if transform is None:
            raise ValueError("Transform type not specified")

        # Apply transform
        if transform == "fft":
            return self._fft(array)
        elif transform == "ifft":
            return self._ifft(array)
        elif transform == "rfft":
            return self._rfft(array)
        elif transform == "irfft":
            return self._irfft(array, **kwargs)
        elif transform == "sph_harm_synthesis":
            return self._sph_harm_synthesis(array, **kwargs)
        elif transform == "sph_harm_analysis":
            return self._sph_harm_analysis(array, **kwargs)
        else:
            raise ValueError(f"Unsupported transform: {transform}")

    def _fft(self, array: np.ndarray) -> np.ndarray:
        """
        Forward FFT.

        Args:
            array: Input array

        Returns:
            FFT result
        """
        if self.use_gpu and CUPY_AVAILABLE:
            if not isinstance(array, cp.ndarray):
                array = cp.asarray(array)
            return cupy_fft.fft(array)
        else:
            return np.fft.fft(array)

    def _ifft(self, array: np.ndarray) -> np.ndarray:
        """
        Inverse FFT.

        Args:
            array: Input array

        Returns:
            IFFT result
        """
        if self.use_gpu and CUPY_AVAILABLE:
            if not isinstance(array, cp.ndarray):
                array = cp.asarray(array)
            return cupy_fft.ifft(array)
        else:
            return np.fft.ifft(array)

    def _rfft(self, array: np.ndarray) -> np.ndarray:
        """
        Real FFT.

        Args:
            array: Input real array

        Returns:
            Real FFT result
        """
        if self.use_gpu and CUPY_AVAILABLE:
            if not isinstance(array, cp.ndarray):
                array = cp.asarray(array)
            return cupy_fft.rfft(array)
        else:
            return np.fft.rfft(array)

    def _irfft(self, array: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        """
        Inverse real FFT.

        Args:
            array: Input array
            n: Output length (for real FFT)

        Returns:
            Inverse real FFT result
        """
        if self.use_gpu and CUPY_AVAILABLE:
            if not isinstance(array, cp.ndarray):
                array = cp.asarray(array)
            return cupy_fft.irfft(array, n=n)
        else:
            return np.fft.irfft(array, n=n)

    def _sph_harm_synthesis(
        self,
        alm: np.ndarray,
        lmax: int,
        nside: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Spherical harmonic synthesis (alm → map).

        Args:
            alm: Spherical harmonic coefficients
            lmax: Maximum multipole
            nside: HEALPix nside (if None, calculated from alm)
            **kwargs: Additional arguments for healpy

        Returns:
            HEALPix map

        Note:
            Currently uses CPU healpy. GPU-accelerated spherical harmonics
            would require custom CUDA kernels or specialized library.
        """
        # Import healpy for spherical harmonics
        try:
            import healpy as hp
        except ImportError:
            raise ImportError("healpy required for spherical harmonic operations")

        # Convert to numpy if CuPy array
        if self.use_gpu and CUPY_AVAILABLE and isinstance(alm, cp.ndarray):
            alm = cp.asnumpy(alm)

        # Use healpy for synthesis (CPU-based)
        # Note: GPU-accelerated spherical harmonics would require custom implementation
        if nside is None:
            # Estimate nside from lmax
            nside = hp.npix2nside(len(alm))

        map_result = hp.alm2map(alm, nside=nside, lmax=lmax, **kwargs)

        # Convert back to GPU if using GPU
        if self.use_gpu and CUPY_AVAILABLE:
            return cp.asarray(map_result)
        return map_result

    def _sph_harm_analysis(
        self, map_data: np.ndarray, lmax: int, **kwargs: Any
    ) -> np.ndarray:
        """
        Spherical harmonic analysis (map → alm).

        Args:
            map_data: HEALPix map
            lmax: Maximum multipole
            **kwargs: Additional arguments for healpy

        Returns:
            Spherical harmonic coefficients

        Note:
            Currently uses CPU healpy. GPU-accelerated spherical harmonics
            would require custom CUDA kernels or specialized library.
        """
        # Import healpy for spherical harmonics
        try:
            import healpy as hp
        except ImportError:
            raise ImportError("healpy required for spherical harmonic operations")

        # Convert to numpy if CuPy array
        if self.use_gpu and CUPY_AVAILABLE and isinstance(map_data, cp.ndarray):
            map_data = cp.asnumpy(map_data)

        # Use healpy for analysis (CPU-based)
        alm = hp.map2alm(map_data, lmax=lmax, **kwargs)

        # Convert back to GPU if using GPU
        if self.use_gpu and CUPY_AVAILABLE:
            return cp.asarray(alm)
        return alm
