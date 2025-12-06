"""
Spherical harmonics utilities for CMB verification project.

Provides functions for spherical harmonic decomposition and synthesis.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Tuple, Union
import numpy as np
import healpy as hp

# Try to import CUDA utilities
try:
    from utils.cuda.array_model import CudaArray
    from utils.cuda.transform_vectorizer import TransformVectorizer

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    CudaArray = None  # type: ignore
    TransformVectorizer = None  # type: ignore


def decompose_map(
    map_data: Union[np.ndarray, "CudaArray"],
    nside: int,
    l_max: Optional[int] = None,
    use_cuda: bool = False,  # noqa: F821
) -> Union[np.ndarray, "CudaArray"]:  # noqa: F821
    """
    Decompose HEALPix map into spherical harmonics.

    Args:
        map_data: HEALPix map array (can be CudaArray for GPU acceleration)
        nside: HEALPix NSIDE parameter
        l_max: Maximum multipole (if None, uses default)
        use_cuda: Use CUDA acceleration if available
                  (only if map_data is CudaArray)

    Returns:
        Array of harmonic coefficients a_lm
        (returns CudaArray if input was CudaArray)

    Raises:
        ValueError: If map data is invalid
    """
    # Check if input is CudaArray and CUDA is available
    is_cuda_array = CUDA_AVAILABLE and isinstance(map_data, CudaArray)

    if (
        is_cuda_array
        and use_cuda
        and CudaArray is not None
        and TransformVectorizer is not None
    ):
        # Use CUDA-accelerated path via TransformVectorizer
        # Type narrowing: map_data is CudaArray here
        cuda_map: CudaArray = map_data  # type: ignore

        # Validate map size
        expected_npix = 12 * nside * nside
        map_numpy = cuda_map.to_numpy()
        if map_numpy.size != expected_npix:
            raise ValueError(
                f"Map size {map_numpy.size} does not match NSIDE {nside} "
                f"(expected {expected_npix} pixels)"
            )

        # Determine l_max if not provided
        if l_max is None:
            l_max = 3 * nside - 1

        if l_max < 0:
            raise ValueError(f"l_max must be non-negative, got {l_max}")

        try:
            # Use TransformVectorizer for decomposition
            # Note: healpy doesn't support GPU directly, but
            # TransformVectorizer handles CPU-GPU conversion and provides
            # consistent interface
            vectorizer = TransformVectorizer(use_gpu=True, whole_array=True)
            alm = vectorizer.vectorize_transform(
                cuda_map, "sph_harm_analysis", lmax=l_max, iter=0
            )

            # Result is already CudaArray from vectorizer
            return alm
        except Exception as e:
            raise ValueError(f"Failed to decompose map: {e}") from e

    # CPU path (original implementation)
    if not isinstance(map_data, np.ndarray):
        # Convert CudaArray to numpy if needed
        if is_cuda_array and CudaArray is not None:
            cuda_map: CudaArray = map_data  # type: ignore
            map_data = cuda_map.to_numpy()
        else:
            raise ValueError("Map data must be a numpy array or CudaArray")

    # Validate map size
    expected_npix = 12 * nside * nside
    if map_data.size != expected_npix:
        raise ValueError(
            f"Map size {map_data.size} does not match NSIDE {nside} "
            f"(expected {expected_npix} pixels)"
        )

    # Determine l_max if not provided
    if l_max is None:
        # Default: 3 * nside - 1 (Nyquist limit for HEALPix)
        l_max = 3 * nside - 1

    if l_max < 0:
        raise ValueError(f"l_max must be non-negative, got {l_max}")

    try:
        # Perform spherical harmonic decomposition
        alm = hp.map2alm(map_data, lmax=l_max, iter=0)
        return alm
    except Exception as e:
        raise ValueError(f"Failed to decompose map: {e}") from e


def synthesize_map(
    alm: Union[np.ndarray, "CudaArray"],  # noqa: F821
    nside: int,
    use_cuda: bool = False,
) -> Union[np.ndarray, "CudaArray"]:  # noqa: F821
    """
    Synthesize HEALPix map from spherical harmonics.

    Args:
        alm: Harmonic coefficients a_lm (can be CudaArray for GPU acceleration)
        nside: HEALPix NSIDE parameter
        use_cuda: Use CUDA acceleration if available (only if alm is CudaArray)

    Returns:
        HEALPix map array (returns CudaArray if input was CudaArray)

    Raises:
        ValueError: If coefficients are invalid
    """
    # Check if input is CudaArray and CUDA is available
    is_cuda_array = CUDA_AVAILABLE and isinstance(alm, CudaArray)

    if (
        is_cuda_array
        and use_cuda
        and CudaArray is not None
        and TransformVectorizer is not None
    ):
        # Use CUDA-accelerated path via TransformVectorizer
        # Type narrowing: alm is CudaArray here
        cuda_alm: CudaArray = alm  # type: ignore

        # Check if array is empty using to_numpy()
        alm_numpy_check = cuda_alm.to_numpy()
        if alm_numpy_check.size == 0:
            raise ValueError("Harmonic coefficients array is empty")

        if nside <= 0:
            raise ValueError(f"NSIDE must be positive, got {nside}")

        try:
            # Use TransformVectorizer for synthesis
            # Note: healpy doesn't support GPU directly, but
            # TransformVectorizer handles CPU-GPU conversion and provides
            # consistent interface
            vectorizer = TransformVectorizer(use_gpu=True, whole_array=True)
            map_data = vectorizer.vectorize_transform(
                cuda_alm, "sph_harm_synthesis", lmax=None, nside=nside
            )

            # Result is already CudaArray from vectorizer
            return map_data
        except Exception as e:
            raise ValueError(f"Failed to synthesize map: {e}") from e

    # CPU path (original implementation)
    if not isinstance(alm, np.ndarray):
        # Convert CudaArray to numpy if needed
        if is_cuda_array and CudaArray is not None:
            cuda_alm: CudaArray = alm  # type: ignore
            alm = cuda_alm.to_numpy()
        else:
            raise ValueError("Harmonic coefficients must be a numpy array or CudaArray")

    if alm.size == 0:
        raise ValueError("Harmonic coefficients array is empty")

    if nside <= 0:
        raise ValueError(f"NSIDE must be positive, got {nside}")

    try:
        # Synthesize map from spherical harmonics
        map_data = hp.alm2map(alm, nside=nside)
        return np.asarray(map_data)
    except Exception as e:
        raise ValueError(f"Failed to synthesize map: {e}") from e


def calculate_power_spectrum_from_alm(
    alm: np.ndarray, l_max: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate power spectrum C_l from harmonic coefficients.

    Args:
        alm: Harmonic coefficients a_lm
        l_max: Maximum multipole (if None, uses data limit)

    Returns:
        Tuple of (multipoles, C_l values)

    Raises:
        ValueError: If coefficients are invalid
    """
    if not isinstance(alm, np.ndarray):
        raise ValueError("Harmonic coefficients must be a numpy array")

    if alm.size == 0:
        raise ValueError("Harmonic coefficients array is empty")

    try:
        # Calculate power spectrum
        cl = hp.alm2cl(alm)

        # Get multipole values
        if l_max is None:
            l_max = len(cl) - 1

        multipoles = np.arange(l_max + 1)

        # Truncate to requested l_max
        if len(cl) > l_max + 1:
            cl = cl[: l_max + 1]

        return (multipoles, cl)
    except Exception as e:
        raise ValueError(f"Failed to calculate power spectrum: {e}") from e


def convert_healpix_to_alm(
    map_data: np.ndarray, nside: int, l_max: Optional[int] = None
) -> np.ndarray:
    """
    Convert HEALPix map to spherical harmonic coefficients.

    Args:
        map_data: HEALPix map array
        nside: HEALPix NSIDE parameter
        l_max: Maximum multipole

    Returns:
        Array of harmonic coefficients a_lm
    """
    # This is essentially the same as decompose_map
    return decompose_map(map_data, nside, l_max)
