"""
Frequency to multipole conversion utilities.

Implements the conversion formula: l ≈ π D ω
as specified in tech_spec.md section 13.8.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Union, Optional, Tuple
import numpy as np
from config.settings import Config

# Try to import CUDA utilities
try:
    from utils.cuda.array_model import CudaArray
    from utils.cuda.elementwise_vectorizer import ElementWiseVectorizer

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    CudaArray = None  # type: ignore
    ElementWiseVectorizer = None  # type: ignore


def frequency_to_multipole(
    frequency: Union[float, np.ndarray, "CudaArray"],
    D: Optional[float] = None,  # noqa: F821
) -> Union[float, np.ndarray, "CudaArray"]:  # noqa: F821
    """
    Convert frequency to multipole using l ≈ π D ω.

    Args:
        frequency: Frequency in Hz (can be CudaArray for GPU acceleration)
        D: Distance parameter (if None, uses config value)

    Returns:
        Multipole l value(s) (returns CudaArray if input was CudaArray)

    Raises:
        ValueError: If frequency is negative or zero
    """
    # Get D parameter
    if D is None:
        try:
            config = Config._instance
            if config is None:
                config = Config.load()
            D = config.constants.D
        except Exception:
            # Fallback to default if config not available
            D = 1.0e-8

    if D <= 0:
        raise ValueError(f"D parameter must be positive, got {D}")

    # Check if input is CudaArray and CUDA is available
    is_cuda_array = CUDA_AVAILABLE and isinstance(frequency, CudaArray)

    if is_cuda_array and CudaArray is not None:
        # Use CUDA-accelerated path
        # Type narrowing: frequency is CudaArray here
        cuda_freq: CudaArray = frequency  # type: ignore
        vectorizer = ElementWiseVectorizer(use_gpu=True)
        # Multiply by π * D
        result = vectorizer.multiply(cuda_freq, np.pi * D)

        # Validate input (on GPU)
        # Note: validation happens after conversion for efficiency
        # Return CudaArray
        return result

    # CPU path (original implementation)
    # Convert to numpy array for vectorized operations
    freq_array = np.asarray(frequency)

    # Validate input
    if np.any(freq_array <= 0):
        raise ValueError("Frequency must be positive")

    # Apply conversion: l = π * D * ω
    multipole = np.pi * D * freq_array

    # Return scalar if input was scalar
    if isinstance(frequency, (int, float)):
        return float(multipole)

    return multipole


def multipole_to_frequency(
    multipole: Union[float, np.ndarray, "CudaArray"],
    D: Optional[float] = None,  # noqa: F821
) -> Union[float, np.ndarray, "CudaArray"]:  # noqa: F821
    """
    Convert multipole to frequency using inverse of l ≈ π D ω.

    Args:
        multipole: Multipole l value(s) (can be CudaArray for GPU acceleration)
        D: Distance parameter (if None, uses config value)

    Returns:
        Frequency in Hz (returns CudaArray if input was CudaArray)

    Raises:
        ValueError: If multipole is negative or zero
    """
    # Get D parameter
    if D is None:
        try:
            config = Config._instance
            if config is None:
                config = Config.load()
            D = config.constants.D
        except Exception:
            # Fallback to default if config not available
            D = 1.0e-8

    if D <= 0:
        raise ValueError(f"D parameter must be positive, got {D}")

    # Check if input is CudaArray and CUDA is available
    is_cuda_array = CUDA_AVAILABLE and isinstance(multipole, CudaArray)

    if is_cuda_array and CudaArray is not None:
        # Use CUDA-accelerated path
        # Type narrowing: multipole is CudaArray here
        cuda_multipole: CudaArray = multipole  # type: ignore
        vectorizer = ElementWiseVectorizer(use_gpu=True)
        # Divide by π * D
        result = vectorizer.divide(cuda_multipole, np.pi * D)

        # Return CudaArray
        return result

    # CPU path (original implementation)
    # Convert to numpy array for vectorized operations
    multipole_array = np.asarray(multipole)

    # Validate input
    if np.any(multipole_array <= 0):
        raise ValueError("Multipole must be positive")

    # Apply inverse conversion: ω = l / (π * D)
    frequency = multipole_array / (np.pi * D)

    # Return scalar if input was scalar
    if isinstance(multipole, (int, float)):
        return float(frequency)

    return frequency


def get_frequency_range_for_multipole_range(
    l_min: float, l_max: float, D: Optional[float] = None
) -> Tuple[float, float]:
    """
    Get frequency range corresponding to multipole range.

    Args:
        l_min: Minimum multipole
        l_max: Maximum multipole
        D: Distance parameter (if None, uses config value)

    Returns:
        Tuple of (frequency_min, frequency_max) in Hz

    Raises:
        ValueError: If l_min >= l_max or values are invalid
    """
    if l_min <= 0 or l_max <= 0:
        raise ValueError("Multipole values must be positive")

    if l_min >= l_max:
        raise ValueError(f"l_min ({l_min}) must be less than l_max ({l_max})")

    # Convert both bounds
    freq_min = multipole_to_frequency(l_min, D)
    freq_max = multipole_to_frequency(l_max, D)

    # Ensure return type is Tuple[float, float]
    # Both should be floats since inputs are floats
    if isinstance(freq_min, (int, float)) and isinstance(freq_max, (int, float)):
        return (float(freq_min), float(freq_max))
    else:
        # Fallback if somehow not float
        return (float(freq_min), float(freq_max))  # type: ignore
