"""
Frequency to multipole conversion utilities.

Implements the conversion formula: l ≈ π D ω
as specified in tech_spec.md section 13.8.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Union
import numpy as np
from config.settings import Config


def frequency_to_multipole(
    frequency: Union[float, np.ndarray],
    D: Optional[float] = None
) -> Union[float, np.ndarray]:
    """
    Convert frequency to multipole using l ≈ π D ω.
    
    Args:
        frequency: Frequency in Hz
        D: Distance parameter (if None, uses config value)
        
    Returns:
        Multipole l value(s)
        
    Raises:
        ValueError: If frequency is negative or zero
    """
    pass


def multipole_to_frequency(
    multipole: Union[float, np.ndarray],
    D: Optional[float] = None
) -> Union[float, np.ndarray]:
    """
    Convert multipole to frequency using inverse of l ≈ π D ω.
    
    Args:
        multipole: Multipole l value(s)
        D: Distance parameter (if None, uses config value)
        
    Returns:
        Frequency in Hz
        
    Raises:
        ValueError: If multipole is negative or zero
    """
    pass


def get_frequency_range_for_multipole_range(
    l_min: float,
    l_max: float,
    D: Optional[float] = None
) -> tuple[float, float]:
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
    pass

