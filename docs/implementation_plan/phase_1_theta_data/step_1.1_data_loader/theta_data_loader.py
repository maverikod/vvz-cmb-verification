"""
Θ-field data loader for CMB verification project.

Loads frequency spectrum ρ_Θ(ω,t) and temporal evolution data
from data/theta/ directory.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass
from config.settings import Config
from utils.io.data_loader import load_csv_data, load_json_data


@dataclass
class ThetaFrequencySpectrum:
    """
    Θ-field frequency spectrum data.

    Attributes:
        frequencies: Frequency array ω in Hz
        times: Time array t
        spectrum: Spectrum values ρ_Θ(ω,t) as 2D array
        metadata: Additional metadata (units, ranges, etc.)
    """

    frequencies: np.ndarray
    times: np.ndarray
    spectrum: np.ndarray
    metadata: dict


@dataclass
class ThetaEvolution:
    """
    Θ-field temporal evolution data.

    Attributes:
        times: Time array t
        omega_min: ω_min(t) values
        omega_macro: ω_macro(t) values
        metadata: Additional metadata
    """

    times: np.ndarray
    omega_min: np.ndarray
    omega_macro: np.ndarray
    metadata: dict


def load_frequency_spectrum(data_path: Optional[Path] = None) -> ThetaFrequencySpectrum:
    """
    Load Θ-field frequency spectrum ρ_Θ(ω,t).

    Args:
        data_path: Path to frequency spectrum data file.
                   If None, uses path from data index.

    Returns:
        ThetaFrequencySpectrum instance

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    pass


def load_evolution_data(data_path: Optional[Path] = None) -> ThetaEvolution:
    """
    Load Θ-field temporal evolution data (ω_min(t), ω_macro(t)).

    Args:
        data_path: Path to evolution data file.
                   If None, uses path from data index.

    Returns:
        ThetaEvolution instance

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    pass


def validate_frequency_spectrum(spectrum: ThetaFrequencySpectrum) -> bool:
    """
    Validate frequency spectrum data.

    Args:
        spectrum: ThetaFrequencySpectrum to validate

    Returns:
        True if valid

    Raises:
        ValueError: If data is invalid
    """
    pass


def validate_evolution_data(evolution: ThetaEvolution) -> bool:
    """
    Validate evolution data.

    Args:
        evolution: ThetaEvolution to validate

    Returns:
        True if valid

    Raises:
        ValueError: If data is invalid
    """
    pass
