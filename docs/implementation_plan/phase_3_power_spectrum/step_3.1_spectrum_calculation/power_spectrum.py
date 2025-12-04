"""
Power spectrum calculation module.

Calculates C_l power spectrum DIRECTLY from Θ-field frequency spectrum ρ_Θ(ω,t)
with C_l ∝ l⁻²–l⁻³ behavior and no Silk damping.

Uses formulas (from tech_spec-new.md):
- Formula 2.2: l(ω) ≈ π D ω
- Formula 2.3: C_l ∝ ρ_Θ(ω(ℓ)) / ℓ²
- Formula 2.4: C_l ∝ l⁻² (result)

Note: Formula 2.3 is equivalent to CMB2.3: C_l ∝ ρ_Θ(ω(l)) · |dω/dl|
where |dω/dl| = 1/(π D) gives C_l ∝ ρ_Θ(ω(ℓ)) / ℓ².

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass
from cmb.theta_data_loader import ThetaFrequencySpectrum, ThetaEvolution
from cmb.theta_evolution_processor import ThetaEvolutionProcessor
from utils.math.frequency_conversion import (
    frequency_to_multipole,
    multipole_to_frequency,
)
from config.settings import Config


@dataclass
class PowerSpectrum:
    """
    CMB power spectrum data.

    Attributes:
        multipoles: Multipole array l
        spectrum: Power spectrum values C_l
        errors: Error bars σ_C_l
        l_max: Maximum multipole
        metadata: Additional metadata
    """

    multipoles: np.ndarray
    spectrum: np.ndarray
    errors: np.ndarray
    l_max: int
    metadata: dict


class PowerSpectrumCalculator:
    """
    CMB power spectrum calculator.

    Calculates C_l DIRECTLY from Θ-field frequency spectrum ρ_Θ(ω,t).
    Uses formulas CMB2.2-CMB2.4: l(ω) ≈ π D ω, C_l ∝ ρ_Θ(ω(l)) · |dω/dl|.

    This is fundamentally different from classical approach which decomposes
    reconstructed map into spherical harmonics.
    """

    def __init__(
        self,
        frequency_spectrum: ThetaFrequencySpectrum,
        evolution: ThetaEvolution,
        evolution_processor: ThetaEvolutionProcessor,
        l_max: int = 10000,
        config: Optional[Config] = None,
    ):
        """
        Initialize calculator.

        Args:
            frequency_spectrum: Θ-field frequency spectrum ρ_Θ(ω,t)
            evolution: Temporal evolution data
            evolution_processor: Processed evolution data with interpolators
            l_max: Maximum multipole to calculate
            config: Configuration instance (uses global if None)
        """
        self.frequency_spectrum = frequency_spectrum
        self.evolution = evolution
        self.evolution_processor = evolution_processor
        self.l_max = l_max
        self.config = config

    def calculate_spectrum(self) -> PowerSpectrum:
        """
        Calculate C_l power spectrum directly from ρ_Θ(ω,t).

        Uses Formula 2.3 (from tech_spec-new.md): C_l ∝ ρ_Θ(ω(ℓ)) / ℓ²
        where ω(ℓ) = ℓ/(π D) from Formula 2.2.

        Result: C_l ∝ l⁻² (Formula 2.4)

        No plasma terms, no sound horizons.

        Returns:
            PowerSpectrum instance

        Raises:
            ValueError: If calculation fails
        """
        pass

    def _convert_frequencies_to_multipoles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert frequency array to multipole array using CMB2.2.

        Formula: l(ω) ≈ π D ω

        Returns:
            Tuple of (multipoles l, frequencies ω for each l)
        """
        pass

    def _calculate_cl_from_spectrum(
        self, multipoles: np.ndarray, frequencies: np.ndarray
    ) -> np.ndarray:
        """
        Calculate C_l directly from ρ_Θ(ω,t) using Formula 2.3.

        Formula (from tech_spec-new.md 2.3): C_l ∝ ρ_Θ(ω(ℓ)) / ℓ²

        This is equivalent to CMB2.3: C_l ∝ ρ_Θ(ω(l)) · |dω/dl|
        where |dω/dl| = 1/(π D) gives C_l ∝ ρ_Θ(ω(ℓ)) / ℓ².

        No plasma terms, no sound horizons.

        Args:
            multipoles: Multipole array l
            frequencies: Corresponding frequencies ω(l) = l/(π D)

        Returns:
            Power spectrum values C_l
        """
        pass

    def _integrate_temporal_evolution(
        self, cl_values: np.ndarray, multipoles: np.ndarray
    ) -> np.ndarray:
        """
        Integrate over temporal evolution ω_min(t), ω_macro(t).

        Accounts for time-dependent evolution of frequency spectrum.

        Args:
            cl_values: Initial C_l values
            multipoles: Multipole array

        Returns:
            C_l values with temporal evolution integrated
        """
        pass

    def _calculate_errors(self, cl: np.ndarray) -> np.ndarray:
        """
        Calculate error bars for power spectrum.

        Args:
            cl: Power spectrum values

        Returns:
            Error bars σ_C_l
        """
        pass

    def validate_power_law(self, spectrum: PowerSpectrum) -> bool:
        """
        Validate C_l ∝ l⁻²–l⁻³ behavior.

        Checks that calculated spectrum follows theoretical prediction
        C_l ∝ l⁻² from CMB2.4, with possible range l⁻²–l⁻³.

        Args:
            spectrum: PowerSpectrum to validate

        Returns:
            True if power law behavior is valid

        Raises:
            ValueError: If spectrum does not match theoretical prediction
        """
        pass
