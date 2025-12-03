"""
High-l peak prediction module.

Predicts and validates high-l peak at l≈4500-6000 for ACT/SPT
observations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from cmb.spectrum.power_spectrum import PowerSpectrum
from cmb.spectrum.subpeaks_analyzer import SubPeakAnalysis
from cmb.theta_evolution_processor import ThetaEvolutionProcessor
from utils.io.data_loader import load_power_spectrum_from_tar


@dataclass
class PeakPrediction:
    """
    High-l peak prediction.
    
    Attributes:
        l_position: Predicted multipole position
        l_position_uncertainty: Position uncertainty
        amplitude: Predicted peak amplitude
        amplitude_uncertainty: Amplitude uncertainty
        theoretical_basis: Theoretical justification
    """
    l_position: float
    l_position_uncertainty: float
    amplitude: float
    amplitude_uncertainty: float
    theoretical_basis: str


@dataclass
class PeakValidation:
    """
    Peak validation results.
    
    Attributes:
        predicted: Peak prediction
        observed_position: Observed peak position (if found)
        observed_amplitude: Observed peak amplitude (if found)
        agreement: Agreement metrics
        validation_passed: Boolean validation result
    """
    predicted: PeakPrediction
    observed_position: Optional[float]
    observed_amplitude: Optional[float]
    agreement: Dict[str, float]
    validation_passed: bool


class HighlPeakPredictor:
    """
    High-l peak predictor.
    
    Predicts l≈4500-6000 peak and validates with observations.
    """
    
    def __init__(
        self,
        power_spectrum: PowerSpectrum,
        subpeak_analysis: SubPeakAnalysis,
        evolution_processor: ThetaEvolutionProcessor
    ):
        """
        Initialize predictor.
        
        Args:
            power_spectrum: Power spectrum data
            subpeak_analysis: Sub-peak analysis results
            evolution_processor: Θ-field evolution processor
        """
        self.power_spectrum = power_spectrum
        self.subpeak_analysis = subpeak_analysis
        self.evolution_processor = evolution_processor
    
    def predict_peak(
        self,
        l_min: float = 4500.0,
        l_max: float = 6000.0
    ) -> PeakPrediction:
        """
        Predict high-l peak position and amplitude.
        
        Args:
            l_min: Minimum multipole for prediction range
            l_max: Maximum multipole for prediction range
            
        Returns:
            PeakPrediction instance
            
        Raises:
            ValueError: If prediction fails
        """
        pass
    
    def compare_with_observations(
        self,
        prediction: PeakPrediction,
        observed_spectrum_path: Path
    ) -> PeakValidation:
        """
        Compare prediction with ACT/SPT observations.
        
        Args:
            prediction: Peak prediction
            observed_spectrum_path: Path to observed spectrum
            
        Returns:
            PeakValidation instance
            
        Raises:
            FileNotFoundError: If observed data doesn't exist
        """
        pass
    
    def calculate_agreement_metrics(
        self,
        predicted: PeakPrediction,
        observed_position: float,
        observed_amplitude: float
    ) -> Dict[str, float]:
        """
        Calculate agreement metrics between prediction and observation.
        
        Args:
            predicted: Peak prediction
            observed_position: Observed peak position
            observed_amplitude: Observed peak amplitude
            
        Returns:
            Dictionary with agreement metrics
        """
        pass
