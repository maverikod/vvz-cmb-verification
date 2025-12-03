"""
Predictions report generation module.

Compiles all ACT/SPT predictions and generates comprehensive
validation report.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from cmb.predictions.highl_peak_predictor import PeakValidation
from cmb.predictions.frequency_invariance import InvarianceTestResult
from cmb.spectrum.subpeaks_analyzer import SubPeakAnalysis
from cmb.correlation.phi_split_analyzer import PhiSplitResult


@dataclass
class PredictionSummary:
    """
    Summary of a single prediction.
    
    Attributes:
        prediction_name: Name of prediction
        predicted_value: Predicted value
        observed_value: Observed value (if available)
        agreement: Agreement metric
        validated: Boolean validation result
    """
    prediction_name: str
    predicted_value: any
    observed_value: Optional[any]
    agreement: float
    validated: bool


@dataclass
class PredictionsReport:
    """
    Comprehensive predictions report.
    
    Attributes:
        predictions: List of prediction summaries
        overall_validation: Overall framework validation
        statistics: Statistical analysis results
        visualizations: Paths to visualization files
    """
    predictions: List[PredictionSummary]
    overall_validation: Dict[str, any]
    statistics: Dict[str, float]
    visualizations: List[Path]


class PredictionsReportGenerator:
    """
    Predictions report generator.
    
    Compiles all predictions and generates comprehensive report.
    """
    
    def __init__(
        self,
        peak_validation: PeakValidation,
        invariance_result: InvarianceTestResult,
        subpeak_analysis: SubPeakAnalysis,
        phi_split_results: List[PhiSplitResult]
    ):
        """
        Initialize report generator.
        
        Args:
            peak_validation: High-l peak validation
            invariance_result: Frequency invariance results
            subpeak_analysis: Sub-peak analysis
            phi_split_results: φ-split results
        """
        self.peak_validation = peak_validation
        self.invariance_result = invariance_result
        self.subpeak_analysis = subpeak_analysis
        self.phi_split_results = phi_split_results
    
    def compile_predictions(self) -> List[PredictionSummary]:
        """
        Compile all predictions into summary list.
        
        Returns:
            List of PredictionSummary instances
        """
        pass
    
    def generate_report(
        self,
        output_path: Optional[Path] = None
    ) -> PredictionsReport:
        """
        Generate comprehensive predictions report.
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            PredictionsReport instance
        """
        pass
    
    def validate_framework(
        self,
        predictions: List[PredictionSummary]
    ) -> Dict[str, any]:
        """
        Validate theoretical framework based on predictions.
        
        Args:
            predictions: List of prediction summaries
            
        Returns:
            Dictionary with framework validation results
        """
        pass
    
    def create_visualizations(
        self,
        report: PredictionsReport,
        output_dir: Path
    ) -> List[Path]:
        """
        Create visualization plots for report.
        
        Args:
            report: PredictionsReport instance
            output_dir: Directory for output files
            
        Returns:
            List of paths to visualization files
        """
        pass
