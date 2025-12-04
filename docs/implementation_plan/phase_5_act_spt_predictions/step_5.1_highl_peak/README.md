# Step 5.1: High-l Peak Prediction

**Phase:** 5 - ACT/SPT Predictions  
**Step:** 5.1 - High-l Peak Prediction  
**Module:** `cmb/predictions/`

---

## Overview

This step predicts and validates the high-l peak at l≈4500-6000 as specified in the theoretical framework. It calculates predicted peak position and amplitude, and compares with ACT/SPT observational data.

---

## Input Data

- **Power Spectrum** (from [Step 3.1](../../phase_3_power_spectrum/step_3.1_spectrum_calculation/README.md)):
  - C_l power spectrum

- **Sub-peaks Analysis** (from [Step 3.2](../../phase_3_power_spectrum/step_3.2_subpeaks_analysis/README.md)):
  - Sub-peak analysis results

- **Θ-Field Evolution** (from [Step 1.3](../../phase_1_theta_data/step_1.3_evolution_data/README.md)):
  - Evolution data for peak prediction

- **ACT/SPT Data** (from `data/in/dr6.02/`):
  - Observational spectra for comparison

---

## Algorithm

### 1. Calculate Predicted Peak Position

- Use evolution data to predict peak position
- Calculate expected l value (≈4500-6000)
- Account for frequency evolution
- Handle theoretical uncertainties

### 2. Estimate Peak Amplitude

- Calculate predicted peak amplitude
- Use power spectrum model
- Account for sub-peak contributions
- Handle amplitude uncertainties

### 3. Compare with Observations

- Load ACT/SPT high-l data
- Compare predicted vs observed peak
- Calculate agreement metrics
- Validate peak detection

### 4. Generate Prediction Report

- Document predicted peak properties
- Compare with observations
- Provide validation results
- Create visualization

---

## Output Data

### Files Created

1. **`cmb/predictions/highl_peak_predictor.py`** - High-l peak prediction module
   - `HighlPeakPredictor` class
   - Prediction functions

### Prediction Results

- **Peak Prediction:**
  - Predicted l position
  - Predicted amplitude
  - Prediction uncertainties
  - Theoretical basis

- **Validation Results:**
  - Observed peak position
  - Observed amplitude
  - Agreement metrics
  - Validation status

---

## Dependencies

- **Step 1.3** (Evolution Data) - For peak prediction
- **Step 3.1** (Power Spectrum) - For spectrum data
- **Step 3.2** (Sub-peaks Analysis) - For sub-peak information

---

## Related Steps

- **Step 5.3** (Predictions Report) - Will compile all predictions

---

## Tests

### Unit Tests

1. **Peak Position Prediction**
   - **What to test:**
     - Test position calculation:
       - Verify predict_peak() calculates predicted l position in range 4500-6000
       - Test that evolution data (ω_min(t), ω_macro(t)) is used correctly
       - Verify prediction uses Θ-model formulas (not classical models)
     - Test evolution data usage:
       - Verify ThetaEvolutionProcessor is used to get evolution values
       - Test evolution interpolation accuracy
     - Test uncertainty handling:
       - Verify l_position_uncertainty is calculated correctly
       - Test uncertainty propagation

2. **Amplitude Estimation**
   - **What to test:**
     - Test amplitude calculation:
       - Verify predicted peak amplitude is calculated correctly
       - Test that power spectrum model is used
       - Verify sub-peak contributions are accounted for
     - Test model application:
       - Verify Θ-model formulas are used (not classical models)
       - Test model consistency
     - Test uncertainty propagation:
       - Verify amplitude_uncertainty is calculated correctly
       - Test uncertainty sources are accounted for

3. **Observation Comparison**
   - **What to test:**
     - Test ACT/SPT data loading:
       - Verify compare_with_observations() loads observed spectrum correctly
       - Test spectrum data parsing
     - Test peak comparison:
       - Verify predicted vs observed peak position comparison
       - Verify predicted vs observed peak amplitude comparison
       - Test peak detection in observed spectrum
     - Test agreement metrics:
       - Verify calculate_agreement_metrics() calculates:
         - Position agreement (difference, significance)
         - Amplitude agreement (ratio, significance)
         - Overall agreement score

4. **Report Generation**
   - **What to test:**
     - Test report creation:
       - Verify PeakValidation structure is created correctly:
         - predicted, observed_position, observed_amplitude
         - agreement, validation_passed
     - Test visualization:
       - Verify peak prediction can be visualized on spectrum plot
       - Test visualization shows predicted vs observed clearly
     - Test validation documentation:
       - Verify validation results are documented correctly
       - Test documentation completeness

### Integration Tests

1. **End-to-End Peak Prediction**
   - **What to test:**
     - Predict peak → Compare with observations → Validate → Report:
       - Load PowerSpectrum and SubPeakAnalysis
       - Load ThetaEvolutionProcessor
       - Predict high-l peak position and amplitude (l≈4500-6000)
       - Load ACT/SPT observed spectrum
       - Compare predicted vs observed peak
       - Calculate agreement metrics
       - Generate PeakValidation report
       - Verify all steps complete successfully
     - Test with actual observational data:
       - Real ACT/SPT high-l spectrum data
       - Real power spectrum and sub-peak analysis
       - Verify peak prediction matches observations
       - Verify validation passes if peak is found

---

## Implementation Notes

- Use theoretical framework for predictions
- Handle observational uncertainties properly
- Provide clear validation criteria
- Document prediction methodology

## Forbidden Elements

**DO NOT USE:**
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)
- ❌ Classical peak prediction models (use Θ-model formulas)

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

