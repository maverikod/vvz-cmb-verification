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
   - Test position calculation
   - Test evolution data usage
   - Test uncertainty handling

2. **Amplitude Estimation**
   - Test amplitude calculation
   - Test model application
   - Test uncertainty propagation

3. **Observation Comparison**
   - Test ACT/SPT data loading
   - Test peak comparison
   - Test agreement metrics

4. **Report Generation**
   - Test report creation
   - Test visualization
   - Test validation documentation

### Integration Tests

1. **End-to-End Peak Prediction**
   - Predict peak → Compare with observations → Validate → Report
   - Test with actual observational data

---

## Implementation Notes

- Use theoretical framework for predictions
- Handle observational uncertainties properly
- Provide clear validation criteria
- Document prediction methodology

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

