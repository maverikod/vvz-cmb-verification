# Step 5.2: Frequency Invariance Test

**Phase:** 5 - ACT/SPT Predictions  
**Step:** 5.2 - Frequency Invariance Test  
**Module:** `cmb/predictions/`

---

## Overview

This step tests the frequency invariance (achromaticity) of CMB microstructure. It tests cross-spectra at 90-350 GHz and validates that the structure is frequency-independent, as predicted by the Θ-field model.

---

## Input Data

- **Reconstructed CMB Maps** (from [Step 2.1](../../phase_2_cmb_reconstruction/step_2.1_reconstruction_core/README.md)):
  - CMB maps (should be frequency-independent)

- **Multi-frequency ACT Data** (from `data/in/dr6.02/`):
  - ACT maps at different frequencies (150 GHz, 220 GHz)
  - Cross-spectra data

- **Configuration** (from [Step 0.1](../../phase_0_foundation/step_0.1_configuration/README.md)):
  - Frequency ranges (90-350 GHz)

---

## Algorithm

### 1. Load Multi-frequency Data

- Load ACT maps at different frequencies
- Extract cross-spectra data
- Handle frequency-dependent systematics
- Normalize frequency responses

### 2. Calculate Cross-spectra

- Calculate cross-spectra between frequencies
- Compare 90-150 GHz, 150-220 GHz, etc.
- Handle frequency-dependent noise
- Account for instrumental effects

### 3. Test Frequency Invariance

- Compare cross-spectra across frequencies
- Test for frequency-dependent variations
- Validate achromaticity
- Calculate invariance metrics

### 4. Validate Predictions

- Compare with theoretical predictions
- Validate frequency independence
- Document invariance results
- Generate validation report

---

## Output Data

### Files Created

1. **`cmb/predictions/frequency_invariance.py`** - Frequency invariance test module
   - `FrequencyInvarianceTester` class
   - Test functions

### Test Results

- **Cross-spectra Data:**
  - Frequency pairs
  - Cross-spectrum values
  - Comparison metrics

- **Invariance Validation:**
  - Invariance metrics
  - Frequency dependence analysis
  - Validation results
  - Comparison with predictions

---

## Dependencies

- **Step 0.2** (Utilities) - For data loading
- **Step 2.1** (CMB Reconstruction) - For reconstructed maps

---

## Related Steps

- **Step 5.3** (Predictions Report) - Will include invariance results

---

## Tests

### Unit Tests

1. **Multi-frequency Loading**
   - Test loading maps at different frequencies
   - Test cross-spectra extraction
   - Test frequency normalization

2. **Cross-spectra Calculation**
   - Test cross-spectrum computation
   - Test frequency pair handling
   - Test noise accounting

3. **Invariance Testing**
   - Test frequency comparison
   - Test invariance metrics
   - Test variation detection

4. **Validation**
   - Test prediction comparison
   - Test validation criteria
   - Test report generation

### Integration Tests

1. **End-to-End Invariance Test**
   - Load multi-frequency data → Calculate cross-spectra → Test invariance → Validate
   - Test with actual ACT multi-frequency data

---

## Implementation Notes

- Handle frequency-dependent systematics carefully
- Account for instrumental frequency responses
- Use proper statistical methods for comparison
- Provide clear invariance documentation

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

