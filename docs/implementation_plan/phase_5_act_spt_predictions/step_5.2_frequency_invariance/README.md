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
   - **What to test:**
     - Test loading maps at different frequencies:
       - Verify maps at 90, 150, 220, 350 GHz are loaded correctly
       - Test map format validation
       - Verify maps have same NSIDE and coverage
     - Test cross-spectra extraction:
       - Verify cross-spectra can be extracted from frequency maps
       - Test cross-spectrum format validation
     - Test frequency normalization:
       - Verify frequency-dependent effects are normalized correctly
       - Test normalization accuracy

2. **Cross-spectra Calculation**
   - **What to test:**
     - Test cross-spectrum computation:
       - Verify calculate_cross_spectra() computes cross-spectra between frequency pairs correctly
       - Test CrossSpectrum structure is created correctly:
         - frequency1, frequency2, multipoles, cross_spectrum, errors
       - Verify cross-spectrum calculation formula
     - Test frequency pair handling:
       - Verify all frequency pairs are processed (90-150, 90-220, 150-220, etc.)
       - Test pair combinations are correct
     - Test noise accounting:
       - Verify noise is accounted for in cross-spectrum calculation
       - Test error propagation

3. **Invariance Testing**
   - **What to test:**
     - Test frequency comparison:
       - Verify test_invariance() compares cross-spectra at different frequencies
       - Test that cross-spectra are consistent across frequencies
     - Test invariance metrics:
       - Verify calculate_invariance_metrics() calculates:
         - Frequency variation (should be minimal for achromatic CMB)
         - Consistency metrics
         - Invariance score
     - Test variation detection:
       - Verify frequency-dependent variations are detected
       - Test variation significance

4. **Validation**
   - **What to test:**
     - Test prediction comparison:
       - Verify validate_achromaticity() checks that CMB is frequency-invariant
       - Test validation against theoretical prediction (achromaticity)
     - Test validation criteria:
       - Verify validation_passed flag is set correctly
       - Test validation thresholds
     - Test report generation:
       - Verify InvarianceTestResult structure is created correctly
       - Test report includes all metrics and validation results

### Integration Tests

1. **End-to-End Invariance Test**
   - **What to test:**
     - Load multi-frequency data → Calculate cross-spectra → Test invariance → Validate:
       - Load CMB maps at multiple frequencies (90-350 GHz)
       - Calculate cross-spectra between frequency pairs
       - Test frequency invariance
       - Calculate invariance metrics
       - Validate achromaticity prediction
       - Generate InvarianceTestResult
       - Verify all steps complete successfully
     - Test with actual ACT multi-frequency data:
       - Real ACT multi-frequency maps
       - Verify cross-spectra are consistent across frequencies
       - Verify achromaticity validation passes

---

## Implementation Notes

- Handle frequency-dependent systematics carefully
- Account for instrumental frequency responses
- Use proper statistical methods for comparison
- Provide clear invariance documentation

## Forbidden Elements

**DO NOT USE:**
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)
- ❌ Frequency-dependent models (achromaticity is fundamental property)

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

