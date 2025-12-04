# Step 4.2: Phi-Split Analysis

**Phase:** 4 - CMB-LSS Correlation  
**Step:** 4.2 - Phi-Split Analysis  
**Module:** `cmb/correlation/`

---

## Overview

This step implements the φ-split analysis technique for CMB-LSS correlation. 
It analyzes signal enhancement through φ-split and validates theoretical predictions.

**φ-split principle:**
Signal enhancement occurs due to m-modes (azimuthal modes) in spherical harmonics.
Splitting by φ angle reveals enhanced correlation in specific directions,
which is a prediction of Θ-model (JW-CMB2.8).

---

## Input Data

- **Correlation Results** (from [Step 4.1](../step_4.1_correlation_core/README.md)):
  - CMB-LSS correlation functions from `CmbLssCorrelator`

- **CMB Map** (from [Step 2.1](../../phase_2_cmb_reconstruction/step_2.1_reconstruction_core/README.md)):
  - Reconstructed CMB map

- **LSS Data** (from `data/in/`):
  - LSS structure data

---

## Algorithm

### 1. Implement φ-Split Technique

- Split CMB map by φ (azimuthal angle) at specified φ_split value
- Create two sub-maps: positive φ and negative φ regions
- Handle full-sky coverage (0 ≤ φ < 2π)
- Account for split boundaries (φ = 0 and φ = 2π are equivalent)
- Split reveals m-mode structure in spherical harmonics

### 2. Calculate Split Correlations

- Calculate correlation for each φ-split
- Compare split correlations
- Analyze correlation differences
- Handle statistical uncertainties

### 3. Analyze Signal Enhancement

- Measure signal enhancement factor: enhancement = correlation_positive / correlation_negative
- Enhancement occurs due to m-modes in Θ-field structure
- Compare with theoretical predictions (JW-CMB2.8)
- Validate enhancement significance (statistical test)
- Document enhancement patterns

### 4. Generate φ-Split Results

- Create φ-split analysis results
- Generate visualization
- Document enhancement findings
- Provide validation metrics

---

## Output Data

### Files Created

1. **`cmb/correlation/phi_split_analyzer.py`** - φ-split analysis module
   - `PhiSplitAnalyzer` class
   - Analysis functions

### φ-Split Results

- **Split Analysis:**
  - φ-split configurations
  - Correlation values per split
  - Signal enhancement measurements
  - Enhancement significance

- **Validation Results:**
  - Comparison with predictions
  - Enhancement validation
  - Statistical analysis

---

## Dependencies

- **Step 4.1** (Correlation Core) - For correlation functions

---

## Related Steps

- **Step 5.3** (Predictions Report) - Will use φ-split results

---

## Tests

### Unit Tests

1. **φ-Split Implementation**
   - **What to test:**
     - Test map splitting by φ:
       - Verify split_by_phi() splits CMB map at specified φ angle correctly
       - Test positive_φ_map and negative_φ_map are created correctly
       - Verify split preserves full-sky coverage
     - Test split boundary handling:
       - Verify φ = 0 and φ = 2π are treated as equivalent
       - Test boundary cases are handled correctly
     - Test full-sky coverage:
       - Verify all pixels are assigned to either positive or negative split
       - Test no pixels are lost in split

2. **Split Correlation**
   - **What to test:**
     - Test correlation calculation per split:
       - Verify correlation is calculated for positive_φ_map
       - Verify correlation is calculated for negative_φ_map
       - Test correlation values are in valid range [-1, 1]
     - Test split comparison:
       - Verify correlation_positive and correlation_negative are compared correctly
       - Test correlation differences are calculated
     - Test statistical analysis:
       - Verify statistical significance of correlation differences
       - Test significance calculation

3. **Signal Enhancement**
   - **What to test:**
     - Test enhancement measurement:
       - Verify measure_signal_enhancement() calculates: enhancement = correlation_positive / correlation_negative
       - Test enhancement factor is positive
       - Verify enhancement is calculated relative to baseline_correlation
     - Test significance calculation:
       - Verify enhancement significance is calculated correctly
       - Test significance threshold handling
     - Test prediction comparison:
       - Verify validate_enhancement() compares measured vs predicted enhancement
       - Test validation against JW-CMB2.8 predictions
       - Verify validation_passed flag

4. **Results Generation**
   - **What to test:**
     - Test result creation:
       - Verify PhiSplitResult structure is created correctly:
         - split_phi, correlation_positive, correlation_negative
         - enhancement, significance
     - Test visualization:
       - Verify φ-split can be visualized
       - Test visualization shows enhancement clearly
     - Test validation metrics:
       - Verify all validation metrics are calculated
       - Test metrics are in expected ranges

### Integration Tests

1. **End-to-End φ-Split Analysis**
   - **What to test:**
     - Split map → Calculate correlations → Analyze enhancement → Validate:
       - Load reconstructed CMB map
       - Load CMB-LSS correlator
       - Split map by φ angle
       - Calculate correlations for each split
       - Measure signal enhancement
       - Validate enhancement against predictions (JW-CMB2.8)
       - Generate φ-split results
       - Verify all steps complete successfully
     - Test with actual CMB and LSS data:
       - Real reconstructed CMB map
       - Real LSS correlation data
       - Verify enhancement is measured correctly
       - Verify enhancement matches theoretical predictions

---

## Implementation Notes

- Handle φ-split boundaries carefully (φ = 0 and φ = 2π are equivalent)
- Use proper statistical methods for enhancement
- Provide clear visualization of splits
- Document enhancement patterns clearly

## Forbidden Elements

**DO NOT USE:**
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)
- ❌ Classical correlation formulas without m-mode consideration

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

