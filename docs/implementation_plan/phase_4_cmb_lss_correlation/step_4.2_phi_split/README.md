# Step 4.2: Phi-Split Analysis

**Phase:** 4 - CMB-LSS Correlation  
**Step:** 4.2 - Phi-Split Analysis  
**Module:** `cmb/correlation/`

---

## Overview

This step implements the φ-split analysis technique for CMB-LSS correlation. It analyzes signal enhancement through φ-split and validates theoretical predictions.

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

- Split CMB map by φ (azimuthal angle)
- Create φ-split sub-maps
- Handle full-sky coverage
- Account for split boundaries

### 2. Calculate Split Correlations

- Calculate correlation for each φ-split
- Compare split correlations
- Analyze correlation differences
- Handle statistical uncertainties

### 3. Analyze Signal Enhancement

- Measure signal enhancement in splits
- Compare with theoretical predictions
- Validate enhancement significance
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
   - Test map splitting by φ
   - Test split boundary handling
   - Test full-sky coverage

2. **Split Correlation**
   - Test correlation calculation per split
   - Test split comparison
   - Test statistical analysis

3. **Signal Enhancement**
   - Test enhancement measurement
   - Test significance calculation
   - Test prediction comparison

4. **Results Generation**
   - Test result creation
   - Test visualization
   - Test validation metrics

### Integration Tests

1. **End-to-End φ-Split Analysis**
   - Split map → Calculate correlations → Analyze enhancement → Validate
   - Test with actual CMB and LSS data

---

## Implementation Notes

- Handle φ-split boundaries carefully
- Use proper statistical methods for enhancement
- Provide clear visualization of splits
- Document enhancement patterns clearly

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

