# Step 3.2: High-l Sub-peaks Analysis

**Phase:** 3 - Power Spectrum Generation  
**Step:** 3.2 - High-l Sub-peaks Analysis  
**Module:** `cmb/spectrum/`

---

## Overview

This step analyzes high-l sub-peaks in the power spectrum. It identifies sub-peaks from ω_min beatings, detects the predicted l≈4500-6000 peak, and analyzes sub-structure.

---

## Input Data

- **Power Spectrum** (from [Step 3.1](../step_3.1_spectrum_calculation/README.md)):
  - C_l power spectrum from `PowerSpectrumCalculator`

- **Θ-Field Evolution** (from [Step 1.3](../../phase_1_theta_data/step_1.3_evolution_data/README.md)):
  - ω_min(t) evolution data for beatings analysis

- **Configuration** (from [Step 0.1](../../phase_0_foundation/step_0.1_configuration/README.md)):
  - Expected peak range (l≈4500-6000)

---

## Algorithm

### 1. Identify Sub-peaks from Beatings

- Analyze power spectrum for sub-peak structures
- Calculate beatings from ω_min(t) evolution
- Match beatings to sub-peak positions
- Validate beating-subpeak correspondence

### 2. Detect High-l Peak

- Search for peak in l≈4500-6000 range
- Identify peak position and amplitude
- Calculate peak significance
- Compare with predictions

### 3. Analyze Sub-structure

- Identify all sub-peaks in high-l range
- Calculate sub-peak properties (position, amplitude, width)
- Analyze sub-peak spacing
- Characterize sub-structure pattern

### 4. Generate Sub-peak Catalog

- Create catalog of identified sub-peaks
- Document peak properties
- Provide analysis results
- Create visualization

---

## Output Data

### Files Created

1. **`cmb/spectrum/subpeaks_analyzer.py`** - Sub-peaks analysis module
   - `SubpeaksAnalyzer` class
   - Analysis functions

### Analysis Results

- **Sub-peak Catalog:**
  - Peak positions (l values)
  - Peak amplitudes
  - Peak widths
  - Beating correspondences

- **High-l Peak Detection:**
  - Peak position (l value)
  - Peak amplitude
  - Significance level
  - Comparison with predictions

---

## Dependencies

- **Step 1.3** (Evolution Data) - For ω_min(t) beatings
- **Step 3.1** (Power Spectrum) - For C_l data

---

## Related Steps

- **Step 3.3** (Spectrum Comparison) - Will use sub-peak analysis
- **Step 5.1** (High-l Peak Prediction) - Will validate predictions

---

## Tests

### Unit Tests

1. **Beating Analysis**
   - Test ω_min(t) beating calculation
   - Test beating-to-subpeak matching
   - Test correspondence validation

2. **Peak Detection**
   - Test peak finding algorithms
   - Test l≈4500-6000 range search
   - Test peak significance calculation

3. **Sub-structure Analysis**
   - Test sub-peak identification
   - Test property calculation
   - Test pattern analysis

4. **Catalog Generation**
   - Test catalog creation
   - Test data integrity
   - Test visualization

### Integration Tests

1. **End-to-End Sub-peak Analysis**
   - Load spectrum → Analyze beatings → Detect peaks → Generate catalog
   - Test with actual power spectrum data

---

## Implementation Notes

- Use robust peak-finding algorithms
- Handle noise in power spectrum
- Provide clear peak identification criteria
- Support multiple peak detection methods

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

