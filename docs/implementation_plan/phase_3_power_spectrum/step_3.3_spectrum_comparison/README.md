# Step 3.3: Spectrum Comparison

**Phase:** 3 - Power Spectrum Generation  
**Step:** 3.3 - Spectrum Comparison  
**Module:** `cmb/spectrum/`

---

## Overview

This step compares the reconstructed power spectrum with ACT DR6.02 observational data. It validates the high-l tail (l>2000), compares spectral shapes, and generates comparison reports.

---

## Input Data

- **Reconstructed Power Spectrum** (from [Step 3.1](../step_3.1_spectrum_calculation/README.md)):
  - C_l power spectrum from `PowerSpectrumCalculator`

- **ACT DR6.02 Spectra** (from `data/in/dr6.02/`):
  - `act_dr6.02_spectra_and_cov_binning_20.tar.gz`
  - `act_dr6.02_spectra_and_cov_binning_50.tar.gz`
  - Covariance matrices

- **Utilities** (from [Step 0.2](../../phase_0_foundation/step_0.2_utilities/README.md)):
  - Data loader for ACT spectra
  - Visualization utilities

---

## Algorithm

### 1. Load Observational Spectra

- Load ACT DR6.02 power spectra from archives
- Extract C_l values and error bars
- Load covariance matrices
- Handle different binning schemes

### 2. Compare Spectra

- Calculate difference between reconstructed and observed
- Compute χ² statistic
- Analyze spectral shape differences
- Handle multipole range matching

### 3. Validate High-l Tail

- Compare high-l range (l>2000)
- Validate no Silk damping in reconstructed
- Check high-l tail shape
- Analyze tail amplitude

### 4. Generate Comparison Report

- Create statistical comparison metrics
- Generate comparison plots
- Document discrepancies
- Provide validation results

---

## Output Data

### Files Created

1. **`cmb/spectrum/spectrum_comparator.py`** - Spectrum comparison module
   - `SpectrumComparator` class
   - Comparison functions

### Comparison Results

- **Comparison Statistics:**
  - χ² value
  - Difference statistics
  - Correlation coefficient
  - High-l tail validation

- **Comparison Report:**
  - Text report with metrics
  - Visualization plots
  - Discrepancy analysis

---

## Dependencies

- **Step 0.2** (Utilities) - For data loading and visualization
- **Step 3.1** (Power Spectrum) - For reconstructed spectrum

---

## Related Steps

- **Step 5.1** (High-l Peak Prediction) - Will use comparison for validation

---

## Tests

### Unit Tests

1. **Spectrum Loading**
   - Test ACT spectrum loading
   - Test covariance matrix loading
   - Test binning handling

2. **Spectrum Comparison**
   - Test difference calculation
   - Test χ² computation
   - Test shape comparison

3. **High-l Validation**
   - Test high-l tail comparison
   - Test no Silk damping validation
   - Test tail shape analysis

4. **Report Generation**
   - Test metric calculations
   - Test report generation
   - Test visualization creation

### Integration Tests

1. **End-to-End Comparison**
   - Load spectra → Compare → Validate → Generate report
   - Test with actual ACT data

---

## Implementation Notes

- Handle different multipole binning schemes
- Use proper error propagation with covariance
- Provide detailed comparison metrics
- Create clear visualization outputs

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

