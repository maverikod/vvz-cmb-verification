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
   - **What to test:**
     - Test ACT spectrum loading:
       - Verify load_observed_spectrum() loads ACT DR6.02 spectrum from tar.gz
       - Test spectrum data parsing (l, C_l, errors)
       - Verify spectrum format validation
     - Test covariance matrix loading:
       - Verify covariance matrix is loaded if available
       - Test covariance matrix format validation
     - Test binning handling:
       - Verify different multipole binning schemes are handled
       - Test binning conversion if needed
       - Verify binning consistency

2. **Spectrum Comparison**
   - **What to test:**
     - Test difference calculation:
       - Verify difference = reconstructed - observed is calculated correctly
       - Test difference statistics (mean_diff, rms_diff)
       - Verify difference is in correct units
     - Test χ² computation:
       - Verify calculate_chi_squared() computes χ² correctly
       - Test χ² with and without covariance matrix
       - Verify χ² formula: χ² = Σ (C_l_recon - C_l_obs)² / σ²
     - Test shape comparison:
       - Verify correlation coefficient calculation
       - Test shape correlation analysis
       - Verify shape matching

3. **High-l Validation**
   - **What to test:**
     - Test high-l tail comparison:
       - Verify validate_high_l_tail() compares l>2000 range correctly
       - Test tail_match boolean result
       - Verify shape_correlation calculation
       - Verify amplitude_ratio calculation
     - Test no Silk damping validation:
       - Verify no_silk_damping flag checks for absence of exp(-l²/l_silk²) behavior
       - Test that high-l tail continues without exponential cutoff
       - Verify tail follows power law (l⁻²–l⁻³)
     - Test tail shape analysis:
       - Verify tail shape matches theoretical prediction
       - Test shape consistency

4. **Report Generation**
   - **What to test:**
     - Test metric calculations:
       - Verify all comparison metrics are calculated:
         - chi_squared, correlation, mean_diff, rms_diff
         - tail_match, shape_correlation, amplitude_ratio, no_silk_damping
     - Test report generation:
       - Verify generate_comparison_report() creates comprehensive report
       - Test report includes all metrics
       - Verify report can be saved to file
     - Test visualization creation:
       - Verify comparison plots are created
       - Test spectrum overlay visualization
       - Verify plots are saved correctly

### Integration Tests

1. **End-to-End Comparison**
   - **What to test:**
     - Load spectra → Compare → Validate → Generate report:
       - Load reconstructed PowerSpectrum (from Step 3.1)
       - Load observed ACT DR6.02 spectrum
       - Compare spectra (difference, χ², correlation)
       - Validate high-l tail (l>2000)
       - Generate comparison report
       - Verify all comparison steps complete successfully
     - Test with actual ACT data:
       - Real ACT DR6.02 spectrum
       - Real reconstructed spectrum
       - Verify comparison metrics match expected values
       - Verify high-l tail validation passes (no Silk damping)

---

## Implementation Notes

- Handle different multipole binning schemes
- Use proper error propagation with covariance
- Provide detailed comparison metrics
- Create clear visualization outputs

## Forbidden Elements

**DO NOT USE:**
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)
- ❌ Silk damping formulas (no damping in Θ-model)
- ❌ Classical power spectrum models for comparison

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

