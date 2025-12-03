# Step 2.2: CMB Map Validation

**Phase:** 2 - CMB Map Reconstruction  
**Step:** 2.2 - CMB Map Validation  
**Module:** `cmb/reconstruction/`

---

## Overview

This step validates reconstructed CMB maps by comparing them with ACT DR6.02 observational data. It checks arcmin-scale structures (2-5′), validates amplitude (20-30 μK), and performs statistical comparisons.

---

## Input Data

- **Reconstructed CMB Map** (from [Step 2.1](../step_2.1_reconstruction_core/README.md)):
  - HEALPix map from `CmbMapReconstructor`
  - Temperature fluctuations ΔT

- **ACT DR6.02 Maps** (from `data/in/dr6.02/`):
  - `act_dr6.02_std_AA_day_pa4_f150_4way_coadd_map_healpix.fits`
  - Other ACT maps for comparison

- **Utilities** (from [Step 0.2](../../phase_0_foundation/step_0.2_utilities/README.md)):
  - Data loader for ACT maps
  - Visualization utilities

---

## Algorithm

### 1. Load Observational Data

- Load ACT DR6.02 maps from FITS files
- Extract temperature maps
- Handle map resolution differences
- Normalize map formats

### 2. Compare Maps

- Calculate difference maps
- Compute correlation coefficients
- Analyze spatial correlation
- Handle resolution matching

### 3. Validate Arcmin-Scale Structures

- Extract structures at 2-5′ scales
- Compare structure positions
- Validate structure amplitudes
- Check structure counts

### 4. Validate Amplitude

- Check temperature fluctuation range (20-30 μK)
- Compare amplitude distributions
- Validate statistical properties
- Check for systematic offsets

### 5. Generate Validation Report

- Create comparison statistics
- Generate visualization plots
- Document discrepancies
- Provide validation metrics

---

## Output Data

### Files Created

1. **`cmb/reconstruction/map_validator.py`** - Map validation module
   - `MapValidator` class
   - Validation functions

### Validation Results

- **Comparison Statistics:**
  - Correlation coefficients
  - Difference map statistics
  - Structure matching results
  - Amplitude validation results

- **Validation Report:**
  - Text report with metrics
  - Visualization plots
  - Discrepancy analysis

---

## Dependencies

- **Step 0.2** (Utilities) - For data loading and visualization
- **Step 2.1** (CMB Reconstruction) - For reconstructed maps

---

## Related Steps

- **Step 2.3** (Node Mapping) - Will use validated maps
- **Step 3.1** (Power Spectrum) - Will use validated maps for spectrum calculation

---

## Tests

### Unit Tests

1. **Map Loading**
   - Test ACT map loading
   - Test resolution handling
   - Test format normalization

2. **Map Comparison**
   - Test difference map calculation
   - Test correlation computation
   - Test spatial correlation analysis

3. **Structure Validation**
   - Test arcmin-scale extraction
   - Test structure matching
   - Test amplitude validation

4. **Validation Metrics**
   - Test metric calculations
   - Test report generation
   - Test visualization creation

### Integration Tests

1. **End-to-End Validation**
   - Load maps → Compare → Validate → Generate report
   - Test with actual ACT data

---

## Implementation Notes

- Handle different NSIDE values between maps
- Use efficient map comparison algorithms
- Provide detailed validation metrics
- Create clear visualization outputs

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

