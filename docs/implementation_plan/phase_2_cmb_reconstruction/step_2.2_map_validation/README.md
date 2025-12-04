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
   - **What to test:**
     - Test ACT map loading:
       - Verify ACT DR6.02 map is loaded correctly from FITS
       - Test NSIDE parameter extraction
       - Verify map data type and units (μK)
     - Test resolution handling:
       - Verify maps with different NSIDE are handled correctly
       - Test resolution matching between reconstructed and observed maps
       - Test resolution conversion if needed
     - Test format normalization:
       - Verify maps are normalized to same format
       - Test unit conversion if needed
       - Verify coordinate system consistency

2. **Map Comparison**
   - **What to test:**
     - Test difference map calculation:
       - Verify difference = reconstructed - observed
       - Test difference map statistics (mean, std, RMS)
       - Verify difference map is in correct units (μK)
     - Test correlation computation:
       - Verify correlation coefficient calculation (Pearson correlation)
       - Test correlation is between -1 and 1
       - Verify correlation significance calculation
     - Test spatial correlation analysis:
       - Test correlation at different angular scales
       - Verify correlation map creation
       - Test correlation pattern analysis

3. **Structure Validation**
   - **What to test:**
     - Test arcmin-scale extraction:
       - Verify structures at 2-5′ scales are identified
       - Test structure detection algorithm
       - Verify structure positions match between maps
     - Test structure matching:
       - Verify position_matches counts structures at same positions
       - Test matching tolerance (angular separation)
       - Verify structure identification accuracy
     - Test amplitude validation:
       - Verify validate_amplitude() checks 20-30 μK range
       - Test mean_amplitude and std_amplitude calculations
       - Verify in_range_fraction calculation
       - Test validation_passed flag

4. **Validation Metrics**
   - **What to test:**
     - Test metric calculations:
       - Verify all metrics are calculated correctly:
         - correlation, mean_diff, std_diff, rms_diff
         - structure_count, position_matches, amplitude_matches
         - mean_amplitude, std_amplitude, in_range_fraction
     - Test report generation:
       - Verify generate_validation_report() creates comprehensive report
       - Test report includes all validation metrics
       - Verify report can be saved to file
     - Test visualization creation:
       - Verify comparison plots are created
       - Test difference map visualization
       - Verify plots are saved correctly

### Integration Tests

1. **End-to-End Validation**
   - **What to test:**
     - Load maps → Compare → Validate → Generate report:
       - Load reconstructed CMB map
       - Load observed ACT DR6.02 map
       - Compare maps (correlation, difference)
       - Validate arcmin structures (2-5′)
       - Validate amplitude (20-30 μK)
       - Generate validation report
       - Verify all validation steps complete successfully
     - Test with actual ACT data:
       - Real ACT DR6.02 maps
       - Verify validation metrics match expected values
       - Verify validation report is comprehensive and accurate

---

## Implementation Notes

- Handle different NSIDE values between maps
- Use efficient map comparison algorithms
- Provide detailed validation metrics
- Create clear visualization outputs

## Forbidden Elements

**CRITICAL PRINCIPLE (from tech_spec-new.md):** Matter does NOT influence Θ-field.
CMB is projection of Θ-nodes, not plasma effects.

**DO NOT USE:**
- ❌ Matter as source of CMB (CMB is projection of Θ-nodes)
- ❌ Baryon density, DM density or material terms
- ❌ Plasma terms or sound horizons
- ❌ Baryon acoustic oscillations as physical mechanism
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)
- ❌ Classical cosmological models for validation (use Θ-model predictions)

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

