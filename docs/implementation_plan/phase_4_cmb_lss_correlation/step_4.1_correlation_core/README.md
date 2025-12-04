# Step 4.1: Correlation Analysis Core

**Phase:** 4 - CMB-LSS Correlation  
**Step:** 4.1 - Correlation Analysis Core  
**Module:** `cmb/correlation/`

---

## Overview

This step implements the core CMB-LSS correlation analysis. It calculates correlation functions between CMB maps and Large Scale Structure data, handles 10-12 Mpc scales, and performs correlation tests.

---

## Input Data

- **Reconstructed CMB Map** (from [Step 2.1](../../phase_2_cmb_reconstruction/step_2.1_reconstruction_core/README.md)):
  - HEALPix CMB map

- **LSS Data** (from `data/in/`):
  - `cells_LSS_correlation_pack_v1-1.zip` - LSS correlation data
  - SDSS or other LSS catalogs
  - Correlation function data

- **Utilities** (from [Step 0.2](../../phase_0_foundation/step_0.2_utilities/README.md)):
  - Data loader for LSS data

---

## Algorithm

### 1. Load LSS Data

- Load LSS correlation data from archives
- Extract LSS structure positions
- Load correlation function data
- Handle LSS catalog formats

### 2. Calculate Correlation Functions

- Compute cross-correlation between CMB and LSS
- Calculate correlation at 10-12 Mpc scales
- Handle angular separation calculations
- Account for survey geometry

### 3. Perform Correlation Tests

- Test CMB ↔ LSS correlation significance
- Calculate correlation coefficients
- Analyze correlation scale dependence
- Handle statistical uncertainties

### 4. Generate Correlation Maps

- Create correlation maps
- Visualize correlation patterns
- Identify correlated regions
- Document correlation structure

---

## Output Data

### Files Created

1. **`cmb/correlation/cmb_lss_correlator.py`** - Correlation analysis module
   - `CmbLssCorrelator` class
   - Correlation functions

### Correlation Results

- **Correlation Functions:**
  - Angular separation array
  - Correlation values
  - Error bars
  - Scale-dependent correlation

- **Correlation Statistics:**
  - Correlation coefficients
  - Significance levels
  - Scale analysis results

---

## Dependencies

- **Step 0.2** (Utilities) - For data loading
- **Step 2.1** (CMB Reconstruction) - For CMB map

---

## Related Steps

- **Step 4.2** (Phi-Split Analysis) - Will use correlation results
- **Step 4.3** (Node-LSS Mapping) - Will use correlation for mapping

---

## Tests

### Unit Tests

1. **LSS Data Loading**
   - **What to test:**
     - Test LSS data loading:
       - Verify load_lss_data() loads LSS correlation data correctly
       - Test loading from tar.gz archives (cells_LSS_correlation_pack_v1-1.zip)
       - Test SDSS catalog loading
       - Verify data format validation
     - Test catalog format handling:
       - Verify different LSS catalog formats are handled
       - Test format conversion if needed
     - Test correlation data extraction:
       - Verify LSS structure positions are extracted correctly
       - Test correlation function data extraction

2. **Correlation Calculation**
   - **What to test:**
     - Test correlation function computation:
       - Verify calculate_correlation_function() computes cross-correlation correctly
       - Test correlation at 10-12 Mpc scales
       - Verify CorrelationFunction structure is created correctly:
         - angular_separation, correlation, errors, scale_mpc
     - Test scale handling (10-12 Mpc):
       - Verify scale_min_mpc and scale_max_mpc parameters are used correctly
       - Test scale conversion (Mpc to angular separation)
     - Test angular separation calculations:
       - Verify angular separation is calculated correctly
       - Test separation range handling

3. **Correlation Tests**
   - **What to test:**
     - Test significance calculation:
       - Verify test_correlation_significance() calculates:
         - correlation_coefficient (Pearson correlation)
         - p_value (statistical significance)
         - significance_sigma (significance in sigma)
     - Test correlation coefficient computation:
       - Verify Pearson correlation is calculated correctly
       - Test correlation is between -1 and 1
     - Test statistical analysis:
       - Verify analyze_scale_dependence() analyzes correlation scale dependence
       - Test scale_slope, peak_scale, scale_range calculations

4. **Map Generation**
   - **What to test:**
     - Test correlation map creation:
       - Verify create_correlation_map() projects correlation function onto HEALPix map
       - Test map has same NSIDE as input CMB map
       - Verify correlation values are correctly mapped
     - Test visualization:
       - Verify correlation maps can be visualized
       - Test visualization accuracy
     - Test data integrity:
       - Verify correlation map data is valid
       - Test map format consistency

### Integration Tests

1. **End-to-End Correlation Analysis**
   - **What to test:**
     - Load data → Calculate correlation → Test → Generate maps:
       - Load reconstructed CMB map
       - Load LSS correlation data
       - Calculate correlation function at 10-12 Mpc scales
       - Test correlation significance
       - Analyze scale dependence
       - Create correlation map
       - Verify all steps complete successfully
     - Test with actual CMB and LSS data:
       - Real reconstructed CMB map
       - Real LSS data (SDSS, etc.)
       - Verify correlation is calculated correctly
       - Verify correlation matches theoretical predictions

---

## Implementation Notes

- Handle different LSS survey geometries
- Account for selection effects
- Use proper statistical methods
- Provide detailed correlation documentation

## Forbidden Elements

**DO NOT USE:**
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)
- ❌ Classical correlation formulas without Θ-model basis

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

