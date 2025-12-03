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
   - Test LSS data loading
   - Test catalog format handling
   - Test correlation data extraction

2. **Correlation Calculation**
   - Test correlation function computation
   - Test scale handling (10-12 Mpc)
   - Test angular separation calculations

3. **Correlation Tests**
   - Test significance calculation
   - Test correlation coefficient computation
   - Test statistical analysis

4. **Map Generation**
   - Test correlation map creation
   - Test visualization
   - Test data integrity

### Integration Tests

1. **End-to-End Correlation Analysis**
   - Load data → Calculate correlation → Test → Generate maps
   - Test with actual CMB and LSS data

---

## Implementation Notes

- Handle different LSS survey geometries
- Account for selection effects
- Use proper statistical methods
- Provide detailed correlation documentation

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

