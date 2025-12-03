# Step 1.1: Θ-Field Data Loader

**Phase:** 1 - Θ-Field Data Processing  
**Step:** 1.1 - Θ-Field Data Loader  
**Module:** `cmb/`

---

## Overview

This step implements data loading for Θ-field model data from `data/theta/` directory. It provides interfaces for loading frequency spectrum ρ_Θ(ω,t), temporal evolution data, and other Θ-field related data files.

---

## Input Data

- **Θ-Field Data Files** (from `data/theta/`):
  - Frequency spectrum data: ρ_Θ(ω,t)
  - Temporal evolution data: ω_min(t), ω_macro(t)
  - Node geometry data
  - Field configuration files

- **Data Index** (from [Step 0.3](../../phase_0_foundation/step_0.3_data_index/README.md)):
  - Location and metadata of theta data files

- **Configuration** (from [Step 0.1](../../phase_0_foundation/step_0.1_configuration/README.md)):
  - Data paths and constants

---

## Algorithm

### 1. Load Frequency Spectrum Data

- Load ρ_Θ(ω,t) from data files
- Parse frequency and time arrays
- Validate data format and ranges
- Handle interpolation if needed

### 2. Load Temporal Evolution Data

- Load ω_min(t) evolution data
- Load ω_macro(t) evolution data
- Parse time arrays
- Validate temporal coverage

### 3. Provide Data Interface

- Create data structures for frequency spectrum
- Create data structures for evolution
- Provide access methods for CMB calculations
- Handle data caching for performance

### 4. Data Validation

- Validate frequency ranges
- Validate time ranges
- Check data consistency
- Report data quality issues

---

## Output Data

### Files Created

1. **`cmb/theta_data_loader.py`** - Θ-field data loading module
   - `ThetaFrequencySpectrum` class
   - `ThetaEvolution` class
   - Loading functions

### Data Structures

- **Frequency Spectrum:**
  - Frequency array: ω
  - Time array: t
  - Spectrum values: ρ_Θ(ω,t)
  - Metadata (units, ranges, etc.)

- **Evolution Data:**
  - Time array: t
  - ω_min(t) values
  - ω_macro(t) values
  - Metadata

---

## Dependencies

- **Step 0.1** (Configuration) - For data paths
- **Step 0.2** (Utilities) - For data loading utilities
- **Step 0.3** (Data Index) - For locating theta data files

---

## Related Steps

- **Step 1.2** (Node Processing) - Will use frequency spectrum data
- **Step 1.3** (Evolution Data) - Will use evolution data
- **Step 2.1** (CMB Reconstruction) - Will use all theta data

---

## Tests

### Unit Tests

1. **Data Loading**
   - Test loading frequency spectrum files
   - Test loading evolution data files
   - Test error handling for missing files
   - Test data format validation

2. **Data Structures**
   - Test FrequencySpectrum class
   - Test Evolution class
   - Test data access methods
   - Test data validation

3. **Data Interpolation**
   - Test interpolation accuracy
   - Test edge cases (extrapolation)
   - Test performance with large datasets

### Integration Tests

1. **End-to-End Data Pipeline**
   - Load data → Validate → Provide interface → Use in calculations
   - Test with actual theta data files

---

## Implementation Notes

- Use efficient data structures (numpy arrays)
- Implement caching for frequently accessed data
- Provide clear error messages for data issues
- Support multiple data file formats if needed

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

