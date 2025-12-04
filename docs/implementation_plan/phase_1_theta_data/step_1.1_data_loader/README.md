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
   - **What to test:**
     - Test loading frequency spectrum files (ρ_Θ(ω,t)):
       - Verify correct parsing of frequency array ω
       - Verify correct parsing of time array t
       - Verify correct parsing of spectrum values ρ_Θ(ω,t) as 2D array
       - Check data shape consistency (frequencies × times = spectrum shape)
     - Test loading evolution data files (ω_min(t), ω_macro(t)):
       - Verify correct parsing of time array
       - Verify correct parsing of ω_min(t) values
       - Verify correct parsing of ω_macro(t) values
       - Check temporal coverage (time range completeness)
     - Test error handling for missing files:
       - FileNotFoundError for non-existent files
       - Appropriate error messages
     - Test data format validation:
       - CSV format validation
       - JSON format validation
       - Required columns/keys presence
       - Data type validation (numeric arrays)

2. **Data Structures**
   - **What to test:**
     - Test ThetaFrequencySpectrum class:
       - Verify all attributes (frequencies, times, spectrum, metadata)
       - Test data access methods
       - Verify data immutability (if applicable)
       - Test metadata handling
     - Test ThetaEvolution class:
       - Verify all attributes (times, omega_min, omega_macro, metadata)
       - Test data access methods
       - Verify temporal consistency (times match omega_min/omega_macro lengths)
     - Test data validation functions:
       - validate_frequency_spectrum() checks:
         - Non-empty arrays
         - Positive frequencies
         - Valid time range
         - Non-negative spectrum values
         - Consistent array shapes
       - validate_evolution_data() checks:
         - Non-empty arrays
         - Valid time range
         - Positive omega values
         - Consistent array lengths

3. **Data Interpolation**
   - **What to test:**
     - Test interpolation accuracy:
       - Interpolation at known data points (should return exact values)
       - Interpolation between data points (check smoothness)
       - Verify interpolation preserves frequency spectrum shape
     - Test edge cases:
       - Extrapolation beyond data range (should raise error or handle gracefully)
       - Empty data arrays
       - Single data point
     - Test performance with large datasets:
       - Memory usage with large frequency/time arrays
       - Interpolation speed for typical data sizes
       - Caching effectiveness (if implemented)

### Integration Tests

1. **End-to-End Data Pipeline**
   - **What to test:**
     - Load data → Validate → Provide interface → Use in calculations:
       - Load frequency spectrum from data index
       - Load evolution data from data index
       - Validate loaded data
       - Use data in CMB calculation (mock)
       - Verify data integrity throughout pipeline
     - Test with actual theta data files:
       - Real data file formats
       - Real data sizes and ranges
       - Verify data matches expected Θ-field model structure

---

## Implementation Notes

- Use efficient data structures (numpy arrays)
- Implement caching for frequently accessed data
- Provide clear error messages for data issues
- Support multiple data file formats if needed

## Forbidden Elements

**CRITICAL PRINCIPLE (from tech_spec-new.md):** Matter does NOT influence Θ-field.
Matter = projection/envelope of Θ-modes.

**DO NOT USE:**
- ❌ Matter as source of Θ-field (matter is projection, not source)
- ❌ Baryon density, DM density or material terms
- ❌ Modification of Θ-field based on observed matter
- ❌ T_μν of matter in Θ-field equations
- ❌ Reverse reaction of matter on Θ-field
- ❌ Potentials V(φ), V(Θ) in data processing
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)
- ❌ Classical cosmological evolution formulas

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

