# Step 1.3: Θ-Field Evolution Data

**Phase:** 1 - Θ-Field Data Processing  
**Step:** 1.3 - Θ-Field Evolution Data  
**Module:** `cmb/`

---

## Overview

This step processes Θ-field temporal evolution data from `data/theta/evolution/` directory. It handles ω_min(t) and ω_macro(t) evolution data and provides time-dependent parameter processing.

---

## Input Data

- **Θ-Field Evolution Data** (from `data/theta/evolution/`):
  - ω_min(t) evolution data
  - ω_macro(t) evolution data
  - Time arrays
  - Evolution metadata

- **Θ-Field Data Loader** (from [Step 1.1](../step_1.1_data_loader/README.md)):
  - Evolution data loaded via `ThetaEvolution` class

- **Configuration** (from [Step 0.1](../../phase_0_foundation/step_0.1_configuration/README.md)):
  - Time range parameters
  - Evolution constants

---

## Algorithm

### 1. Load Evolution Data

- Load ω_min(t) data from files
- Load ω_macro(t) data from files
- Parse time arrays
- Validate temporal coverage

### 2. Process Time-Dependent Parameters

- Interpolate evolution data if needed
- Calculate evolution rates
- Handle time range validation
- Process evolution statistics

### 3. Provide Evolution Interface

- Create evolution data structures
- Provide access methods for time queries
- Handle interpolation for arbitrary times
- Support evolution calculations

### 4. Validate Evolution Data

- Check time range coverage
- Validate frequency ranges
- Check data consistency
- Report data quality issues

---

## Output Data

### Files Created

1. **`cmb/theta_evolution_processor.py`** - Θ-field evolution processing module
   - `ThetaEvolutionProcessor` class
   - Evolution processing functions

### Data Structures

- **Processed Evolution Data:**
  - Time array: t
  - ω_min(t) interpolated values
  - ω_macro(t) interpolated values
  - Evolution rates
  - Metadata

---

## Dependencies

- **Step 0.1** (Configuration) - For time parameters
- **Step 1.1** (Θ-Field Data Loader) - For loading evolution data

---

## Related Steps

- **Step 2.1** (CMB Reconstruction) - Will use evolution data for temporal weighting
- **Step 3.2** (Sub-peaks Analysis) - Will use ω_min(t) for beatings analysis
- **Step 5.1** (High-l Peak) - Will use evolution for peak predictions

---

## Tests

### Unit Tests

1. **Evolution Data Loading**
   - Test loading ω_min(t) files
   - Test loading ω_macro(t) files
   - Test time array parsing
   - Test error handling

2. **Time Processing**
   - Test time range validation
   - Test interpolation accuracy
   - Test evolution rate calculation

3. **Evolution Interface**
   - Test time query methods
   - Test interpolation for arbitrary times
   - Test evolution calculations

4. **Data Validation**
   - Test time coverage validation
   - Test frequency range validation
   - Test data consistency checks

### Integration Tests

1. **End-to-End Evolution Processing**
   - Load data → Process → Validate → Provide interface
   - Test with actual evolution data files

---

## Implementation Notes

- Use efficient interpolation (scipy.interpolate)
- Handle large time arrays efficiently
- Provide clear error messages for time range issues
- Support multiple evolution data formats if needed

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

