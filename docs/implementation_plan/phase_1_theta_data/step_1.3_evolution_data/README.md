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
   - **What to test:**
     - Test loading ω_min(t) files:
       - Verify correct parsing of time array t
       - Verify correct parsing of ω_min(t) values
       - Test data format validation
     - Test loading ω_macro(t) files:
       - Verify correct parsing of time array t
       - Verify correct parsing of ω_macro(t) values
       - Verify time arrays match between ω_min and ω_macro
     - Test time array parsing:
       - Verify time values are in valid range
       - Test time unit handling
       - Verify time array is sorted (monotonic)
     - Test error handling:
       - Missing files (FileNotFoundError)
       - Invalid formats (ValueError)
       - Inconsistent time arrays

2. **Time Processing**
   - **What to test:**
     - Test time range validation:
       - Verify validate_time_range() checks time is within data range
       - Test error for times outside range
       - Test boundary conditions (min/max time)
     - Test interpolation accuracy:
       - Verify interpolation at known data points returns exact values
       - Test interpolation between data points (check smoothness)
       - Verify interpolation preserves evolution trends
     - Test evolution rate calculation:
       - Verify get_evolution_rate_min() calculates d(ω_min)/dt correctly
       - Verify get_evolution_rate_macro() calculates d(ω_macro)/dt correctly
       - Test rate calculation accuracy (numerical derivative)

3. **Evolution Interface**
   - **What to test:**
     - Test time query methods:
       - Verify get_omega_min(time) returns correct value
       - Verify get_omega_macro(time) returns correct value
       - Test with various time values within range
     - Test interpolation for arbitrary times:
       - Test interpolation at times not in original data
       - Verify interpolation is smooth and continuous
       - Test extrapolation behavior (should raise error or handle gracefully)
     - Test evolution calculations:
       - Verify evolution processor creates correct interpolators
       - Test that interpolators are created during process() call

4. **Data Validation**
   - **What to test:**
     - Test time coverage validation:
       - Verify time array covers required range
       - Test for gaps in time coverage
       - Verify time array completeness
     - Test frequency range validation:
       - Verify ω_min(t) values are positive
       - Verify ω_macro(t) values are positive
       - Verify ω_min(t) < ω_macro(t) (if applicable)
     - Test data consistency checks:
       - Verify all arrays have consistent lengths
       - Verify time arrays match between ω_min and ω_macro
       - Test for NaN or infinite values

### Integration Tests

1. **End-to-End Evolution Processing**
   - **What to test:**
     - Load data → Process → Validate → Provide interface:
       - Load ThetaEvolution data
       - Create ThetaEvolutionProcessor
       - Call process() to create interpolators
       - Query evolution values at various times
       - Calculate evolution rates
       - Validate all operations work correctly
     - Test with actual evolution data files:
       - Real data file formats
       - Real time ranges and frequency values
       - Verify processing produces valid interpolators
       - Test that interpolators can be used in CMB calculations

---

## Implementation Notes

- Use efficient interpolation (scipy.interpolate)
- Handle large time arrays efficiently
- Provide clear error messages for time range issues
- Support multiple evolution data formats if needed

## Forbidden Elements

**CRITICAL PRINCIPLE (from tech_spec-new.md):** Matter does NOT influence Θ-field.
Evolution uses ONLY phase parameters.

**DO NOT USE:**
- ❌ Matter as source of evolution (evolution is from Θ-field only)
- ❌ Baryon density, DM density or material terms
- ❌ Classical cosmological evolution formulas (FRW, ΛCDM)
- ❌ FRW metric as primary model
- ❌ Mass, dark matter, or gravitational sources
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)
- ❌ Use ONLY phase parameters (ω_min(t), ω_macro(t))

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

