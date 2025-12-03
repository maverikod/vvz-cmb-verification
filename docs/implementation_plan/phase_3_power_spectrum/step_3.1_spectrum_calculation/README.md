# Step 3.1: Power Spectrum Calculation

**Phase:** 3 - Power Spectrum Generation  
**Step:** 3.1 - Power Spectrum Calculation  
**Module:** `cmb/spectrum/`

---

## Overview

This step calculates the C_l power spectrum from the reconstructed CMB map. It implements C_l ∝ l⁻²–l⁻³ behavior without Silk damping and handles high-l range up to l≈10000.

---

## Input Data

- **Reconstructed CMB Map** (from [Step 2.1](../../phase_2_cmb_reconstruction/step_2.1_reconstruction_core/README.md)):
  - HEALPix map from `CmbMapReconstructor`
  - Temperature fluctuations ΔT

- **Utilities** (from [Step 0.2](../../phase_0_foundation/step_0.2_utilities/README.md)):
  - Spherical harmonics utilities
  - Mathematical functions

---

## Algorithm

### 1. Spherical Harmonic Decomposition

- Decompose CMB map into spherical harmonics
- Calculate harmonic coefficients a_lm
- Handle full-sky coverage
- Use efficient HEALPix operations

### 2. Calculate Power Spectrum

- Compute C_l = (1/(2l+1)) * Σ_m |a_lm|²
- Handle all multipoles from l=2 to l_max≈10000
- Implement without Silk damping
- Calculate error bars

### 3. Validate Power Law Behavior

- Check C_l ∝ l⁻²–l⁻³ behavior
- Verify high-l tail (l>2000)
- Validate no Silk damping
- Compare with theoretical expectations

### 4. Handle High-l Range

- Process high-l multipoles (l>2000)
- Ensure computational efficiency
- Handle memory constraints
- Provide progress tracking

---

## Output Data

### Files Created

1. **`cmb/spectrum/power_spectrum.py`** - Power spectrum calculation module
   - `PowerSpectrumCalculator` class
   - Calculation functions

### Power Spectrum Data

- **C_l Power Spectrum:**
  - Multipole array: l
  - Power spectrum values: C_l
  - Error bars: σ_C_l
  - Metadata (nside, l_max, etc.)

---

## Dependencies

- **Step 0.2** (Utilities) - For spherical harmonics
- **Step 2.1** (CMB Reconstruction) - For reconstructed map

---

## Related Steps

- **Step 3.2** (Sub-peaks Analysis) - Will analyze power spectrum for peaks
- **Step 3.3** (Spectrum Comparison) - Will compare with observations

---

## Tests

### Unit Tests

1. **Harmonic Decomposition**
   - Test spherical harmonic decomposition
   - Test coefficient calculation
   - Test full-sky handling

2. **Power Spectrum Calculation**
   - Test C_l calculation formula
   - Test multipole range handling
   - Test error bar calculation

3. **Power Law Validation**
   - Test l⁻²–l⁻³ behavior
   - Test high-l tail
   - Test no Silk damping

4. **High-l Processing**
   - Test high-l range handling
   - Test computational efficiency
   - Test memory usage

### Integration Tests

1. **End-to-End Spectrum Calculation**
   - Load map → Decompose → Calculate C_l → Validate
   - Test with reconstructed maps

---

## Implementation Notes

- Use efficient HEALPix anafast for decomposition
- Handle large l_max values efficiently
- Implement proper error propagation
- Provide detailed logging

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

