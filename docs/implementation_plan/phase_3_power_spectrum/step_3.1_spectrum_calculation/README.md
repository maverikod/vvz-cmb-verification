# Step 3.1: Power Spectrum Calculation

**Phase:** 3 - Power Spectrum Generation  
**Step:** 3.1 - Power Spectrum Calculation  
**Module:** `cmb/spectrum/`

---

## Overview

This step calculates the C_l power spectrum **directly from Θ-field frequency spectrum** ρ_Θ(ω,t). 
It implements C_l ∝ l⁻²–l⁻³ behavior without Silk damping and handles high-l range up to l≈10000.

**CRITICAL:** This step calculates C_l **directly from ρ_Θ(ω,t)**, NOT from reconstructed map.
This is the fundamental difference from classical cosmology.

---

## Input Data

- **Θ-Field Frequency Spectrum** (from [Step 1.1](../../phase_1_theta_data/step_1.1_data_loader/README.md)):
  - Frequency spectrum ρ_Θ(ω,t) from `ThetaFrequencySpectrum`
  - Temporal evolution data from `ThetaEvolution`

- **Configuration** (from [Step 0.1](../../phase_0_foundation/step_0.1_configuration/README.md)):
  - Physical constants (D, z_CMB)
  - Distance parameter D (~46 Gly)

- **Utilities** (from [Step 0.2](../../phase_0_foundation/step_0.2_utilities/README.md)):
  - Frequency conversion utilities (l = π D ω)

---

## Algorithm

### 1. Convert Frequencies to Multipoles

- Use fundamental formula: **l(ω) ≈ π D ω** (Formula 2.2 from tech_spec-new.md)
- Convert frequency array ω to multipole array l
- Handle frequency range corresponding to l=2 to l_max≈10000
- Calculate dω/dl = 1/(π D) for transformation

### 2. Calculate C_l Directly from ρ_Θ(ω,t)

- Use formula: **C_l ∝ ρ_Θ(ω(ℓ)) / ℓ²** (Formula 2.3 from tech_spec-new.md)
- Where ω(ℓ) = ℓ/(π D) from Formula 2.2
- **No plasma terms, no sound horizons**
- Result: **C_l ∝ l⁻²** (Formula 2.4)
- Handle temporal evolution by integrating over time
- **NO spherical harmonic decomposition** - direct calculation from spectrum
- **Note:** This is equivalent to CMB2.3: C_l ∝ ρ_Θ(ω(l)) · |dω/dl| where |dω/dl| = 1/(π D) gives C_l ∝ ρ_Θ(ω(ℓ)) / ℓ²

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

- **Step 0.1** (Configuration) - For physical constants (D)
- **Step 0.2** (Utilities) - For frequency conversion (l = π D ω)
- **Step 1.1** (Θ-Field Data Loader) - For frequency spectrum ρ_Θ(ω,t)
- **Step 1.3** (Evolution Data) - For temporal evolution ω_min(t), ω_macro(t)

**NOTE:** This step does NOT depend on Step 2.1 (CMB Reconstruction).
C_l is calculated directly from Θ-field spectrum, not from reconstructed map.

---

## Related Steps

- **Step 3.2** (Sub-peaks Analysis) - Will analyze power spectrum for peaks
- **Step 3.3** (Spectrum Comparison) - Will compare with observations

---

## Tests

### Unit Tests

1. **Frequency to Multipole Conversion**
   - **What to test:**
     - Test l = π D ω conversion:
       - Verify formula: l = π D ω (test with known values)
       - Verify D parameter from config is used correctly
       - Test conversion accuracy and precision
       - Test array conversion (vectorized operation)
     - Test dω/dl = 1/(π D) calculation:
       - Verify derivative calculation is correct
       - Test dω/dl is used in C_l calculation
     - Test multipole range handling:
       - Verify multipole range l=2 to l_max≈10000 is handled
       - Test edge cases (very low/high multipoles)
       - Verify frequency range corresponds to multipole range

2. **Direct C_l Calculation**
   - **What to test:**
     - Test C_l ∝ ρ_Θ(ω(ℓ)) / ℓ² formula (Formula 2.3 from tech_spec-new.md):
       - Verify formula is used correctly: C_l ∝ ρ_Θ(ω(ℓ)) / ℓ²
       - Test that ρ_Θ(ω(l)) is evaluated at correct frequencies
       - Verify calculation is DIRECT from spectrum (not from map decomposition)
       - Test that NO Spherical Harmonic Decomposition is used (direct calculation approach)
       - **Note:** Module B in tech_spec-new.md mentions Spherical Harmonic Decomposition,
         but this implementation uses direct calculation from ρ_Θ(ω,t) which is theoretically
         equivalent and more efficient. If Spherical Harmonic Decomposition is required,
         it should be tested separately.
     - Test C_l ∝ l⁻² behavior:
       - Verify calculated C_l follows l⁻² power law
       - Test power law fit accuracy
       - Verify C_l values are in expected range
     - Test temporal evolution integration:
       - Verify _integrate_temporal_evolution() integrates over time correctly
       - Test that ω_min(t) and ω_macro(t) are used
       - Verify integration preserves spectrum shape
     - Test error bar calculation:
       - Verify _calculate_errors() produces valid error bars
       - Test error propagation is correct
       - Verify errors are positive and reasonable

3. **Power Law Validation**
   - **What to test:**
     - Test l⁻²–l⁻³ behavior:
       - Verify validate_power_law() checks C_l ∝ l⁻²–l⁻³
       - Test power law fit to calculated spectrum
       - Verify power law index is within expected range (-2 to -3)
     - Test high-l tail:
       - Verify high-l tail (l>2000) follows power law
       - Test tail behavior matches theoretical prediction
     - Test no Silk damping:
       - Verify spectrum does NOT show exp(-l²/l_silk²) behavior
       - Test that high-l tail continues without exponential cutoff

4. **High-l Processing**
   - **What to test:**
     - Test high-l range handling:
       - Verify high-l multipoles (l>2000, up to l≈10000) are processed correctly
       - Test memory usage with large l_max
     - Test computational efficiency:
       - Verify calculation completes in reasonable time
       - Test vectorized operations are used
     - Test memory usage:
       - Verify memory usage is acceptable for large l_max
       - Test that arrays are not unnecessarily duplicated

### Integration Tests

1. **End-to-End Spectrum Calculation**
   - **What to test:**
     - Load ρ_Θ(ω,t) → Convert ω to l → Calculate C_l → Validate:
       - Load ThetaFrequencySpectrum with ρ_Θ(ω,t)
       - Load ThetaEvolution with ω_min(t), ω_macro(t)
       - Convert frequencies to multipoles using l = π D ω (Formula 2.2)
       - Calculate C_l directly from ρ_Θ(ω(ℓ)) / ℓ² (Formula 2.3)
       - Integrate over temporal evolution
       - Validate C_l ∝ l⁻² behavior (Formula 2.4)
       - Verify NO spherical harmonic decomposition is used (direct calculation approach)
       - **Note:** This uses direct calculation from ρ_Θ(ω,t), not Spherical Harmonic
         Decomposition of ΔT(n̂) as mentioned in Module B. This approach is theoretically
         equivalent and more efficient.
     - Test with actual Θ-field frequency spectrum data:
       - Real frequency spectrum data
       - Real evolution data
       - Verify calculation produces valid PowerSpectrum
     - Verify C_l ∝ l⁻² behavior matches ACT/SPT observations:
       - Compare calculated spectrum with ACT/SPT observations
       - Verify power law behavior matches
       - Verify high-l tail matches (no Silk damping)

---

## Implementation Notes

- **CRITICAL:** Calculate C_l directly from ρ_Θ(ω,t), NOT from map decomposition
- Use Formula 2.3: C_l ∝ ρ_Θ(ω(ℓ)) / ℓ² (from tech_spec-new.md)
- Handle large l_max values efficiently (up to l≈10000) - Module B requirement
- Integrate over temporal evolution ω_min(t), ω_macro(t)
- Implement proper error propagation
- Provide detailed logging
- **Note on Module B:** Module B in tech_spec-new.md mentions Spherical Harmonic
  Decomposition of ΔT(n̂), but this implementation uses direct calculation from
  ρ_Θ(ω,t) which is theoretically equivalent (Formula 2.3) and more efficient.
  Sub-peaks are extracted according to ρ_Θ(ω) structure (Module B requirement).

## Forbidden Elements

**DO NOT USE:**
- ❌ Matter as source of spectrum (spectrum comes from Θ-field only)
- ❌ Baryon density, DM density or material terms
- ❌ Plasma terms or sound horizons
- ❌ Modification of Θ-field based on observed matter
- ❌ Spherical harmonic decomposition of reconstructed map
- ❌ Classical formula: C_l = (1/(2l+1)) * Σ_m |a_lm|²
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)
- ❌ Silk damping formulas

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

