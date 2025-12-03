# Step 2.1: CMB Reconstruction Core

**Phase:** 2 - CMB Map Reconstruction  
**Step:** 2.1 - CMB Reconstruction Core  
**Module:** `cmb/reconstruction/`

---

## Overview

This step implements the core CMB map reconstruction from Θ-field frequency spectrum. It uses formula CMB2.1 to generate spherical harmonic map ΔT(n̂) from Θ-field nodes.

---

## Input Data

- **Θ-Field Data** (from [Step 1.1](../../phase_1_theta_data/step_1.1_data_loader/README.md)):
  - Frequency spectrum ρ_Θ(ω,t)
  - Node depth data (Δω/ω)
  - Node geometry data

- **Configuration** (from [Step 0.1](../../phase_0_foundation/step_0.1_configuration/README.md)):
  - Physical constants (D, z_CMB)
  - Conversion factors

- **Utilities** (from [Step 0.2](../../phase_0_foundation/step_0.2_utilities/README.md)):
  - Spherical harmonics utilities
  - Frequency conversion utilities

---

## Algorithm

### 1. Convert Θ-Field Nodes to Temperature Fluctuations

- Map node depth (Δω/ω) to temperature fluctuation (ΔT)
- Formula: ΔT ≈ 20-30 μK via Δω/ω (see [tech_spec.md](../../../tech_spec.md) section 13.6)
- Handle node geometry (scale ~300 pc at z≈1100)

### 2. Project Nodes to Sky Coordinates

- Map Θ-nodes from early universe (z≈1100) to current sky coordinates
- Handle arcmin-scale projection (2-5′ → 100-300 pc)
- Account for cosmological evolution

### 3. Generate Spherical Harmonic Map

- Use formula CMB2.1 for reconstruction
- Convert node positions and amplitudes to spherical harmonics
- Generate full-sky HEALPix map ΔT(n̂)
- Use spherical harmonic synthesis

### 4. Apply Frequency Spectrum

- Weight contributions by frequency spectrum ρ_Θ(ω,t)
- Integrate over frequency range
- Handle temporal evolution

---

## Output Data

### Files Created

1. **`cmb/reconstruction/cmb_map_reconstructor.py`** - Main reconstruction class
   - `CmbMapReconstructor` class
   - Reconstruction methods

### Output Maps

- **Reconstructed CMB Map:**
  - HEALPix format
  - Temperature fluctuations ΔT in μK
  - Full-sky coverage
  - Arcmin-scale resolution

---

## Dependencies

- **Step 0.1** (Configuration) - For constants
- **Step 0.2** (Utilities) - For spherical harmonics and conversions
- **Step 1.1** (Θ-Field Data Loader) - For theta data
- **Step 1.2** (Node Processing) - For node depth calculations

---

## Related Steps

- **Step 2.2** (Map Validation) - Will validate reconstructed maps
- **Step 2.3** (Node Mapping) - Will create node-to-CMB mapping
- **Step 3.1** (Power Spectrum) - Will use reconstructed map

---

## Tests

### Unit Tests

1. **Temperature Conversion**
   - Test Δω/ω → ΔT conversion
   - Test amplitude range (20-30 μK)
   - Test edge cases

2. **Sky Projection**
   - Test node to sky coordinate mapping
   - Test arcmin scale conversion
   - Test cosmological projection

3. **Spherical Harmonic Generation**
   - Test harmonic decomposition
   - Test map synthesis
   - Test HEALPix conversion

4. **Reconstruction Accuracy**
   - Test reconstruction formula (CMB2.1)
   - Test frequency spectrum integration
   - Test temporal evolution handling

### Integration Tests

1. **End-to-End Reconstruction**
   - Load theta data → Reconstruct map → Validate → Save
   - Compare with ACT DR6.02 observations

---

## Implementation Notes

- Use efficient HEALPix operations
- Handle large maps (high NSIDE)
- Implement progress tracking for long operations
- Provide detailed logging

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

