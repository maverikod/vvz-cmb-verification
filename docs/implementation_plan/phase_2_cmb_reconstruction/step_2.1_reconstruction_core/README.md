# Step 2.1: CMB Reconstruction Core

**Phase:** 2 - CMB Map Reconstruction  
**Step:** 2.1 - CMB Reconstruction Core  
**Module:** `cmb/reconstruction/`

---

## Overview

This step implements the core CMB map reconstruction from Θ-field frequency spectrum. 
It uses formula CMB2.1 to generate spherical harmonic map ΔT(n̂) from Θ-field nodes.

**Formula CMB2.1:**
```
ρ_Θ(ω,t) ∝ ω^(-2±0.5) · (t/t_0)^(-3±1)
```

This frequency spectrum is the fundamental source of CMB microstructure.

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

**CRITICAL:** Frequency spectrum ρ_Θ(ω,t) must be integrated from the beginning,
NOT applied as a correction after map creation.

### 1. Integrate Frequency Spectrum ρ_Θ(ω,t)

- Use formula CMB2.1: ρ_Θ(ω,t) ∝ ω^(-2±0.5) · (t/t_0)^(-3±1)
- Integrate over frequency range for each node
- Account for temporal evolution ω_min(t), ω_macro(t)
- This is the PRIMARY source of CMB structure

### 2. Convert Θ-Field Nodes to Temperature Fluctuations

- Map node depth (Δω/ω) to temperature fluctuation (ΔT)
- **Formula (from tech_spec-new.md 2.1):** ΔT = (Δω/ω_CMB) T_0
  - Where T_0 = 2.725 K (CMB temperature)
  - ω_CMB ~ 10^11 Hz
  - Δω = ω - ω_min (depth of node)
  - Result: ΔT ≈ 20-30 μK
- Handle node geometry (scale ~300 pc at z≈1100)

### 3. Project Nodes to Sky Coordinates

- Map Θ-nodes from early universe (z≈1100) to current sky coordinates
- Handle arcmin-scale projection (2-5′ → 100-300 pc)
- **Use ONLY phase parameters** (ω_min(t), ω_macro(t)) for evolution
- **DO NOT use classical cosmological formulas** (FRW, ΛCDM)

### 4. Generate Spherical Harmonic Map

- Use integrated frequency spectrum from step 1
- Convert node positions and weighted amplitudes to spherical harmonics
- Generate full-sky HEALPix map ΔT(n̂)
- Use spherical harmonic synthesis
- Map represents direct projection of Θ-field nodes

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

1. **Node Depth Calculation for Each Direction (Module A requirement)**
   - **What to test:**
     - Test Δω(n̂) = ω(n̂) - ω_min(n̂) calculation for each direction n̂:
       - Verify for each direction n̂, depth is calculated as Δω(n̂) = ω(n̂) - ω_min(n̂)
       - Test that ω(n̂) is obtained from frequency spectrum ρ_Θ(ω) for direction n̂
       - Test that ω_min(n̂) is obtained from node map ω_min(x) (from Step 1.4) for direction n̂
       - Verify calculation is performed for all directions n̂ on sky
       - Test that node map ω_min(x) is correctly used for each direction
       - Verify depth calculation uses correct node map from Module C
     - Test direction-by-direction processing:
       - Verify each direction n̂ gets its own depth calculation
       - Test that depth varies correctly across sky
       - Verify depth calculation is independent for each direction
       - Test that node map is correctly mapped to sky directions

2. **Temperature Conversion**
   - **What to test:**
     - Test Δω(n̂) → ΔT conversion formula (from tech_spec-new.md 2.1): ΔT = (Δω/ω_CMB) T_0
       - Verify T_0 = 2.725 K is used correctly
       - Verify ω_CMB ~ 10^11 Hz is used correctly
       - Verify Δω = ω - ω_min (depth of node) is used correctly
       - Verify conversion produces ΔT ≈ 20-30 μK for typical node depths
       - Test that conversion is direct (not linear approximation)
       - Verify formula is applied for each direction n̂
     - Test amplitude range (20-30 μK):
       - Verify all converted temperatures fall within expected range
       - Test with various node depth values (Δω/ω)
       - Verify range is correct for all directions n̂
     - Test edge cases:
       - Zero depth (Δω/ω = 0) → ΔT = 0
       - Very large depth values
       - Negative depths (should raise error or handle appropriately)

2. **Sky Projection**
   - **What to test:**
     - Test node to sky coordinate mapping:
       - Verify (theta, phi) coordinates are in valid ranges [0, π] and [0, 2π]
       - Test projection from z≈1100 to z=0
       - Verify projection uses ONLY phase parameters (ω_min(t), ω_macro(t))
       - Verify NO classical cosmological formulas (FRW, ΛCDM) are used
     - Test arcmin scale conversion:
       - Verify 2-5′ → 100-300 pc at z≈1100 conversion
       - Test scale consistency across nodes
     - Test cosmological projection:
       - Verify all nodes are mapped to valid sky positions
       - Test projection preserves node relationships

3. **Map Assembly (Module A requirement)**
   - **What to test:**
     - Test map assembly from direction-by-direction calculations:
       - Verify map is assembled from ΔT(n̂) values for each direction n̂
       - Test that all directions n̂ are processed
       - Verify map covers full sky
       - Test map assembly uses correct HEALPix pixelization
     - Test HEALPix map creation:
       - Verify map is created in HEALPix format
       - Verify map has Nside≥2048 (Module A requirement)
       - Test correct pixel indexing
       - Test map format compatibility
       - Verify temperature units (μK)

4. **Spherical Harmonic Generation**
   - **What to test:**
     - Test harmonic decomposition (if used):
       - Verify correct spherical harmonic coefficients a_lm calculation
       - Test decomposition accuracy
     - Test map synthesis:
       - Verify synthesis from a_lm produces correct HEALPix map
       - Test full-sky coverage
       - Verify map resolution (NSIDE parameter)

5. **Reconstruction Accuracy**
   - **What to test:**
     - Test reconstruction formula (CMB2.1):
       - Verify ρ_Θ(ω,t) ∝ ω^(-2±0.5) · (t/t_0)^(-3±1) is used correctly
       - Test that spectrum is integrated from beginning (not post-processing)
       - Verify integration over frequency range for each node
     - Test frequency spectrum integration:
       - Verify integration produces correct weights for each node
       - Test integration accounts for temporal evolution
       - Verify integration preserves spectrum shape
     - Test temporal evolution handling:
       - Verify ω_min(t) and ω_macro(t) are used correctly
       - Test evolution interpolation accuracy
       - Verify evolution is integrated over time

### Integration Tests

1. **End-to-End Reconstruction (Module A)**
   - **What to test:**
     - Load theta data → Calculate depth for each direction → Apply ΔT formula → Assemble map → Validate:
       - Load frequency spectrum ρ_Θ(ω) (Module A input)
       - Load node map ω_min(x) from Step 1.4 (Module A input)
       - For each direction n̂: calculate Δω(n̂) = ω(n̂) - ω_min(n̂)
       - For each direction n̂: apply formula ΔT = (Δω/ω_CMB) T_0
       - Assemble map ΔT(n̂) in HEALPix format (Nside≥2048)
       - Validate map properties (amplitude, scale, coverage)
       - Save map in HEALPix format
       - Verify saved map can be reloaded correctly
       - Verify map matches Module A requirements
     - Compare with ACT DR6.02 observations:
       - Load ACT DR6.02 map
       - Compare reconstructed vs observed:
         - Arcmin-scale structures (2-5′)
         - Temperature amplitude (20-30 μK)
         - Spatial correlation
       - Verify reconstruction matches observations within expected uncertainties

---

## Implementation Notes

- **CRITICAL:** Integrate ρ_Θ(ω,t) from the beginning, not as post-processing
- Use efficient HEALPix operations
- Handle large maps (high NSIDE)
- Implement progress tracking for long operations
- Provide detailed logging

## Forbidden Elements

**DO NOT USE:**
- ❌ Classical cosmological formulas (FRW, ΛCDM) for evolution
- ❌ Potentials V(φ), V(Θ)
- ❌ Matter as source of nodes (nodes are topological structures of Θ-field)
- ❌ Baryon density, DM density or material terms
- ❌ Modification of Θ-field based on observed matter
- ❌ T_μν of matter in Θ-field equations
- ❌ Reverse reaction of matter on Θ-field
- ❌ Baryon acoustic oscillations as physical mechanism
- ❌ FRW metric as primary model
- ❌ Mass, dark matter, or gravitational sources
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)
- ❌ "Proper motion" (classical concept, not applicable to Θ-nodes)
- ❌ Creating map first, then applying frequency spectrum

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

