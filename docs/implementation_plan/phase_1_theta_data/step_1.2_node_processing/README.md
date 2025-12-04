# Step 1.2: Θ-Node Data Processing

**Phase:** 1 - Θ-Field Data Processing  
**Step:** 1.2 - Θ-Node Data Processing  
**Module:** `cmb/`

---

## Overview

This step processes Θ-node structure data from `data/theta/` directory. It handles node geometry (scale ~300 pc at z≈1100), processes node depth data (Δω/ω), and maps depth to temperature fluctuations (ΔT ≈ 20-30 μK).

---

## Input Data

- **Θ-Field Data** (from [Step 1.1](../step_1.1_data_loader/README.md)):
  - Frequency spectrum data loaded via `ThetaFrequencySpectrum`
  - Node geometry data files from `data/theta/nodes/`
  - Node depth data (Δω/ω values)

- **Configuration** (from [Step 0.1](../../phase_0_foundation/step_0.1_configuration/README.md)):
  - Physical constants (z_CMB ≈ 1100)
  - Node scale parameters (~300 pc)
  - Temperature conversion factors

---

## Algorithm

### 1. Load Node Geometry Data

- Load node structure data from `data/theta/nodes/`
- Parse node positions and scales
- Validate node geometry (scale ~300 pc at z≈1100)
- Handle arcmin-scale projection (2-5′ → 100-300 pc)

### 2. Process Node Depth Data

- Load node depth values (Δω/ω)
- Validate depth ranges
- Handle depth distribution
- Calculate depth statistics

### 3. Map Depth to Temperature

- Convert node depth (Δω/ω) to temperature fluctuation (ΔT)
- **Formula (from tech_spec-new.md 2.1):** ΔT = (Δω/ω_CMB) T_0
  - Where T_0 = 2.725 K (CMB temperature)
  - ω_CMB ~ 10^11 Hz
  - Δω = ω - ω_min (depth of node)
  - Result: ΔT ≈ 20-30 μK
- This is direct conversion from node depth, NOT linear approximation
- Validate temperature range (20-30 μK)
- Create depth-to-temperature mapping

### 4. Create Node Data Structure

- Combine geometry and depth data
- Create node catalog with positions, depths, and temperatures
- Provide interface for CMB reconstruction

---

## Output Data

### Files Created

1. **`cmb/theta_node_processor.py`** - Θ-node data processing module
   - `ThetaNodeData` class
   - Node processing functions

### Data Structures

- **ThetaNodeData:**
  - Node positions (sky coordinates)
  - Node scales (~300 pc)
  - Node depths (Δω/ω)
  - Temperature fluctuations (ΔT in μK)
  - Metadata

---

## Dependencies

- **Step 0.1** (Configuration) - For constants and parameters
- **Step 1.1** (Θ-Field Data Loader) - For loading node data files

---

## Related Steps

- **Step 1.3** (Evolution Data) - Will use node structure
- **Step 2.1** (CMB Reconstruction) - Will use processed node data
- **Step 2.3** (Node Mapping) - Will use node positions

---

## Tests

### Unit Tests

1. **Node Geometry Loading**
   - **What to test:**
     - Test loading node geometry files:
       - Verify correct parsing of node positions (theta, phi) in radians
       - Verify correct parsing of node scales in parsec
       - Test data format validation (CSV, JSON)
       - Verify array shapes: positions (N, 2), scales (N,)
     - Test scale validation (~300 pc):
       - Verify scales are approximately 300 pc at z≈1100
       - Test scale range validation
       - Verify scale units are correct (parsec)
     - Test arcmin conversion:
       - Verify 2-5′ → 100-300 pc conversion at z≈1100
       - Test conversion accuracy
     - Test error handling:
       - Missing files (FileNotFoundError)
       - Invalid formats (ValueError)
       - Inconsistent data shapes

2. **Node Depth Processing**
   - **What to test:**
     - Test depth data loading:
       - Verify correct parsing of node depths (Δω/ω)
       - Test data format validation
       - Verify array shape consistency with geometry data
     - Test depth range validation:
       - Verify depths are in valid range (typically small positive values)
       - Test for negative depths (should raise error or handle appropriately)
     - Test depth statistics calculation:
       - Mean, std, min, max depth values
       - Depth distribution analysis

3. **Temperature Mapping**
   - **What to test:**
     - Test Δω/ω → ΔT conversion:
       - Verify formula (from tech_spec-new.md 2.1): ΔT = (Δω/ω_CMB) T_0
       - Verify Δω = ω - ω_min (depth of node) is used correctly
       - Verify T_0 = 2.725 K is used correctly
       - Verify ω_CMB ~ 10^11 Hz is used correctly
       - Test that conversion is direct (NOT linear approximation)
     - Test temperature range (20-30 μK):
       - Verify all converted temperatures fall within 20-30 μK for typical depths
       - Test with various depth values
     - Test conversion accuracy:
       - Verify conversion precision
       - Test round-trip consistency (if applicable)

4. **Node Data Structure**
   - **What to test:**
     - Test ThetaNodeData class:
       - Verify all attributes (positions, scales, depths, temperatures, metadata)
       - Test data access methods
       - Verify data consistency (all arrays have same length N)
     - Test data access methods:
       - Test accessing individual node properties
       - Test array slicing and indexing
     - Test data validation:
       - validate_node_data() checks:
         - All arrays have consistent lengths
         - Positions are in valid ranges [0, π] × [0, 2π]
         - Scales are positive
         - Depths are in valid range
         - Temperatures are in 20-30 μK range

### Integration Tests

1. **End-to-End Node Processing**
   - **What to test:**
     - Load geometry → Process depth → Map to temperature → Create structure:
       - Load node geometry from data files
       - Load node depth data
       - Map depths to temperatures using formula ΔT = (Δω/ω_CMB) T_0 (where Δω = ω - ω_min)
       - Create ThetaNodeData structure
       - Validate complete structure
     - Test with actual node data files:
       - Real data file formats
       - Real data sizes and ranges
       - Verify processing produces valid ThetaNodeData

---

## Implementation Notes

- Node scale must match theoretical value (~300 pc at z≈1100)
- Temperature conversion must be accurate (20-30 μK range)
- Handle large numbers of nodes efficiently
- Provide clear error messages for invalid data

## Forbidden Elements

**CRITICAL PRINCIPLE (from tech_spec-new.md):** Matter does NOT influence Θ-field.
Nodes are topological structures - NOT created by matter.

**DO NOT USE:**
- ❌ Matter as source of nodes (nodes are topological structures of Θ-field)
- ❌ Baryon density, DM density or material terms
- ❌ Modification of Θ-field based on observed matter
- ❌ Linear approximation for Δω/ω → ΔT (use exact formula: ΔT = (Δω/ω_CMB) T_0)
- ❌ Classical thermodynamic formulas
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

