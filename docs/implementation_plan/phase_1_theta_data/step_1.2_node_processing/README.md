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
- Formula: ΔT ≈ 20-30 μK via Δω/ω (see [tech_spec.md](../../../tech_spec.md) section 13.6)
- Validate temperature range
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
   - Test loading node geometry files
   - Test scale validation (~300 pc)
   - Test arcmin conversion
   - Test error handling

2. **Node Depth Processing**
   - Test depth data loading
   - Test depth range validation
   - Test depth statistics calculation

3. **Temperature Mapping**
   - Test Δω/ω → ΔT conversion
   - Test temperature range (20-30 μK)
   - Test conversion accuracy

4. **Node Data Structure**
   - Test ThetaNodeData class
   - Test data access methods
   - Test data validation

### Integration Tests

1. **End-to-End Node Processing**
   - Load geometry → Process depth → Map to temperature → Create structure
   - Test with actual node data files

---

## Implementation Notes

- Node scale must match theoretical value (~300 pc at z≈1100)
- Temperature conversion must be accurate (20-30 μK range)
- Handle large numbers of nodes efficiently
- Provide clear error messages for invalid data

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

