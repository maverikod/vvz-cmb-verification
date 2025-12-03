# Step 2.3: Node-to-Map Mapping

**Phase:** 2 - CMB Map Reconstruction  
**Step:** 2.3 - Node-to-Map Mapping  
**Module:** `cmb/nodes/`

---

## Overview

This step creates a mapping between Θ-nodes and CMB map positions. It maps nodes from early universe coordinates (z≈1100) to current sky coordinates and creates a node catalog with CMB positions.

---

## Input Data

- **Θ-Node Data** (from [Step 1.2](../../phase_1_theta_data/step_1.2_node_processing/README.md)):
  - Node positions and properties from `ThetaNodeData`

- **Reconstructed CMB Map** (from [Step 2.1](../step_2.1_reconstruction_core/README.md)):
  - HEALPix map with node contributions

- **Configuration** (from [Step 0.1](../../phase_0_foundation/step_0.1_configuration/README.md)):
  - Cosmological parameters (z_CMB ≈ 1100)
  - Projection parameters

---

## Algorithm

### 1. Map Nodes to Sky Coordinates

- Convert early universe node positions to current sky coordinates
- Handle cosmological projection (z≈1100 → z=0)
- Account for proper motion and evolution
- Map to HEALPix pixel indices

### 2. Create Node Catalog

- Combine node properties with sky positions
- Create catalog with:
  - Node IDs
  - Sky coordinates (theta, phi)
  - HEALPix pixel indices
  - Node depths and temperatures
  - CMB map values at node positions

### 3. Validate Node Positions

- Check that nodes map to valid sky positions
- Verify node-CMB correspondence
- Handle edge cases (poles, boundaries)
- Validate coordinate transformations

### 4. Create Mapping Interface

- Provide query methods for node positions
- Support reverse lookup (sky position → nodes)
- Handle node clustering
- Provide visualization support

---

## Output Data

### Files Created

1. **`cmb/nodes/node_to_cmb_mapper.py`** - Node mapping module
   - `NodeToCmbMapper` class
   - Mapping functions

### Node Catalog

- **Node-CMB Mapping:**
  - Node catalog with sky positions
  - HEALPix pixel indices
  - CMB map values at nodes
  - Mapping metadata

---

## Dependencies

- **Step 0.1** (Configuration) - For cosmological parameters
- **Step 1.2** (Node Processing) - For node data
- **Step 2.1** (CMB Reconstruction) - For CMB map

---

## Related Steps

- **Step 4.3** (Node-LSS Mapping) - Will use node positions for LSS correlation
- **Step 6.1** (Cluster Plateau) - Will use node directions

---

## Tests

### Unit Tests

1. **Coordinate Mapping**
   - Test early universe to sky coordinate conversion
   - Test cosmological projection
   - Test HEALPix pixel mapping

2. **Node Catalog**
   - Test catalog creation
   - Test data integrity
   - Test query methods

3. **Position Validation**
   - Test position validity checks
   - Test edge case handling
   - Test coordinate transformations

4. **Mapping Interface**
   - Test query methods
   - Test reverse lookup
   - Test visualization support

### Integration Tests

1. **End-to-End Mapping**
   - Load nodes → Map to sky → Create catalog → Validate
   - Test with actual node and map data

---

## Implementation Notes

- Use efficient coordinate transformations
- Handle large numbers of nodes
- Provide clear mapping documentation
- Support multiple coordinate systems if needed

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

