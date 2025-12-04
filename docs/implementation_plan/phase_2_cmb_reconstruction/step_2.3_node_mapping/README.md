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
- Handle projection from z≈1100 to z=0
- **Use ONLY phase parameters** (ω_min(t), ω_macro(t)) for evolution
- **DO NOT use classical concepts** like "proper motion" (not applicable to Θ-nodes)
- **DO NOT use classical cosmological formulas** (FRW, ΛCDM)
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
   - **What to test:**
     - Test early universe to sky coordinate conversion:
       - Verify map_nodes_to_sky() converts z≈1100 to z=0 correctly
       - Test that ONLY phase parameters (ω_min(t), ω_macro(t)) are used
       - Verify NO classical cosmological formulas (FRW, ΛCDM) are used
       - Test coordinate ranges: theta [0, π], phi [0, 2π]
     - Test cosmological projection:
       - Verify projection preserves node relationships
       - Test projection accuracy
       - Verify all nodes are mapped to valid sky positions
     - Test HEALPix pixel mapping:
       - Verify sky coordinates are converted to HEALPix pixel indices
       - Test pixel indexing correctness
       - Verify pixel indices are valid for given NSIDE

2. **Node Catalog**
   - **What to test:**
     - Test catalog creation:
       - Verify create_node_catalog() creates NodeCmbMapping structure
       - Test all catalog fields are populated:
         - node_ids, sky_positions, pixel_indices
         - cmb_values, node_depths, node_temperatures
       - Verify catalog data integrity (all arrays have same length)
     - Test data integrity:
       - Verify sky_positions match pixel_indices
       - Verify cmb_values match CMB map at pixel positions
       - Verify node_depths and node_temperatures are consistent
     - Test query methods:
       - Verify get_nodes_at_position() finds nodes near given position
       - Test search radius parameter
       - Verify get_cmb_value_at_node() returns correct CMB map value

3. **Position Validation**
   - **What to test:**
     - Test position validity checks:
       - Verify positions are in valid ranges [0, π] × [0, 2π]
       - Test for invalid coordinates (should raise error)
     - Test edge case handling:
       - Test positions at boundaries (theta=0, π; phi=0, 2π)
       - Test positions near poles
       - Test empty node data
     - Test coordinate transformations:
       - Verify coordinate system consistency
       - Test transformation round-trip (if applicable)

4. **Mapping Interface**
   - **What to test:**
     - Test query methods:
       - Verify get_nodes_at_position() returns correct node IDs
       - Test search radius affects results correctly
       - Verify get_cmb_value_at_node() returns correct value
     - Test reverse lookup:
       - Verify can find nodes from CMB map positions
       - Test lookup accuracy
     - Test visualization support:
       - Verify mapping data can be used for visualization
       - Test node overlay on CMB maps

### Integration Tests

1. **End-to-End Mapping**
   - **What to test:**
     - Load nodes → Map to sky → Create catalog → Validate:
       - Load ThetaNodeData with node positions
       - Load reconstructed CMB map
       - Map nodes to sky coordinates (using phase parameters only)
       - Create node catalog with CMB positions
       - Validate catalog integrity
       - Test query methods
     - Test with actual node and map data:
       - Real Θ-node data
       - Real reconstructed CMB map
       - Verify mapping produces valid catalog
       - Verify nodes are correctly positioned on CMB map

---

## Implementation Notes

- Use efficient coordinate transformations
- Handle large numbers of nodes
- Provide clear mapping documentation
- Support multiple coordinate systems if needed

## Forbidden Elements

**CRITICAL PRINCIPLE (from tech_spec-new.md):** Matter does NOT influence Θ-field.
Nodes are topological structures - NOT created by matter.

**DO NOT USE:**
- ❌ Matter as source of nodes (nodes are topological structures of Θ-field)
- ❌ Baryon density, DM density or material terms
- ❌ Modification of Θ-field based on observed matter
- ❌ "Proper motion" (classical concept, not applicable to Θ-nodes)
- ❌ Classical cosmological formulas (FRW, ΛCDM) for projection
- ❌ FRW metric as primary model
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

