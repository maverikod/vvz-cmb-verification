# Step 1.4: Θ-Node Map Generation

**Phase:** 1 - Θ-Field Data Processing  
**Step:** 1.4 - Θ-Node Map Generation  
**Module:** `cmb/nodes/`

---

## Overview

This step implements **Module C** from tech_spec-new.md: generation of Θ-node map from field ω(x).

Finds nodes as points of local minima ω_min(x) and classifies them by depth, area, and local curvature.

**CRITICAL PRINCIPLE:** Nodes are topological structures of Θ-field. Matter does NOT influence Θ-field.
Nodes are NOT created by matter - matter "sits" in nodes.

---

## Input Data

- **Θ-Field Data** (from [Step 1.1](../step_1.1_data_loader/README.md)):
  - Field ω(x) on grid
  - Frequency spectrum ρ_Θ(ω)

- **Configuration** (from [Step 0.1](../../phase_0_foundation/step_0.1_configuration/README.md)):
  - Physical constants
  - Grid parameters

---

## Algorithm

### 1. Find Nodes as Local Minima

- Find points where ω(x) = ω_min(x) locally
- Use local minimum detection algorithm
- Handle grid resolution and boundary conditions
- **Node definition:** x_node = {x : ω(x) = ω_min(x)}

### 2. Classify Nodes

- **By depth (Δω/ω):**
  - Calculate depth for each node: Δω = ω - ω_min
  - Classify nodes by depth ranges
- **By area:**
  - Calculate node area (spatial extent)
  - Classify by area ranges
- **By local curvature:**
  - Calculate local curvature at node positions
  - Classify by curvature values

### 3. Create Node Map

- Generate map ω_min(x) showing node positions
- Create node mask (binary map of node locations)
- Store node classification data

### 4. Validate Node Map

- Verify all nodes are detected
- Check node classification consistency
- Validate map completeness

---

## Output Data

### Files Created

1. **`cmb/nodes/theta_node_map.py`** - Θ-node map generation module
   - `ThetaNodeMapGenerator` class
   - Node detection and classification functions

### Output Maps

- **Node Map ω_min(x):**
  - Map of local minima positions
  - Node depth values
  - Spatial grid format

- **Node Mask:**
  - Binary mask of node locations
  - Same grid as input field

- **Node Classification:**
  - Node properties (depth, area, curvature)
  - Node catalog with classifications

---

## Dependencies

- **Step 0.1** (Configuration) - For grid parameters
- **Step 1.1** (Θ-Field Data Loader) - For field ω(x) data

---

## Related Steps

- **Step 1.2** (Node Processing) - Will use node map for depth calculations
- **Step 2.1** (CMB Reconstruction) - Will use node map for reconstruction
- **Step 4.3** (Node-LSS Mapping) - Will use node map for LSS correlation

---

## Tests

### Unit Tests

1. **Node Detection**
   - **What to test:**
     - Test local minimum detection:
       - Verify nodes are found at points where ω(x) = ω_min(x)
       - Test detection algorithm accuracy
       - Test handling of grid boundaries
       - Test detection with various field configurations
     - Test node definition:
       - Verify x_node = {x : ω(x) = ω_min(x)} is correctly implemented
       - Test node identification accuracy

2. **Node Classification**
   - **What to test:**
     - Test depth classification:
       - Verify Δω = ω - ω_min is calculated correctly
       - Test depth ranges are correctly assigned
       - Verify depth classification accuracy
     - Test area classification:
       - Verify node area calculation
       - Test area ranges are correctly assigned
       - Verify area classification accuracy
     - Test curvature classification:
       - Verify local curvature calculation at node positions
       - Test curvature ranges are correctly assigned
       - Verify curvature classification accuracy

3. **Map Generation**
   - **What to test:**
     - Test ω_min(x) map creation:
       - Verify map shows all node positions correctly
       - Test map format and grid consistency
       - Verify map values are correct
     - Test node mask creation:
       - Verify mask is binary (0/1) at correct positions
       - Test mask covers all detected nodes
       - Verify mask format consistency

4. **Map Validation**
   - **What to test:**
     - Test node detection completeness:
       - Verify all nodes are detected (no false negatives)
       - Test no false positives
     - Test classification consistency:
       - Verify all nodes are classified
       - Test classification data integrity
     - Test map completeness:
       - Verify map covers entire field
       - Test map data integrity

### Integration Tests

1. **End-to-End Node Map Generation**
   - **What to test:**
     - Load field ω(x) → Detect nodes → Classify → Generate maps → Validate:
       - Load Θ-field data with ω(x) on grid
       - Detect nodes as local minima
       - Classify nodes (depth, area, curvature)
       - Generate ω_min(x) map and node mask
       - Validate maps and classifications
       - Verify all steps complete successfully
     - Test with actual Θ-field data:
       - Real field data on grid
       - Verify nodes are correctly detected
       - Verify classifications are correct
       - Verify maps are valid

---

## Implementation Notes

- Use efficient local minimum detection algorithms
- Handle grid resolution carefully (nodes should be at grid scale ~300 pc)
- Classify nodes by multiple properties (depth, area, curvature)
- Provide clear node map documentation
- **CRITICAL:** Nodes are topological structures - NOT created by matter

## Forbidden Elements

**DO NOT USE:**
- ❌ Matter as source of nodes (nodes are topological structures of Θ-field)
- ❌ Baryon density, DM density in node detection
- ❌ Modification of Θ-field based on observed matter
- ❌ T_μν of matter in Θ-field equations
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com
