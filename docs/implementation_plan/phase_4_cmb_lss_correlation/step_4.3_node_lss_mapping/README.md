# Step 4.3: Node-LSS Mapping

**Phase:** 4 - CMB-LSS Correlation  
**Step:** 4.3 - Node-LSS Mapping  
**Module:** `cmb/correlation/`

---

## Overview

This step implements **Module D** from tech_spec-new.md: maps CMB nodes to LSS structures.

**CRITICAL PRINCIPLE:** Galaxies "sit" in Θ-nodes, NOT create them.
Matter does NOT influence Θ-field. Nodes are topological structures.

**Module D Requirements:**
- Maximize overlap with filaments
- Predict galaxy types by node strength:
  - Stronger nodes (larger ΔT) → more U3 galaxies
  - Weaker nodes (smaller ΔT) → more U1 galaxies
  - (without using matter as source)

---

## Input Data

- **Node-CMB Mapping** (from [Step 2.3](../../phase_2_cmb_reconstruction/step_2.3_node_mapping/README.md)):
  - Node positions and CMB mapping from `NodeToCmbMapper`

- **Correlation Results** (from [Step 4.1](../step_4.1_correlation_core/README.md)):
  - CMB-LSS correlation functions

- **LSS Data** (from `data/in/`):
  - Cluster catalogs
  - Filament data
  - LSS structure positions

---

## Algorithm

**Implements Module D from tech_spec-new.md**

### 1. Maximize Overlap with Filaments

- Maximize overlap between node mask and filaments
- Find optimal alignment between nodes and filament structures
- Calculate overlap metrics
- **CRITICAL:** Nodes are topological structures - filaments align with nodes, not vice versa
- Matter does NOT influence Θ-field

### 2. Map Nodes to LSS Structures

- Match node positions to LSS structure positions
- **Node definition:** x_node = {x : ω(x) = ω_min(x)}
- Identify clusters at node locations
- Identify filaments at node locations
- Handle position matching uncertainties
- **CRITICAL:** Galaxies "sit" in nodes, NOT create them

### 3. Predict Galaxy Types by Node Strength

- **Stronger nodes (larger ΔT) → more U3 galaxies:**
  - Identify strong nodes (high ΔT values)
  - Correlate with U3 galaxy distribution
  - Verify prediction WITHOUT using matter as source
  - Document correlation strength
- **Weaker nodes (smaller ΔT) → more U1 galaxies:**
  - Identify weak nodes (low ΔT values)
  - Correlate with U1 galaxy distribution
  - Verify prediction WITHOUT using matter as source
  - Document correlation strength

### 4. Create Node-LSS Catalog

- Create catalog of node-LSS correspondences
- Document structure types at nodes
- Record correlation strengths
- Handle multiple structures per node
- Include galaxy type predictions

### 5. Generate Correlation Maps

- **Create correlation map (карта корреляций):**
  - Maps showing node-LSS correspondence
  - Correlation visualization
  - Highlight correlated regions
- **Create displacement map (карта смещений):**
  - Map of displacements between nodes and LSS structures
  - Visualize alignment patterns
- Document mapping quality

### 6. Analyze Node-LSS Relationships

- Analyze structure types at nodes
- Calculate correlation statistics
- Validate node-LSS correspondence
- Verify galaxies "sit" in nodes (not create them)
- Document relationship patterns

---

## Output Data

### Files Created

1. **`cmb/correlation/node_lss_mapper.py`** - Node-LSS mapping module
   - `NodeLssMapper` class
   - Mapping functions

### Mapping Results

- **Node-LSS Catalog:**
  - Node IDs
  - LSS structure IDs
  - Structure types (cluster, filament, etc.)
  - Correlation strengths
  - Position matches

- **Correlation Maps:**
  - Node-LSS correspondence maps
  - Correlation visualization
  - Mapping quality metrics

---

## Dependencies

- **Step 2.3** (Node Mapping) - For node positions
- **Step 4.1** (Correlation Core) - For correlation functions

---

## Related Steps

- **Step 6.1** (Cluster Plateau) - Will use node-LSS mapping
- **Step 6.2** (Galaxy Distribution) - Will use node directions

---

## Tests

### Unit Tests

1. **Filament Overlap Maximization**
   - **What to test:**
     - Test maximize_filament_overlap():
       - Verify overlap between node mask and filaments is maximized
       - Test optimal alignment is found
       - Verify overlap metrics are calculated correctly
       - Test that nodes are topological structures (filaments align with nodes)
     - Test overlap calculation:
       - Verify overlap is calculated correctly
       - Test alignment optimization
       - Verify no matter influence on Θ-field

2. **Galaxy Type Prediction**
   - **What to test:**
     - Test predict_galaxy_types_by_node_strength():
       - Verify stronger nodes (larger ΔT) correlate with U3 galaxies
       - Verify weaker nodes (smaller ΔT) correlate with U1 galaxies
       - Test prediction is made WITHOUT using matter as source
       - Verify correlation strength calculation
     - Test node strength classification:
       - Verify nodes are classified by strength (ΔT) correctly
       - Test strong/weak node thresholds

3. **Position Matching**
   - **What to test:**
     - Test node-LSS position matching:
       - Verify match_nodes_to_structures() matches nodes to LSS structures correctly
       - Test node definition: x_node = {x : ω(x) = ω_min(x)}
       - Test search_radius parameter affects matching
       - Verify matching uses sky coordinates (theta, phi)
       - Verify galaxies "sit" in nodes (not create them)
     - Test matching uncertainty handling:
       - Verify matching handles position uncertainties correctly
       - Test uncertainty propagation
     - Test multiple structure matching:
       - Verify nodes can match to multiple LSS structures (if applicable)
       - Test best match selection

4. **Catalog Creation**
   - **What to test:**
     - Test catalog generation:
       - Verify NodeLssMapping structure is created correctly:
         - node_ids, structure_ids, structure_types
         - correlation_strengths, position_matches
       - Test all fields are populated correctly
     - Test data integrity:
       - Verify all arrays have consistent lengths
       - Verify node_ids match input node data
       - Verify structure_ids match LSS data
     - Test structure type identification:
       - Verify structure types are identified correctly (cluster, filament, void, etc.)
       - Test type classification accuracy

5. **Map Generation**
   - **What to test:**
     - Test correlation map creation (карта корреляций):
       - Verify create_correlation_map() creates HEALPix correlation map
       - Test map shows node-LSS correlation pattern
       - Verify map has correct NSIDE
     - Test displacement map creation (карта смещений):
       - Verify create_displacement_map() creates displacement map
       - Test map shows displacements between nodes and LSS structures
       - Verify map format is correct
     - Test visualization:
       - Verify correlation maps can be visualized
       - Verify displacement maps can be visualized
       - Test visualization shows node-LSS relationships
     - Test mapping quality metrics:
       - Verify mapping quality is assessed correctly
       - Test quality metrics calculation

6. **Relationship Analysis**
   - **What to test:**
     - Test structure type analysis:
       - Verify analyze_node_lss_relationships() analyzes relationships correctly
       - Test structure type distribution analysis
     - Test correlation statistics:
       - Verify correlation statistics are calculated correctly
       - Test statistics match NodeLssMapping data
     - Test correspondence validation:
       - Verify node-LSS correspondences are validated
       - Test validation criteria
       - Verify galaxies "sit" in nodes (not create them)
       - Verify matter does NOT influence Θ-field

### Integration Tests

1. **End-to-End Node-LSS Mapping**
   - **What to test:**
     - Load nodes → Maximize filament overlap → Predict galaxy types → Match to LSS → Create catalog → Analyze:
       - Load NodeCmbMapping (from Step 2.3)
       - Load node mask (from Step 1.4)
       - Load LSS structure data (SDSS / DESI / Euclid)
       - Load filament data
       - Maximize overlap with filaments
       - Predict galaxy types by node strength (stronger → U3, weaker → U1)
       - Match nodes to LSS structures
       - Create NodeLssMapping catalog
       - Create correlation map and displacement map
       - Analyze node-LSS relationships
       - Verify all steps complete successfully
       - Verify matter does NOT influence Θ-field
     - Test with actual node and LSS data:
       - Real node-CMB mapping
       - Real LSS structure data
       - Verify nodes are correctly matched to LSS structures
       - Verify correlation patterns match theoretical predictions

---

## Implementation Notes

- Handle position matching uncertainties properly
- Support multiple LSS structure types
- Provide clear mapping documentation
- Use efficient spatial matching algorithms

## Forbidden Elements

**DO NOT USE:**
- ❌ Matter as source of nodes (nodes are topological structures of Θ-field)
- ❌ Baryon density, DM density in node-LSS correlation
- ❌ Modification of Θ-field based on observed matter
- ❌ T_μν of matter in Θ-field equations
- ❌ Using matter to predict node positions (nodes predict matter, not vice versa)
- ❌ Baryon acoustic oscillations as physical mechanism
- ❌ FRW metric as primary model
- ❌ Mass, dark matter, or gravitational sources
- ❌ Reverse reaction of matter on Θ-field
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)
- ❌ Classical gravitational correlation models

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

