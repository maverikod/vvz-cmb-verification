# Step 4.3: Node-LSS Mapping

**Phase:** 4 - CMB-LSS Correlation  
**Step:** 4.3 - Node-LSS Mapping  
**Module:** `cmb/correlation/`

---

## Overview

This step maps CMB nodes to LSS structures. It identifies clusters and filaments at node positions and creates correlation maps showing the node-LSS correspondence.

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

### 1. Map Nodes to LSS Structures

- Match node positions to LSS structure positions
- Identify clusters at node locations
- Identify filaments at node locations
- Handle position matching uncertainties

### 2. Create Node-LSS Catalog

- Create catalog of node-LSS correspondences
- Document structure types at nodes
- Record correlation strengths
- Handle multiple structures per node

### 3. Generate Correlation Maps

- Create maps showing node-LSS correspondence
- Visualize correlation patterns
- Highlight correlated regions
- Document mapping quality

### 4. Analyze Node-LSS Relationships

- Analyze structure types at nodes
- Calculate correlation statistics
- Validate node-LSS correspondence
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

1. **Position Matching**
   - Test node-LSS position matching
   - Test matching uncertainty handling
   - Test multiple structure matching

2. **Catalog Creation**
   - Test catalog generation
   - Test data integrity
   - Test structure type identification

3. **Map Generation**
   - Test correlation map creation
   - Test visualization
   - Test mapping quality metrics

4. **Relationship Analysis**
   - Test structure type analysis
   - Test correlation statistics
   - Test correspondence validation

### Integration Tests

1. **End-to-End Node-LSS Mapping**
   - Load nodes → Match to LSS → Create catalog → Analyze
   - Test with actual node and LSS data

---

## Implementation Notes

- Handle position matching uncertainties properly
- Support multiple LSS structure types
- Provide clear mapping documentation
- Use efficient spatial matching algorithms

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

