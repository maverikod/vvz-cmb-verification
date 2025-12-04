# Implementation Plan: CMB Verification Project

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

---

## Overview

This document provides a step-by-step implementation plan for the CMB verification project based on the theoretical framework and technical specification. The plan considers dependencies between modules and follows the project structure rules.

---

## Project Goals

Based on the technical specification (tech_spec-new.md), the project must implement:

1. **CMB map reconstruction** from Θ-field frequency spectrum (Module A)
2. **C_l power spectrum generation** without Silk damping (Module B)
3. **Θ-node map generation** from field ω(x) (Module C - NEW)
4. **CMB → LSS correlation** with galaxy type prediction (Module D)
5. **ACT/SPT predictions** (high-l peaks, frequency invariance)
6. **CMB → LSS → clusters → galaxies chain verification**

**CRITICAL PRINCIPLE:** Matter does NOT influence Θ-field. Matter = projection/envelope of Θ-modes.

---

## Dependencies Analysis

### External Dependencies
- **Θ-field model data** (`data/theta/`) - contains:
  - Frequency spectrum ρ_Θ(ω,t) data
  - Evolution ω_min(t), ω_macro(t) data
  - Θ-node geometry and structure data
  - Node depth calculations data
- **Note:** Θ-field model is data, not a code module. Processing code goes in `cmb/` module.

### Data Dependencies
- **ACT DR6.02 data** (`data/in/dr6.02/`) - CMB maps and spectra
- **LSS correlation data** (`data/in/`) - for cross-correlation analysis
- **Galaxy data** (`data/in/`) - for chain verification
- **Θ-field data** (`data/theta/`) - Θ-field model data files

### Internal Module Dependencies
```
data/theta/ → cmb/reconstruction → cmb/spectrum → cmb/correlation → cmb/predictions
     ↓              ↓                    ↓                ↓
utils/io        utils/math        utils/visualization   cmb/nodes
```

---

## Implementation Phases

### Phase 0: Foundation and Infrastructure

**Goal:** Set up basic infrastructure, utilities, and configuration

#### Step 0.1: Configuration Management
- **Module:** `config/`
- **Files:**
  - `config/settings.py` - Main configuration
  - `config/cmb_config.yaml` - CMB-specific settings
- **Tasks:**
  - Define constants (D, conversion factors, z_CMB ≈ 1100)
  - Configure data paths
  - Set up logging
- **Dependencies:** None
- **Output:** Configuration files

#### Step 0.2: Utility Modules
- **Module:** `utils/`
- **Files:**
  - `utils/io/data_loader.py` - Load ACT data, FITS files
  - `utils/io/data_saver.py` - Save results
  - `utils/math/frequency_conversion.py` - l ≈ πDω conversion
  - `utils/math/spherical_harmonics.py` - Spherical harmonics operations
  - `utils/visualization/cmb_plots.py` - CMB visualization
  - `utils/visualization/spectrum_plots.py` - Power spectrum plots
- **Tasks:**
  - Implement HEALPix map loading
  - Implement frequency-to-multipole conversion
  - Implement spherical harmonics utilities
  - Create visualization functions
- **Dependencies:** None (external libraries: healpy, numpy, matplotlib)
- **Output:** Utility functions ready for use

#### Step 0.3: Data Index Integration
- **Module:** `utils/io/`
- **Files:**
  - `utils/io/data_index_loader.py` - Load and query data_index.yaml
- **Tasks:**
  - Parse data_index.yaml
  - Provide query interface for data files
  - Validate data file existence
- **Dependencies:** None
- **Output:** Data index query interface

---

### Phase 1: Θ-Field Data Processing

**Goal:** Implement data loading and processing for Θ-field model data

#### Step 1.1: Θ-Field Data Loader
- **Module:** `cmb/` (Θ-field processing is part of CMB module)
- **Files:**
  - `cmb/theta_data_loader.py` - Load Θ-field data from `data/theta/`
- **Tasks:**
  - Load frequency spectrum ρ_Θ(ω,t) from data files
  - Load temporal evolution data
  - Provide interface for CMB calculations
- **Dependencies:** `utils/io/data_loader.py`
- **Output:** Θ-field data loading interface

#### Step 1.2: Θ-Node Data Processing
- **Module:** `cmb/`
- **Files:**
  - `cmb/theta_node_processor.py` - Process Θ-node data
- **Tasks:**
  - Load node geometry data (scale ~300 pc at z≈1100)
  - Process node depth data (Δω/ω)
  - Map depth to ΔT (20-30 μK)
- **Dependencies:** `cmb/theta_data_loader.py`
- **Output:** Node structure processing

#### Step 1.3: Θ-Field Evolution Data
- **Module:** `cmb/`
- **Files:**
  - `cmb/theta_evolution_processor.py` - Process evolution data
- **Tasks:**
  - Load ω_min(t), ω_macro(t) data
  - Process time-dependent parameters
- **Dependencies:** `cmb/theta_data_loader.py`
- **Output:** Evolution data processing

#### Step 1.4: Θ-Node Map Generation (NEW - Module C)
- **Module:** `cmb/nodes/`
- **Files:**
  - `cmb/nodes/theta_node_map.py` - Generate Θ-node map
- **Tasks:**
  - Find nodes as points of local minima: x_node = {x : ω(x) = ω_min(x)}
  - Classify nodes by depth, area, and local curvature
  - Create map ω_min(x) and node mask
- **Dependencies:** `cmb/theta_data_loader.py`
- **Output:** Θ-node map and node mask

---

### Phase 2: CMB Map Reconstruction

**Goal:** Reconstruct CMB temperature map from Θ-field

#### Step 2.1: CMB Reconstruction Core (Module A)
- **Module:** `cmb/reconstruction/`
- **Files:**
  - `cmb/reconstruction/cmb_map_reconstructor.py` - Main reconstruction class
- **Tasks:**
  - Implement Formula 2.1 (from tech_spec-new.md) for map reconstruction
  - For each direction n̂: compute node depth Δω(n̂) = ω(n̂) - ω_min(n̂)
  - Apply formula ΔT = (Δω/ω_CMB) T_0
  - Generate spherical harmonic map ΔT(n̂)
- **Dependencies:**
  - `cmb/theta_node_processor.py`
  - `cmb/theta_data_loader.py`
  - `cmb/nodes/theta_node_map.py` (node mask from Step 1.4)
  - `utils/math/spherical_harmonics.py`
  - `utils/math/frequency_conversion.py`
- **Output:** Reconstructed CMB map (HEALPix format, Nside≥2048)

#### Step 2.2: CMB Map Validation
- **Module:** `cmb/reconstruction/`
- **Files:**
  - `cmb/reconstruction/map_validator.py` - Validate reconstructed maps
- **Tasks:**
  - Compare with ACT DR6.02 maps
  - Validate arcmin-scale structures (2-5′)
  - Check amplitude (20-30 μK)
- **Dependencies:**
  - `cmb/reconstruction/cmb_map_reconstructor.py`
  - `utils/io/data_loader.py`
- **Output:** Validation results

#### Step 2.3: Node-to-Map Mapping
- **Module:** `cmb/nodes/`
- **Files:**
  - `cmb/nodes/node_to_cmb_mapper.py` - Map Θ-nodes to CMB positions
- **Tasks:**
  - Map Θ-nodes to sky coordinates
  - Handle projection from early universe (z≈1100)
  - Create node catalog with CMB positions
- **Dependencies:**
  - `cmb/theta_node_processor.py`
  - `cmb/reconstruction/cmb_map_reconstructor.py`
- **Output:** Node-CMB position mapping

---

### Phase 3: Power Spectrum Generation

**Goal:** Generate C_l power spectrum DIRECTLY from Θ-field frequency spectrum

**CRITICAL:** C_l is calculated directly from ρ_Θ(ω,t), NOT from reconstructed map.
This is the fundamental difference from classical cosmology.

#### Step 3.1: Power Spectrum Calculation (Module B)
- **Module:** `cmb/spectrum/`
- **Files:**
  - `cmb/spectrum/power_spectrum.py` - C_l calculation
- **Tasks:**
  - Calculate C_l DIRECTLY from ρ_Θ(ω,t) using formulas from tech_spec-new.md
  - Use: l(ω) ≈ π D ω (Formula 2.2)
  - Use: C_l ∝ ρ_Θ(ω(ℓ)) / ℓ² (Formula 2.3)
  - Result: C_l ∝ l⁻² (Formula 2.4)
  - Handle high-l range (up to l≈10000)
  - No Silk damping implementation
  - No plasma terms, no sound horizons
  - Spherical Harmonic Decomposition (for Module B)
  - Extract sub-peaks according to ρ_Θ(ω) structure
- **Dependencies:**
  - `cmb/theta_data_loader.py` (for ρ_Θ(ω,t))
  - `cmb/theta_evolution_processor.py` (for temporal evolution)
  - `utils/math/frequency_conversion.py` (for l = π D ω)
- **Output:** C_l power spectrum array

#### Step 3.2: High-l Sub-peaks Analysis
- **Module:** `cmb/spectrum/`
- **Files:**
  - `cmb/spectrum/subpeaks_analyzer.py` - Analyze sub-peaks
- **Tasks:**
  - Identify sub-peaks from ω_min beatings
  - Detect l≈4500-6000 peak
  - Analyze sub-structure
- **Dependencies:**
  - `cmb/spectrum/power_spectrum.py`
  - `cmb/theta_evolution_processor.py`
- **Output:** Sub-peak analysis results

#### Step 3.3: Spectrum Comparison
- **Module:** `cmb/spectrum/`
- **Files:**
  - `cmb/spectrum/spectrum_comparator.py` - Compare with observations
- **Tasks:**
  - Load ACT DR6.02 spectra
  - Compare reconstructed vs observed
  - Validate high-l tail (l>2000)
- **Dependencies:**
  - `cmb/spectrum/power_spectrum.py`
  - `utils/io/data_loader.py`
- **Output:** Comparison results

---

### Phase 4: CMB-LSS Correlation

**Goal:** Analyze correlation between CMB and Large Scale Structure

#### Step 4.1: Correlation Analysis Core
- **Module:** `cmb/correlation/`
- **Files:**
  - `cmb/correlation/cmb_lss_correlator.py` - Main correlation analysis
- **Tasks:**
  - Implement CMB ↔ LSS correlation tests
  - Calculate correlation functions
  - Handle 10-12 Mpc scales
- **Dependencies:**
  - `cmb/reconstruction/cmb_map_reconstructor.py`
  - `utils/io/data_loader.py` (LSS data)
- **Output:** Correlation analysis results

#### Step 4.2: Phi-Split Analysis
- **Module:** `cmb/correlation/`
- **Files:**
  - `cmb/correlation/phi_split_analyzer.py` - φ-split analysis
- **Tasks:**
  - Implement φ-split technique
  - Analyze signal enhancement
  - Validate predictions
- **Dependencies:**
  - `cmb/correlation/cmb_lss_correlator.py`
- **Output:** φ-split results

#### Step 4.3: Node-LSS Mapping (Module D)
- **Module:** `cmb/correlation/`
- **Files:**
  - `cmb/correlation/node_lss_mapper.py` - Map nodes to LSS
- **Tasks:**
  - **Maximize overlap with filaments** (Module D requirement)
  - **Predict galaxy types by node strength:**
    - Stronger nodes (larger ΔT) → more U3 galaxies
    - Weaker nodes (smaller ΔT) → more U1 galaxies
    - (without using matter as source)
  - Map CMB nodes to LSS structures (SDSS / DESI / Euclid)
  - Identify clusters/filaments at node positions
  - Create correlation map (карта корреляций)
  - Create displacement map (карта смещений)
- **Dependencies:**
  - `cmb/nodes/node_to_cmb_mapper.py`
  - `cmb/nodes/theta_node_map.py` (node mask from Step 1.4)
  - `cmb/correlation/cmb_lss_correlator.py`
- **Output:** Node-LSS mapping with galaxy type predictions

---

### Phase 5: ACT/SPT Predictions

**Goal:** Generate and validate predictions for ACT/SPT/Simons Observatory

#### Step 5.1: High-l Peak Prediction
- **Module:** `cmb/predictions/`
- **Files:**
  - `cmb/predictions/highl_peak_predictor.py` - Predict l≈4500-6000 peak
- **Tasks:**
  - Calculate predicted peak position
  - Estimate amplitude
  - Compare with ACT/SPT data
- **Dependencies:**
  - `cmb/spectrum/subpeaks_analyzer.py`
  - `cmb/theta_evolution_processor.py`
- **Output:** Peak predictions

#### Step 5.2: Frequency Invariance Test
- **Module:** `cmb/predictions/`
- **Files:**
  - `cmb/predictions/frequency_invariance.py` - Test achromaticity
- **Tasks:**
  - Test cross-spectra at 90-350 GHz
  - Validate frequency invariance
  - Compare with observations
- **Dependencies:**
  - `cmb/reconstruction/cmb_map_reconstructor.py`
  - `utils/io/data_loader.py` (multi-frequency data)
- **Output:** Frequency invariance validation

#### Step 5.3: Predictions Report
- **Module:** `cmb/predictions/`
- **Files:**
  - `cmb/predictions/predictions_report.py` - Generate predictions report
- **Tasks:**
  - Compile all predictions
  - Compare with observations
  - Generate validation report
- **Dependencies:**
  - All prediction modules
- **Output:** Comprehensive predictions report

---

### Phase 6: Chain Verification

**Goal:** Verify the chain CMB → LSS → clusters → galaxies

#### Step 6.1: Cluster Plateau Analysis
- **Module:** `cmb/correlation/`
- **Files:**
  - `cmb/correlation/cluster_plateau_analyzer.py` - Analyze cluster plateaus
- **Tasks:**
  - Load cluster data
  - Analyze plateau slopes
  - Correlate with CMB node directions
- **Dependencies:**
  - `cmb/nodes/node_to_cmb_mapper.py`
  - `utils/io/data_loader.py` (cluster data)
- **Output:** Cluster plateau analysis

#### Step 6.2: Galaxy Distribution Analysis
- **Module:** `cmb/correlation/`
- **Files:**
  - `cmb/correlation/galaxy_distribution_analyzer.py` - Analyze galaxy distributions
- **Tasks:**
  - Analyze U1/U2/U3 distributions
  - Calculate SWI, χ₆ by node directions
  - Verify chain connection
- **Dependencies:**
  - `cmb/correlation/cluster_plateau_analyzer.py`
  - `utils/io/data_loader.py` (galaxy data)
- **Output:** Galaxy distribution analysis

#### Step 6.3: Chain Verification Report
- **Module:** `cmb/correlation/`
- **Files:**
  - `cmb/correlation/chain_verifier.py` - Verify complete chain
- **Tasks:**
  - Compile all chain verification results
  - Validate CMB → LSS → clusters → galaxies connection
  - Generate final verification report
- **Dependencies:**
  - All correlation modules
- **Output:** Chain verification report

---

## Implementation Order (Critical Path)

### Must Complete First (Foundation):
1. Phase 0: Foundation and Infrastructure
2. Phase 1: Θ-Field Model Foundation

### Can Proceed in Parallel After Foundation:
3. Phase 2: CMB Map Reconstruction (depends on Phase 1)
4. Phase 3: Power Spectrum Generation (depends on Phase 1, NOT Phase 2)
5. Phase 4: CMB-LSS Correlation (depends on Phase 2 and Phase 1.4)

### Final Steps:
6. Phase 5: ACT/SPT Predictions (depends on Phases 3, 4)
7. Phase 6: Chain Verification (depends on Phase 4)

---

## File Structure Summary

```
cmb/
├── reconstruction/
│   ├── cmb_map_reconstructor.py
│   ├── map_validator.py
│   └── __init__.py
├── spectrum/
│   ├── power_spectrum.py
│   ├── subpeaks_analyzer.py
│   ├── spectrum_comparator.py
│   └── __init__.py
├── nodes/
│   ├── node_to_cmb_mapper.py
│   ├── theta_node_map.py
│   └── __init__.py
├── correlation/
│   ├── cmb_lss_correlator.py
│   ├── phi_split_analyzer.py
│   ├── node_lss_mapper.py
│   ├── cluster_plateau_analyzer.py
│   ├── galaxy_distribution_analyzer.py
│   ├── chain_verifier.py
│   └── __init__.py
├── predictions/
│   ├── highl_peak_predictor.py
│   ├── frequency_invariance.py
│   ├── predictions_report.py
│   └── __init__.py
├── theta_data_loader.py
├── theta_node_processor.py
├── theta_evolution_processor.py
└── __init__.py

**Note:** Step 1.4 (Node Map Generation) creates `cmb/nodes/theta_node_map.py`

utils/
├── io/
│   ├── data_loader.py
│   ├── data_saver.py
│   ├── data_index_loader.py
│   └── __init__.py
├── math/
│   ├── frequency_conversion.py
│   ├── spherical_harmonics.py
│   └── __init__.py
├── visualization/
│   ├── cmb_plots.py
│   ├── spectrum_plots.py
│   └── __init__.py
└── __init__.py

config/
├── settings.py
├── cmb_config.yaml
└── __init__.py
```

---

## Testing Strategy

### Unit Tests
- Each module should have corresponding tests in `tests/unit/`
- Test individual functions and classes
- Mock dependencies where appropriate

### Integration Tests
- Test module interactions
- Test data pipeline flows
- Validate end-to-end workflows

### Validation Tests
- Compare with observational data
- Validate theoretical predictions
- Check numerical accuracy

---

## Documentation Requirements

- Each module must have comprehensive docstrings
- API documentation in `docs/api/`
- Usage examples for each major component
- Theory references for key formulas

---

## Success Criteria

Based on tech_spec-new.md requirements:

1. ✅ Reconstructed CMB map matches ACT DR6.02 observations (arcmin scale, 20-30 μK)
2. ✅ Generated C_l spectrum shows l⁻²–l⁻³ behavior without Silk damping
3. ✅ High-l peak detected at l≈4500-6000
4. ✅ CMB-LSS correlation validated at 10-12 Mpc scales
5. ✅ Frequency invariance confirmed (90-350 GHz)
6. ✅ Chain CMB → LSS → clusters → galaxies verified
7. ✅ **Θ-node map generated** (ω_min(x) map and node mask) - Module C
8. ✅ **Filament overlap maximized** - Module D
9. ✅ **Galaxy type prediction validated** (stronger nodes → U3, weaker → U1) - Module D
10. ✅ **Principle verified:** Matter does NOT influence Θ-field

---

**Last Updated:** 2024-12-03

**Updated based on:** tech_spec-new.md (see PLAN_UPDATES_FROM_NEW_SPEC.md for details)

