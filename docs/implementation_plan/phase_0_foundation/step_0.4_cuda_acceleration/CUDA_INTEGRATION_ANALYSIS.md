# CUDA Integration Analysis for Implementation Plan

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

---

## Overview

This document provides detailed analysis of each phase and step in the implementation plan, identifying specific opportunities for CUDA acceleration and recommending appropriate vectorizer types and implementation strategies.

---

## Phase 0: Foundation and Infrastructure

### Step 0.1: Configuration Management
**CUDA Integration:** None (configuration only)

### Step 0.2: Utility Modules
**CUDA Integration:** Medium priority

#### `frequency_conversion.py`
- **Operation:** `l = π * D * ω`
- **Vectorizer Type:** `ElementWiseVectorizer`
- **Implementation:**
  ```python
  def frequency_to_multipole_cuda(frequency: CudaArray, D: float) -> CudaArray:
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      return vectorizer.vectorize_operation(
          frequency, "multiply", np.pi * D
      )
  ```
- **Benefit:** High - used frequently throughout project

#### `spherical_harmonics.py`
- **Operation:** Spherical harmonic transforms
- **Vectorizer Type:** `TransformVectorizer`
- **Implementation:**
  ```python
  def sph_harm_synthesis_cuda(alm: CudaArray, lmax: int) -> CudaArray:
      vectorizer = TransformVectorizer(use_gpu=True, whole_array=True)
      return vectorizer.vectorize_transform(
          alm, "sph_harm_synthesis", lmax=lmax
      )
  ```
- **Benefit:** Very High - critical for map reconstruction

### Step 0.3: Data Index Integration
**CUDA Integration:** None (data management only)

---

## Phase 1: Θ-Field Data Processing

### Step 1.1: Θ-Field Data Loader
**CUDA Integration:** Low priority

**Operations:**
- Data loading (I/O bound, no CUDA benefit)
- Initial validation (minimal computation)

**Recommendation:** Load data into `CudaArray` for later processing, but no CUDA acceleration needed for loading itself.

### Step 1.2: Θ-Node Data Processing
**CUDA Integration:** High priority

#### Node Depth Calculation
- **Operation:** `Δω = ω - ω_min` for all nodes
- **Vectorizer Type:** `ElementWiseVectorizer`
- **Implementation:**
  ```python
  def calculate_node_depth_cuda(omega: CudaArray, omega_min: CudaArray) -> CudaArray:
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      return vectorizer.vectorize_operation(omega, "subtract", omega_min)
  ```
- **Benefit:** High - processes all nodes simultaneously

#### Temperature Mapping
- **Operation:** `ΔT = (Δω/ω_CMB) * T_0` for all nodes
- **Vectorizer Type:** `ElementWiseVectorizer`
- **Implementation:**
  ```python
  def map_depth_to_temperature_cuda(depth: CudaArray, omega_cmb: float, T_0: float) -> CudaArray:
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      factor = T_0 / omega_cmb
      return vectorizer.vectorize_operation(depth, "multiply", factor)
  ```
- **Benefit:** High - element-wise operation on large arrays

### Step 1.3: Θ-Field Evolution Data
**CUDA Integration:** Medium priority

#### Temporal Evolution Interpolation
- **Operation:** Interpolation over time arrays
- **Vectorizer Type:** `ElementWiseVectorizer` (for interpolation weights)
- **Implementation:**
  ```python
  def interpolate_evolution_cuda(times: CudaArray, values: CudaArray, new_times: CudaArray) -> CudaArray:
      # Use linear interpolation with vectorized operations
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      # Find interpolation indices and weights
      # Apply interpolation using element-wise operations
  ```
- **Benefit:** Medium - interpolation can be vectorized

#### Evolution Rate Calculation
- **Operation:** Numerical derivatives `dω/dt`
- **Vectorizer Type:** `ElementWiseVectorizer`
- **Implementation:**
  ```python
  def calculate_evolution_rate_cuda(omega: CudaArray, times: CudaArray) -> CudaArray:
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      # Use central differences: dω/dt = (ω[i+1] - ω[i-1]) / (t[i+1] - t[i-1])
      # Implement using element-wise operations
  ```
- **Benefit:** Medium - derivative calculation is vectorizable

### Step 1.4: Θ-Node Map Generation
**CUDA Integration:** Very High priority

#### Local Minimum Detection
- **Operation:** Find `x_node = {x : ω(x) = ω_min(x)}`
- **Vectorizer Type:** `GridVectorizer`
- **Implementation:**
  ```python
  def find_local_minima_cuda(omega_field: CudaArray, neighborhood_size: int = 5) -> CudaArray:
      grid_vectorizer = GridVectorizer(use_gpu=True)
      return grid_vectorizer.vectorize_grid_operation(
          omega_field, "local_minima", neighborhood_size=neighborhood_size
      )
  ```
- **Benefit:** Very High - grid operations are highly parallelizable

#### Node Classification
- **Depth Classification:**
  - **Operation:** Classify nodes by depth ranges
  - **Vectorizer Type:** `ElementWiseVectorizer` (for depth calculation)
  - **Benefit:** High

- **Area Classification:**
  - **Operation:** Calculate node area (spatial extent)
  - **Vectorizer Type:** `GridVectorizer` (for area calculation)
  - **Benefit:** High

- **Curvature Classification:**
  - **Operation:** Calculate local curvature
  - **Vectorizer Type:** `GridVectorizer`
  - **Implementation:**
    ```python
    def calculate_curvature_cuda(omega_field: CudaArray) -> CudaArray:
        grid_vectorizer = GridVectorizer(use_gpu=True)
        return grid_vectorizer.vectorize_grid_operation(
            omega_field, "curvature"
        )
    ```
  - **Benefit:** High

---

## Phase 2: CMB Map Reconstruction

### Step 2.1: CMB Reconstruction Core
**CUDA Integration:** Very High priority (critical path)

#### Frequency Spectrum Integration
- **Operation:** Integrate `ρ_Θ(ω,t) ∝ ω^(-2±0.5) · (t/t_0)^(-3±1)` for each direction
- **Vectorizer Type:** `ElementWiseVectorizer`
- **Implementation:**
  ```python
  def integrate_frequency_spectrum_cuda(
      omega: CudaArray, 
      time: CudaArray, 
      t_0: float,
      alpha: float = -2.0,
      beta: float = -3.0
  ) -> CudaArray:
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      # Calculate: omega^alpha * (time/t_0)^beta
      omega_factor = vectorizer.vectorize_operation(omega, "power", alpha)
      time_factor = vectorizer.vectorize_operation(
          vectorizer.vectorize_operation(time, "divide", t_0),
          "power", beta
      )
      return vectorizer.vectorize_operation(omega_factor, "multiply", time_factor)
  ```
- **Benefit:** Very High - processes all directions simultaneously

#### Depth Calculation for Each Direction
- **Operation:** `Δω(n̂) = ω(n̂) - ω_min(n̂)` for all directions n̂
- **Vectorizer Type:** `ElementWiseVectorizer`
- **Implementation:**
  ```python
  def calculate_depth_per_direction_cuda(
      omega_directions: CudaArray,
      omega_min_directions: CudaArray
  ) -> CudaArray:
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      return vectorizer.vectorize_operation(
          omega_directions, "subtract", omega_min_directions
      )
  ```
- **Benefit:** Very High - processes all directions (Nside≥2048 = millions of pixels)

#### Temperature Conversion
- **Operation:** `ΔT = (Δω/ω_CMB) * T_0` for all directions
- **Vectorizer Type:** `ElementWiseVectorizer`
- **Benefit:** Very High - same as Step 1.2 but for full sky map

#### Spherical Harmonic Synthesis
- **Operation:** Generate HEALPix map from spherical harmonic coefficients
- **Vectorizer Type:** `TransformVectorizer`
- **Implementation:**
  ```python
  def synthesize_cmb_map_cuda(alm: CudaArray, lmax: int, nside: int) -> CudaArray:
      transform_vectorizer = TransformVectorizer(use_gpu=True, whole_array=True)
      return transform_vectorizer.vectorize_transform(
          alm, "sph_harm_synthesis", lmax=lmax, nside=nside
      )
  ```
- **Benefit:** Very High - critical bottleneck, FFT-like operation

### Step 2.2: CMB Map Validation
**CUDA Integration:** Medium priority

#### Map Comparison
- **Operation:** Compare reconstructed vs observed maps
- **Vectorizer Type:** `ElementWiseVectorizer`
- **Implementation:**
  ```python
  def compare_maps_cuda(map1: CudaArray, map2: CudaArray) -> Dict:
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      diff = vectorizer.vectorize_operation(map1, "subtract", map2)
      reduction = ReductionVectorizer(use_gpu=True)
      mse = reduction.vectorize_reduction(diff ** 2, "mean")
      return {"mse": mse, "diff_map": diff}
  ```
- **Benefit:** Medium - useful for validation but not critical path

### Step 2.3: Node-to-Map Mapping
**CUDA Integration:** Medium priority

#### Coordinate Transformation
- **Operation:** Project nodes from z≈1100 to z=0
- **Vectorizer Type:** `ElementWiseVectorizer`
- **Benefit:** Medium - coordinate transformations are vectorizable

---

## Phase 3: Power Spectrum Generation

### Step 3.1: Power Spectrum Calculation
**CUDA Integration:** Very High priority

#### Frequency to Multipole Conversion
- **Operation:** `l = π * D * ω` for large frequency arrays
- **Vectorizer Type:** `ElementWiseVectorizer`
- **Implementation:**
  ```python
  def frequency_to_multipole_cuda(frequency: CudaArray, D: float) -> CudaArray:
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      return vectorizer.vectorize_operation(
          frequency, "multiply", np.pi * D
      )
  ```
- **Benefit:** Very High - processes up to l≈10000 (large arrays)

#### Direct C_l Calculation
- **Operation:** `C_l ∝ ρ_Θ(ω(ℓ)) / ℓ²`
- **Vectorizer Type:** `ElementWiseVectorizer`
- **Implementation:**
  ```python
  def calculate_cl_from_spectrum_cuda(
      rho_theta: CudaArray,
      multipole: CudaArray
  ) -> CudaArray:
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      # Calculate ℓ²
      l_squared = vectorizer.vectorize_operation(multipole, "power", 2)
      # Calculate C_l = ρ_Θ / ℓ²
      cl = vectorizer.vectorize_operation(rho_theta, "divide", l_squared)
      return cl
  ```
- **Benefit:** Very High - processes large multipole arrays

#### Temporal Evolution Integration
- **Operation:** Integrate over time for each frequency
- **Vectorizer Type:** `ReductionVectorizer` (along time axis)
- **Benefit:** High - reduction operations are efficient on GPU

### Step 3.2: High-l Sub-peaks Analysis
**CUDA Integration:** High priority

#### Peak Detection
- **Operation:** Find peaks in power spectrum
- **Vectorizer Type:** `GridVectorizer` (for 1D peak detection)
- **Implementation:**
  ```python
  def find_peaks_cuda(spectrum: CudaArray, threshold: float) -> CudaArray:
      grid_vectorizer = GridVectorizer(use_gpu=True)
      # Use local maximum detection
      peaks = grid_vectorizer.vectorize_grid_operation(
          spectrum, "local_maxima", neighborhood_size=5
      )
      # Filter by threshold
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      above_threshold = vectorizer.vectorize_operation(
          spectrum, "greater", threshold
      )
      return peaks & above_threshold
  ```
- **Benefit:** High - peak detection is parallelizable

### Step 3.3: Spectrum Comparison
**CUDA Integration:** Medium priority

#### Spectrum Comparison
- **Operation:** Compare calculated vs observed spectra
- **Vectorizer Type:** `ElementWiseVectorizer` (for difference calculation)
- **Benefit:** Medium - similar to map comparison

---

## Phase 4: CMB-LSS Correlation

### Step 4.1: Correlation Analysis Core
**CUDA Integration:** Very High priority

#### Cross-Correlation Calculation
- **Operation:** Cross-correlation between CMB and LSS maps
- **Vectorizer Type:** `CorrelationVectorizer`
- **Implementation:**
  ```python
  def calculate_correlation_cuda(
      cmb_map: CudaArray,
      lss_map: CudaArray,
      method: str = "fft"
  ) -> CudaArray:
      correlation_vectorizer = CorrelationVectorizer(use_gpu=True)
      return correlation_vectorizer.vectorize_correlation(
          cmb_map, lss_map, method=method
      )
  ```
- **Benefit:** Very High - correlation is computationally expensive, FFT-based method is highly parallelizable

#### Correlation Function at Scales
- **Operation:** Calculate correlation at 10-12 Mpc scales
- **Vectorizer Type:** `CorrelationVectorizer`
- **Benefit:** Very High - multiple scale calculations can be batched

### Step 4.2: Phi-Split Analysis
**CUDA Integration:** Medium priority

#### Signal Enhancement
- **Operation:** φ-split technique calculations
- **Vectorizer Type:** `ElementWiseVectorizer` (for signal processing)
- **Benefit:** Medium - signal processing operations are vectorizable

### Step 4.3: Node-LSS Mapping
**CUDA Integration:** High priority

#### Node-to-LSS Position Mapping
- **Operation:** Map nodes to LSS positions
- **Vectorizer Type:** `ElementWiseVectorizer` (for coordinate transformations)
- **Benefit:** High - processes all nodes simultaneously

#### Galaxy Type Prediction
- **Operation:** Predict galaxy types based on node strength
- **Vectorizer Type:** `ElementWiseVectorizer` (for classification)
- **Implementation:**
  ```python
  def predict_galaxy_types_cuda(node_strength: CudaArray, thresholds: Dict) -> CudaArray:
      vectorizer = ElementWiseVectorizer(use_gpu=True)
      # Classify based on thresholds
      # Stronger nodes → U3, weaker → U1
      u3_mask = vectorizer.vectorize_operation(
          node_strength, "greater", thresholds["u3"]
      )
      u1_mask = vectorizer.vectorize_operation(
          node_strength, "less", thresholds["u1"]
      )
      return {"u3": u3_mask, "u1": u1_mask}
  ```
- **Benefit:** High - classification is vectorizable

---

## Phase 5: ACT/SPT Predictions

### Step 5.1: High-l Peak Prediction
**CUDA Integration:** Medium priority

#### Peak Position Calculation
- **Operation:** Calculate predicted peak position
- **Vectorizer Type:** `ReductionVectorizer` (for argmax)
- **Benefit:** Medium - peak finding is efficient on GPU

### Step 5.2: Frequency Invariance Test
**CUDA Integration:** High priority

#### Multi-Frequency Cross-Spectra
- **Operation:** Cross-spectra at 90-350 GHz
- **Vectorizer Type:** `CorrelationVectorizer`
- **Implementation:**
  ```python
  def calculate_cross_spectra_cuda(
      maps_90ghz: CudaArray,
      maps_350ghz: CudaArray
  ) -> CudaArray:
      correlation_vectorizer = CorrelationVectorizer(use_gpu=True)
      return correlation_vectorizer.vectorize_correlation(
          maps_90ghz, maps_350ghz, method="fft"
      )
  ```
- **Benefit:** High - multiple frequency pairs can be processed in parallel

### Step 5.3: Predictions Report
**CUDA Integration:** Low priority (report generation)

---

## Phase 6: Chain Verification

### Step 6.1: Cluster Plateau Analysis
**CUDA Integration:** Medium priority

#### Plateau Slope Calculation
- **Operation:** Calculate slopes for cluster plateaus
- **Vectorizer Type:** `ElementWiseVectorizer` (for slope calculation)
- **Benefit:** Medium - slope calculations are vectorizable

### Step 6.2: Galaxy Distribution Analysis
**CUDA Integration:** High priority

#### Distribution Calculations
- **Operation:** Calculate U1/U2/U3 distributions
- **Vectorizer Type:** `ReductionVectorizer` (for counting/statistics)
- **Benefit:** High - distribution calculations are efficient on GPU

#### SWI and χ₆ Calculations
- **Operation:** Calculate SWI, χ₆ by node directions
- **Vectorizer Type:** `ElementWiseVectorizer` + `ReductionVectorizer`
- **Benefit:** High - statistical calculations are vectorizable

### Step 6.3: Chain Verification Report
**CUDA Integration:** Low priority (report generation)

---

## Summary: CUDA Integration Priority by Phase

### Very High Priority (Critical Path)
1. **Phase 2, Step 2.1:** CMB Reconstruction Core
   - Frequency spectrum integration
   - Depth calculation for all directions
   - Temperature conversion
   - Spherical harmonic synthesis

2. **Phase 3, Step 3.1:** Power Spectrum Calculation
   - Frequency to multipole conversion
   - Direct C_l calculation
   - Temporal evolution integration

3. **Phase 4, Step 4.1:** Correlation Analysis Core
   - Cross-correlation calculations
   - Correlation functions at scales

4. **Phase 1, Step 1.4:** Node Map Generation
   - Local minimum detection
   - Node classification

### High Priority
1. **Phase 1, Step 1.2:** Node Processing
2. **Phase 3, Step 3.2:** Sub-peaks Analysis
3. **Phase 4, Step 4.3:** Node-LSS Mapping
4. **Phase 5, Step 5.2:** Frequency Invariance Test
5. **Phase 6, Step 6.2:** Galaxy Distribution Analysis

### Medium Priority
1. **Phase 0, Step 0.2:** Utility Modules
2. **Phase 1, Step 1.3:** Evolution Data
3. **Phase 2, Step 2.2:** Map Validation
4. **Phase 2, Step 2.3:** Node Mapping
5. **Phase 3, Step 3.3:** Spectrum Comparison
6. **Phase 4, Step 4.2:** Phi-Split Analysis
7. **Phase 5, Step 5.1:** High-l Peak Prediction
8. **Phase 6, Step 6.1:** Cluster Plateau Analysis

### Low Priority
1. **Phase 1, Step 1.1:** Data Loader (I/O bound)
2. **Phase 5, Step 5.3:** Predictions Report (report generation)
3. **Phase 6, Step 6.3:** Chain Report (report generation)

---

## Recommended Implementation Order

1. **Step 0.4:** CUDA Infrastructure (this step)
   - Array model
   - Base vectorizer
   - ElementWiseVectorizer (most common)

2. **Step 0.2 (update):** Add CUDA to utilities
   - Frequency conversion
   - Spherical harmonics

3. **Phase 1, Step 1.4:** Node Map Generation
   - GridVectorizer implementation
   - Local minimum detection

4. **Phase 2, Step 2.1:** CMB Reconstruction
   - TransformVectorizer implementation
   - Spherical harmonic synthesis

5. **Phase 3, Step 3.1:** Power Spectrum
   - Large-scale element-wise operations

6. **Phase 4, Step 4.1:** Correlation Analysis
   - CorrelationVectorizer implementation
   - FFT-based correlation

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com
