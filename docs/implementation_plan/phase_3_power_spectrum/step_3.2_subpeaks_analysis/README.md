# Step 3.2: High-l Sub-peaks Analysis

**Phase:** 3 - Power Spectrum Generation  
**Step:** 3.2 - High-l Sub-peaks Analysis  
**Module:** `cmb/spectrum/`

---

## Overview

This step analyzes high-l sub-peaks in the power spectrum. It identifies sub-peaks from ω_min beatings, detects the predicted l≈4500-6000 peak, and analyzes sub-structure.

---

## Input Data

- **Power Spectrum** (from [Step 3.1](../step_3.1_spectrum_calculation/README.md)):
  - C_l power spectrum from `PowerSpectrumCalculator`

- **Θ-Field Evolution** (from [Step 1.3](../../phase_1_theta_data/step_1.3_evolution_data/README.md)):
  - ω_min(t) evolution data for beatings analysis

- **Configuration** (from [Step 0.1](../../phase_0_foundation/step_0.1_configuration/README.md)):
  - Expected peak range (l≈4500-6000)

---

## Algorithm

### 1. Calculate Beatings from ω_min(t)

- Calculate beating frequencies from temporal evolution ω_min(t)
- **Formula:** Beatings arise from modulations in ω_min(t) evolution
- Convert beating frequencies to multipoles using: **l = π D ω**
- Beatings create sub-peak structure in power spectrum
- Each beating corresponds to a sub-peak in C_l

### 2. Identify Sub-peaks from Beatings

- Analyze power spectrum for sub-peak structures
- Match calculated beatings to observed sub-peak positions
- Validate beating-subpeak correspondence
- Document beating-subpeak relationships

### 3. Detect High-l Peak

- Search for peak in l≈4500-6000 range
- This corresponds to characteristic node scale R≈300 pc
- Using formula: l ≈ π/θ, where θ = R/D ≈ 2.2'
- Identify peak position and amplitude
- Calculate peak significance
- Compare with predictions

### 4. Analyze Sub-structure

- Identify all sub-peaks in high-l range
- Calculate sub-peak properties (position, amplitude, width)
- Analyze sub-peak spacing (should match beating frequencies)
- Characterize sub-structure pattern
- Verify that sub-peaks correspond to ω_min beatings

### 4. Generate Sub-peak Catalog

- Create catalog of identified sub-peaks
- Document peak properties
- Provide analysis results
- Create visualization

---

## Output Data

### Files Created

1. **`cmb/spectrum/subpeaks_analyzer.py`** - Sub-peaks analysis module
   - `SubpeaksAnalyzer` class
   - Analysis functions

### Analysis Results

- **Sub-peak Catalog:**
  - Peak positions (l values)
  - Peak amplitudes
  - Peak widths
  - Beating correspondences

- **High-l Peak Detection:**
  - Peak position (l value)
  - Peak amplitude
  - Significance level
  - Comparison with predictions

---

## Dependencies

- **Step 1.3** (Evolution Data) - For ω_min(t) beatings
- **Step 3.1** (Power Spectrum) - For C_l data

---

## Related Steps

- **Step 3.3** (Spectrum Comparison) - Will use sub-peak analysis
- **Step 5.1** (High-l Peak Prediction) - Will validate predictions

---

## Tests

### Unit Tests

1. **Beating Analysis**
   - **What to test:**
     - Test ω_min(t) beating calculation:
       - Verify _calculate_beatings() analyzes ω_min(t) evolution for modulations
       - Test that beatings are extracted from modulations correctly
       - Verify beating frequencies are in Hz
       - Test conversion to multipoles using l = π D ω
     - Test beating-to-subpeak matching:
       - Verify _match_beatings_to_peaks() matches beatings to sub-peak positions
       - Test matching tolerance (frequency/multipole range)
       - Verify all beatings are matched to peaks (or marked as unmatched)
     - Test correspondence validation:
       - Verify beating frequencies correspond to sub-peak l positions
       - Test correspondence accuracy

2. **Peak Detection**
   - **What to test:**
     - Test peak finding algorithms:
       - Verify _find_subpeaks() identifies peaks in power spectrum correctly
       - Test peak detection with various noise levels
       - Verify peak properties (position, amplitude, width) are calculated correctly
     - Test l≈4500-6000 range search:
       - Verify _detect_high_l_peak() searches in correct range
       - Test that high-l peak is found if present
       - Verify peak position is within 4500-6000 range
     - Test peak significance calculation:
       - Verify significance is calculated correctly (statistical test)
       - Test significance threshold handling

3. **Sub-structure Analysis**
   - **What to test:**
     - Test sub-peak identification:
       - Verify all sub-peaks are identified correctly
       - Test sub-peak properties (l_position, amplitude, width, significance)
       - Verify sub-peak catalog is complete
     - Test property calculation:
       - Verify peak amplitude calculation
       - Verify peak width calculation
       - Verify peak significance calculation
     - Test pattern analysis:
       - Verify sub-peak pattern matches beating pattern
       - Test pattern consistency

4. **Catalog Generation**
   - **What to test:**
     - Test catalog creation:
       - Verify SubPeakAnalysis structure is created correctly
       - Test all fields are populated (peaks, high_l_peak, beating_frequencies, analysis_metadata)
     - Test data integrity:
       - Verify all arrays have consistent lengths
       - Verify peak positions match spectrum data
     - Test visualization:
       - Verify sub-peaks can be visualized on spectrum plot
       - Test visualization accuracy

### Integration Tests

1. **End-to-End Sub-peak Analysis**
   - **What to test:**
     - Load spectrum → Analyze beatings → Detect peaks → Generate catalog:
       - Load PowerSpectrum data
       - Load ThetaEvolutionProcessor
       - Calculate beatings from ω_min(t) evolution
       - Find sub-peaks in power spectrum
       - Detect high-l peak in l≈4500-6000 range
       - Match beatings to peaks
       - Generate SubPeakAnalysis catalog
       - Verify all steps complete successfully
     - Test with actual power spectrum data:
       - Real calculated C_l spectrum
       - Real evolution data
       - Verify sub-peaks are correctly identified
       - Verify high-l peak is found (if present)

---

## Implementation Notes

- Calculate beatings from ω_min(t) evolution first
- Convert beatings to multipoles using l = π D ω
- Use robust peak-finding algorithms
- Handle noise in power spectrum
- Provide clear peak identification criteria
- Support multiple peak detection methods

## Forbidden Elements

**DO NOT USE:**
- ❌ Classical peak detection without beating analysis
- ❌ Potentials V(φ), V(Θ)
- ❌ Mass terms m²φ², m²Θ²
- ❌ Exponential damping exp(-r/λ)

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

