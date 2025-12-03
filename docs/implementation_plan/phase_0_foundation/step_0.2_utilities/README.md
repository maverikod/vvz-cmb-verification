# Step 0.2: Utility Modules

**Phase:** 0 - Foundation and Infrastructure  
**Step:** 0.2 - Utility Modules  
**Module:** `utils/`

---

## Overview

This step implements core utility modules for data I/O, mathematical operations, and visualization. These utilities provide the foundation for all CMB verification operations.

---

## Input Data

- **Configuration** (from [Step 0.1](../step_0.1_configuration/README.md)):
  - Data paths from `config/settings.py`
  - Physical constants for conversions

- **External Libraries:**
  - `healpy` - For HEALPix map operations
  - `numpy` - For numerical operations
  - `astropy` - For astronomical data handling
  - `matplotlib` - For visualization

---

## Algorithm

### 1. Data I/O Utilities (`utils/io/`)

#### `data_loader.py`
- Load HEALPix FITS files (ACT DR6.02 maps)
- Load power spectrum data from tar.gz archives
- Load CSV/JSON data files
- Validate file formats and data integrity

#### `data_saver.py`
- Save HEALPix maps to FITS format
- Save power spectra to various formats
- Save analysis results to JSON/CSV
- Handle output directory creation

### 2. Mathematical Utilities (`utils/math/`)

#### `frequency_conversion.py`
- Implement `l ≈ π D ω` conversion (see [tech_spec.md](../../../tech_spec.md) section 13.8)
- Convert between frequency and multipole space
- Handle frequency evolution over time

#### `spherical_harmonics.py`
- Spherical harmonic decomposition
- HEALPix to spherical harmonic conversion
- Spherical harmonic synthesis for map generation

### 3. Visualization Utilities (`utils/visualization/`)

#### `cmb_plots.py`
- Plot HEALPix CMB maps (Mollweide projection)
- Visualize temperature fluctuations
- Create node overlay visualizations

#### `spectrum_plots.py`
- Plot power spectra C_l vs l
- Highlight peaks and sub-peaks
- Compare theoretical vs observed spectra

---

## Output Data

### Files Created

1. **`utils/io/data_loader.py`** - Data loading utilities
2. **`utils/io/data_saver.py`** - Data saving utilities
3. **`utils/math/frequency_conversion.py`** - Frequency-multipole conversions
4. **`utils/math/spherical_harmonics.py`** - Spherical harmonics operations
5. **`utils/visualization/cmb_plots.py`** - CMB map visualization
6. **`utils/visualization/spectrum_plots.py`** - Power spectrum visualization

### Utility Functions

All modules provide well-documented, reusable functions that can be imported and used throughout the project.

---

## Dependencies

- **Step 0.1** (Configuration) - For data paths and constants
- External libraries: healpy, numpy, astropy, matplotlib

---

## Related Steps

- **Step 0.3** (Data Index) - Will use data_loader for file operations
- **Step 1.1** (Θ-Field Data Loader) - Will use data_loader utilities
- **Step 2.1** (CMB Reconstruction) - Will use spherical_harmonics and data_loader
- **Step 3.1** (Power Spectrum) - Will use frequency_conversion and spectrum_plots
- **All visualization steps** - Will use visualization utilities

---

## Tests

### Unit Tests

1. **Data Loader Tests**
   - Test HEALPix FITS file loading
   - Test power spectrum archive extraction
   - Test CSV/JSON loading
   - Test error handling for invalid files

2. **Data Saver Tests**
   - Test FITS file writing
   - Test output directory creation
   - Test format validation

3. **Frequency Conversion Tests**
   - Test l ≈ π D ω conversion accuracy
   - Test inverse conversion (l → ω)
   - Test edge cases (very low/high frequencies)

4. **Spherical Harmonics Tests**
   - Test decomposition accuracy
   - Test synthesis round-trip
   - Test HEALPix conversion

5. **Visualization Tests**
   - Test plot generation (no errors)
   - Test figure saving
   - Test projection correctness

### Integration Tests

1. **End-to-End Data Pipeline**
   - Load data → Process → Save → Load again
   - Verify data integrity throughout pipeline

---

## Implementation Notes

- All functions should have comprehensive type hints
- Use context managers for file operations
- Implement proper error handling with informative messages
- Follow project coding standards (max 400 lines per file)

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

