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
   - **What to test:**
     - Test HEALPix FITS file loading:
       - Verify correct reading of HEALPix maps from FITS
       - Test NSIDE parameter extraction
       - Test field selection (field parameter)
       - Test HDU selection (hdu parameter)
       - Verify map data type and shape
     - Test power spectrum archive extraction:
       - Verify tar.gz archive extraction
       - Test finding spectrum file within archive
       - Verify spectrum data parsing (l, C_l, errors)
       - Test archive format handling
     - Test CSV/JSON loading:
       - Verify CSV column parsing
       - Verify JSON structure parsing
       - Test data type conversion
       - Test missing value handling
     - Test error handling for invalid files:
       - FileNotFoundError for missing files
       - ValueError for invalid formats
       - Appropriate error messages

2. **Data Saver Tests**
   - **What to test:**
     - Test FITS file writing:
       - Verify HEALPix map is written correctly
       - Test NSIDE parameter is saved
       - Verify map can be read back correctly
       - Test overwrite flag behavior
     - Test output directory creation:
       - Verify directories are created if missing
       - Test permission handling
       - Test path resolution
     - Test format validation:
       - Verify data format matches output format
       - Test format conversion (json, csv, npy)
       - Test required fields presence

3. **Frequency Conversion Tests**
   - **What to test:**
     - Test l ≈ π D ω conversion accuracy:
       - Verify formula: l = π D ω (test with known values)
       - Test D parameter from config is used correctly
       - Verify conversion precision
       - Test array conversion (vectorized operation)
     - Test inverse conversion (l → ω):
       - Verify ω = l/(π D) formula
       - Test round-trip conversion (ω → l → ω)
       - Verify inverse is exact (within numerical precision)
     - Test edge cases:
       - Very low frequencies (near zero)
       - Very high frequencies
       - Negative values (should raise error)
       - Zero frequency (should raise error)

4. **Spherical Harmonics Tests**
   - **What to test:**
     - Test decomposition accuracy:
       - Verify decomposition of known map produces correct a_lm
       - Test decomposition preserves map information
       - Verify l_max parameter handling
     - Test synthesis round-trip:
       - Decompose map → Synthesize map → Compare with original
       - Verify round-trip preserves map within numerical precision
       - Test with different NSIDE values
     - Test HEALPix conversion:
       - Verify HEALPix to a_lm conversion
       - Verify a_lm to HEALPix conversion
       - Test pixel indexing correctness

5. **Visualization Tests**
   - **What to test:**
     - Test plot generation (no errors):
       - Verify plots are created without exceptions
       - Test with various map sizes
       - Test with different projections (mollweide, orthographic)
     - Test figure saving:
       - Verify figures are saved to specified paths
       - Test different formats (PNG, PDF, SVG)
       - Verify saved files can be opened
     - Test projection correctness:
       - Verify Mollweide projection shows correct sky coverage
       - Test coordinate system (theta, phi) mapping
       - Verify colorbar units and scales

### Integration Tests

1. **End-to-End Data Pipeline**
   - **What to test:**
     - Load data → Process → Save → Load again:
       - Load HEALPix map from FITS
       - Process map (e.g., apply conversion)
       - Save processed map to FITS
       - Load saved map and verify it matches processed map
     - Verify data integrity throughout pipeline:
       - Test that no data is lost or corrupted
       - Verify metadata is preserved
       - Test with actual ACT DR6.02 data files

---

## Implementation Notes

- All functions should have comprehensive type hints
- Use context managers for file operations
- Implement proper error handling with informative messages
- Follow project coding standards (max 400 lines per file)

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

