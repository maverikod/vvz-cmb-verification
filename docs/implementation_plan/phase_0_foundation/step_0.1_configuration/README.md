# Step 0.1: Configuration Management

**Phase:** 0 - Foundation and Infrastructure  
**Step:** 0.1 - Configuration Management  
**Module:** `config/`

---

## Overview

This step sets up the configuration management system for the CMB verification project. It defines all constants, data paths, and configuration parameters needed throughout the project.

---

## Input Data

- **None** - This is a foundational step with no external data dependencies
- Configuration values are defined based on:
  - Theoretical framework (see [docs/ALL.md](../../../ALL.md))
  - Technical specification (see [docs/tech_spec.md](../../../tech_spec.md))
  - Project requirements

---

## Algorithm

### 1. Define Physical Constants

- **D** - Distance parameter for frequency-to-multipole conversion
- **z_CMB** ≈ 1100 - Redshift of CMB surface
- **Conversion factors** for:
  - Frequency to multipole: `l ≈ π D ω` (see [tech_spec.md](../../../tech_spec.md) section 13.8)
  - Node depth to temperature: `ΔT ≈ 20-30 μK` via `Δω/ω` (see [tech_spec.md](../../../tech_spec.md) section 13.6)
  - Arcmin to physical scale: `2-5′ → 100-300 pc at z≈1100` (see [tech_spec.md](../../../tech_spec.md) section 13.3)

### 2. Configure Data Paths

- Input data directory: `data/in/`
- Θ-field data directory: `data/theta/`
- Output data directory: `data/out/`
- Temporary data directory: `data/tmp/`
- Data index file: `data/in/data_index.yaml`

### 3. Set Up Logging

- Configure logging levels
- Define log file locations
- Set up log rotation

### 4. Define CMB-Specific Settings

- Frequency ranges: 90-350 GHz (for frequency invariance tests)
- Multipole ranges: l up to 10000 (for high-l analysis)
- Expected peak positions: l≈4500-6000
- Node scale: ~300 pc at z≈1100

---

## Output Data

### Files Created

1. **`config/settings.py`** - Main configuration module
   - Physical constants
   - Data paths
   - Logging configuration
   - Project-wide settings

2. **`config/cmb_config.yaml`** - CMB-specific configuration
   - Frequency ranges
   - Multipole ranges
   - Peak positions
   - Node parameters

### Configuration Interface

- `Config` class providing access to all settings
- Type-safe configuration access
- Validation of configuration values

---

## Dependencies

- **None** - This is a foundational step

---

## Related Steps

- **Step 0.2** (Utilities) - Will use configuration for data paths
- **Step 0.3** (Data Index) - Will use configuration for data index path
- **All subsequent steps** - Will depend on configuration for constants and paths

---

## Tests

### Unit Tests

1. **Constants Validation**
   - **What to test:**
     - Verify all physical constants are defined (D, z_CMB, conversion factors)
     - Check that conversion factors match theoretical values:
       - Frequency to multipole: l ≈ π D ω (verify D value is correct)
       - Node depth to temperature: ΔT ≈ 20-30 μK (verify conversion formula)
       - Arcmin to physical scale: 2-5′ → 100-300 pc at z≈1100 (verify scale conversion)
     - Validate z_CMB ≈ 1100 (check value is within expected range)
     - Test that constants are immutable after initialization
     - Verify no missing or undefined constants

2. **Path Configuration**
   - **What to test:**
     - Verify all data paths are correctly set (data/in/, data/theta/, data/out/, data/tmp/)
     - Check that paths exist or can be created (test directory creation)
     - Validate path format (absolute vs relative paths)
     - Test path resolution (relative to project root)
     - Verify data_index.yaml path is correct
     - Test path validation (check for invalid characters, permissions)

3. **Configuration Loading**
   - **What to test:**
     - Test loading from YAML file (valid YAML structure)
     - Verify type conversion (string to Path, float, int)
     - Check error handling for invalid config:
       - Missing required fields
       - Invalid YAML syntax
       - Type mismatches
       - Out-of-range values
     - Test default configuration when no file provided
     - Verify configuration validation (validate() method)

4. **Configuration Access**
   - **What to test:**
     - Test Config class interface (all methods accessible)
     - Verify type safety (type hints work correctly)
     - Check default values (when config not loaded)
     - Test get_config() global access
     - Verify initialize_config() sets global config
     - Test that configuration is read-only after initialization

### Integration Tests

1. **Configuration Usage**
   - **What to test:**
     - Test that other modules can import and use config
     - Verify configuration is accessible throughout project
     - Test that changes to config file are reflected (if reload supported)
     - Verify configuration consistency across modules
     - Test configuration with actual data paths and constants

---

## Implementation Notes

- Use Python's `dataclasses` or `pydantic` for type-safe configuration
- YAML files should be validated against schema
- Configuration should be immutable after initialization
- All paths should be absolute or relative to project root

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

