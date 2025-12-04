# Step 0.3: Data Index Integration

**Phase:** 0 - Foundation and Infrastructure  
**Step:** 0.3 - Data Index Integration  
**Module:** `utils/io/`

---

## Overview

This step implements data index loading and querying functionality. The data index (`data/in/data_index.yaml`) provides a comprehensive catalog of all input data files, their contents, and usage information.

---

## Input Data

- **Data Index File:** `data/in/data_index.yaml`
  - Comprehensive index of all input data files
  - File descriptions, sizes, and usage information
  - Data categories and relationships

- **Configuration** (from [Step 0.1](../step_0.1_configuration/README.md)):
  - Data index path from `config/settings.py`

---

## Algorithm

### 1. Parse Data Index

- Load YAML file using `yaml` library
- Validate YAML structure
- Parse nested data structure
- Handle missing or invalid entries

### 2. Create Query Interface

- Implement search by category (CMB data, LSS data, etc.)
- Implement search by file name
- Implement search by data type
- Provide filtered views of data catalog

### 3. Validate File Existence

- Check if indexed files actually exist
- Report missing files
- Handle path resolution (relative/absolute)

### 4. Provide Data Access Helpers

- Get file paths for specific data types
- Get metadata for files
- Get usage guidelines for data sets

---

## Output Data

### Files Created

1. **`utils/io/data_index_loader.py`** - Data index loading and querying
   - `DataIndex` class for index management
   - Query methods for finding data files
   - Validation methods

### Interface

- `DataIndex` class with methods:
  - `load()` - Load index from YAML
  - `get_file_path(category, name)` - Get file path
  - `get_files_by_category(category)` - Get all files in category
  - `validate_files()` - Check file existence
  - `get_usage_guidelines(category)` - Get usage information

---

## Dependencies

- **Step 0.1** (Configuration) - For data index path
- External library: `yaml` (PyYAML)

---

## Related Steps

- **Step 1.1** (Θ-Field Data Loader) - Will use index to find theta data
- **Step 2.1** (CMB Reconstruction) - Will use index to find ACT data
- **All data loading steps** - Will use index to locate data files

---

## Tests

### Unit Tests

1. **Index Loading**
   - **What to test:**
     - Test YAML parsing:
       - Verify YAML file is parsed correctly
       - Test nested structure handling
       - Verify all sections are loaded
     - Test structure validation:
       - Verify required fields are present
       - Test data type validation
       - Test structure consistency
     - Test error handling for invalid YAML:
       - Invalid YAML syntax (should raise YAMLError)
       - Missing required fields (should raise ValueError)
       - Type mismatches (should raise ValueError)
       - Appropriate error messages

2. **Query Interface**
   - **What to test:**
     - Test category-based search:
       - Verify get_files_by_category() returns correct files
       - Test with valid categories (e.g., 'act_dr6_02', 'cmb_microcells')
       - Test with invalid categories (should return empty list or raise error)
     - Test file name search:
       - Verify search_by_name() finds matching files
       - Test wildcard pattern matching
       - Test case sensitivity
     - Test data type filtering:
       - Verify filtering by data type works correctly
       - Test multiple filter combinations

3. **File Validation**
   - **What to test:**
     - Test file existence checking:
       - Verify validate_files() checks all indexed files
       - Test that existing files are reported as found
       - Test that missing files are reported as missing
     - Test path resolution:
       - Verify relative paths are resolved correctly
       - Test absolute paths are handled correctly
       - Test path resolution relative to index file location
     - Test missing file reporting:
       - Verify missing files are listed in 'missing' key
       - Verify found files are listed in 'found' key
       - Test reporting format

4. **Data Access**
   - **What to test:**
     - Test getting file paths:
       - Verify get_file_path() returns correct path for valid category/name
       - Test returns None for non-existent files
       - Verify paths are absolute or correctly relative
     - Test metadata retrieval:
       - Verify file metadata is accessible
       - Test metadata structure matches index format
     - Test usage guidelines access:
       - Verify get_usage_guidelines() returns guidelines for valid category
       - Test returns None for categories without guidelines
       - Verify guidelines structure

### Integration Tests

1. **End-to-End Data Discovery**
   - **What to test:**
     - Load index → Query for data → Validate files → Return paths:
       - Load data_index.yaml
       - Query for specific data file (e.g., ACT DR6.02 map)
       - Validate that file exists
       - Return file path
       - Verify path can be used to load actual data
     - Test with actual data files:
       - Use real data_index.yaml
       - Query for real data files
       - Verify all queries return valid paths
       - Test that missing files are correctly identified

---

## Implementation Notes

- Index should be cached after first load
- Provide clear error messages for missing data
- Support both relative and absolute paths
- Handle updates to index file gracefully

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

