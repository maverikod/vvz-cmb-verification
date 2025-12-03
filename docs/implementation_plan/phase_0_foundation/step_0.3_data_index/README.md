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
   - Test YAML parsing
   - Test structure validation
   - Test error handling for invalid YAML

2. **Query Interface**
   - Test category-based search
   - Test file name search
   - Test data type filtering

3. **File Validation**
   - Test file existence checking
   - Test path resolution
   - Test missing file reporting

4. **Data Access**
   - Test getting file paths
   - Test metadata retrieval
   - Test usage guidelines access

### Integration Tests

1. **End-to-End Data Discovery**
   - Load index → Query for data → Validate files → Return paths
   - Test with actual data files

---

## Implementation Notes

- Index should be cached after first load
- Provide clear error messages for missing data
- Support both relative and absolute paths
- Handle updates to index file gracefully

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

