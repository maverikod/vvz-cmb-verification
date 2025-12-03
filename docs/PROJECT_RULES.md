# Project Rules and Guidelines

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Coding Standards](#coding-standards)
3. [File Naming Conventions](#file-naming-conventions)
4. [Code Organization](#code-organization)
5. [Documentation Standards](#documentation-standards)
6. [Git Workflow](#git-workflow)
7. [Testing and Quality Assurance](#testing-and-quality-assurance)

---

## Project Structure

### Directory Tree

```
cmb-verification/
├── cmb/                   # CMB verification module
│   ├── __init__.py
│   ├── reconstruction/   # CMB map reconstruction
│   ├── spectrum/         # C_l spectrum generation
│   ├── nodes/            # Θ-node analysis
│   ├── correlation/      # CMB-LSS correlation
│   └── predictions/      # ACT/SPT predictions
├── theta/                # Θ-field model
│   ├── __init__.py
│   ├── field/            # Θ-field calculations
│   ├── nodes/            # Node structure
│   └── evolution/        # Temporal evolution
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── io/               # Input/output operations
│   ├── math/             # Mathematical utilities
│   └── visualization/    # Plotting functions
├── config/               # Configuration management
│   ├── __init__.py
│   └── settings.py
├── tests/                # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test data and fixtures
├── scripts/              # Standalone scripts
│   ├── data_processing/  # Data processing scripts
│   └── analysis/         # Analysis scripts
├── data/                 # Data directory
│   ├── in/               # Input data (tracked in git)
│   ├── out/              # Output data (gitignored)
│   └── tmp/              # Temporary data (gitignored)
├── docs/                 # Documentation
│   ├── ALL.md           # Complete theoretical framework and documentation
│   ├── ALL_index.yaml   # Index for search and analysis of theory content
│   ├── PROJECT_RULES.md # Project rules and guidelines
│   ├── tech_spec.md     # Technical specification
│   ├── api/             # API documentation
│   ├── implementation_plan/  # Implementation plan
│   │   ├── implementation_plan.md  # Main plan document
│   │   └── phase_*/              # Phase directories
│   │       └── step_*/          # Step directories
│   └── reports/         # Generated reports
├── code_analysis/        # Code analysis results
│   ├── code_map.yaml    # File and method index
│   ├── method_index.yaml # Method descriptions
│   └── code_issues.yaml # Code issues
├── .venv/               # Virtual environment (gitignored)
├── .gitignore
├── README.md            # Project overview
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project configuration
└── setup.py             # Package setup (if needed)
```

### Directory Rules

- **cmb/**, **theta/**, **utils/**, **config/**: Main production code modules in root directory
- **tests/**: Mirror structure of main modules for easy navigation
- **scripts/**: Standalone executable scripts (not imported modules)
- **data/**: ALL input/output data must be in this directory
  - **data/in/**: Input data files (small files can be tracked)
    - **data/in/data_index.yaml**: Comprehensive index of all input data (MUST be maintained)
  - **data/out/**: Generated output files (never tracked, gitignored)
  - **data/tmp/**: Temporary data files (never tracked, gitignored)
- **docs/**: All documentation files (theory, rules, specs, API docs)
- **docs/reports/**: Generated reports and analysis outputs
- **code_analysis/**: Auto-generated code analysis (updated by code_mapper)

---

## Coding Standards

### Language and Style

- **Language**: Python 3.8+
- **Code Style**: PEP 8 compliant
- **Formatter**: Black (line length: 88)
- **Linter**: Flake8
- **Type Checking**: MyPy (strict mode)

### Code Comments and Documentation

- **Language**: English (code, comments, docstrings, tests)
- **Communication**: Russian (with user)
- **Documentation**: English (unless user requests Russian)

### File Headers

Every code file MUST contain a header docstring:

```python
"""
Module description.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""
```

### Docstring Format

Use Google-style docstrings:

```python
def calculate_cmb_spectrum(frequencies: np.ndarray, theta_nodes: List[Node]) -> np.ndarray:
    """
    Calculate CMB power spectrum from Θ-field nodes.
    
    Args:
        frequencies: Array of frequency values in Hz
        theta_nodes: List of Θ-field node structures
        
    Returns:
        Power spectrum array C_l
        
    Raises:
        ValueError: If frequencies array is empty
    """
    pass
```

### Code Quality Rules

1. **File Size**: Maximum 350-400 lines per file
2. **One Class Per File**: Except for exceptions, enums, and error classes
3. **Large Classes**: Split into facade class + smaller helper classes
4. **No Pass in Production**: All methods must be implemented (except abstract methods with `NotImplemented`)
5. **No Hardcoding**: Use configuration files or constants
6. **Imports**: All imports at the top of file (except lazy loading)

### Abstract Methods

- Abstract methods MUST use `NotImplemented` instead of `pass`
- Example:
```python
def abstract_method(self) -> None:
    """Abstract method description."""
    raise NotImplementedError("This method must be implemented in subclass")
```

---

## File Naming Conventions

### Python Files

- **Modules**: `snake_case.py`
  - Example: `cmb_reconstruction.py`, `theta_field.py`
- **Classes**: One class per file, file name matches class name (lowercase)
  - Example: `class CmbReconstructor` → `cmb_reconstructor.py`
- **Packages**: `snake_case/` directory with `__init__.py`
- **Tests**: `test_<module_name>.py`
  - Example: `test_cmb_reconstruction.py`
- **Scripts**: `snake_case.py` in `scripts/` directory

### Configuration Files

- **YAML**: `snake_case.yaml`
  - Example: `config.yaml`, `code_map.yaml`
- **JSON**: `snake_case.json`
- **TOML**: `snake_case.toml` or `pyproject.toml`

### Data Files

- **Input Data**: Descriptive names with version
  - Example: `act_dr6.02_spectra_binning_20.tar.gz`
- **Output Data**: `<analysis_type>_<timestamp>.<ext>`
  - Example: `cmb_spectrum_20240115.npy`

### Documentation Files

- **All documentation** must be in `docs/` directory
- **Reports** must be in `docs/reports/` directory
- **Markdown**: `UPPER_CASE.md` for main docs, `snake_case.md` for sections
  - Example: `docs/PROJECT_RULES.md`, `docs/tech_spec.md`, `docs/api/api_reference.md`
  - Reports: `docs/reports/analysis_20240115.md`

---

## Code Organization

### Module Structure

Each module should follow this structure:

```python
"""
Module description.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

# Standard library imports
import os
from typing import List, Optional

# Third-party imports
import numpy as np
import astropy.units as u

# Local imports
from utils.math import calculate_distance
from config.settings import Config

# Constants
DEFAULT_FREQUENCY = 150.0  # GHz

# Classes
class MyClass:
    """Class description."""
    pass

# Functions
def my_function() -> None:
    """Function description."""
    pass
```

### Import Order

1. Standard library
2. Third-party packages
3. Local application imports
4. Constants
5. Classes
6. Functions

### Class Design

- **Single Responsibility**: Each class has one clear purpose
- **Facade Pattern**: For complex modules, use a facade class
- **Dependency Injection**: Prefer dependency injection over global state
- **Type Hints**: All functions and methods must have type hints

---

## Documentation Standards

### Module Documentation

Every module must have:
- Header docstring with author and email
- Module-level docstring explaining purpose
- Public API documentation

### Function/Method Documentation

- Purpose description
- Args section (all parameters)
- Returns section (return type and description)
- Raises section (all exceptions)
- Examples (for complex functions)

### Inline Comments

- Use comments to explain **why**, not **what**
- Complex algorithms require step-by-step comments
- Avoid obvious comments

---

## Git Workflow

### Branch Strategy

- **main**: Production-ready code
- **develop**: Development branch
- **feature/<name>**: Feature branches
- **fix/<name>**: Bug fix branches

### Commit Messages

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

Examples:
```
feat: add CMB spectrum calculation
fix: correct theta node depth calculation
docs: update API documentation
refactor: split large CmbReconstructor class
```

### Commit Frequency

- Commit after each logically complete action
- Commit after fixing linter errors
- Commit after code_mapper updates
- **Never commit** broken code or failing tests

### Pre-Commit Checklist

After completing a logical unit of work:

1. ✅ Check virtual environment is activated: `source .venv/bin/activate`
2. ✅ Run code formatter: `black .`
3. ✅ Run linter: `flake8 .`
4. ✅ Run type checker: `mypy .`
5. ✅ Run code_mapper: `code_mapper` (updates code_analysis/)
6. ✅ Fix issues from `code_issues.yaml`
7. ✅ Run tests: `pytest`
8. ✅ Commit changes

---

## Testing and Quality Assurance

### Test Structure

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test module interactions
- **Fixtures**: Reusable test data in `tests/fixtures/`

### Test Naming

- Test files: `test_<module_name>.py`
- Test functions: `test_<functionality>`
- Example: `test_calculate_cmb_spectrum()`

### Code Analysis Tools

1. **code_mapper**: Generates code analysis indexes
   - Run after each code change
   - Updates `code_analysis/` directory
   - Check `code_issues.yaml` for problems

2. **Search Tools**:
   - Code search: Use `code_map.yaml` and `method_index.yaml`
   - Theory search and analysis: Use `docs/ALL_index.yaml` (index for `docs/ALL.md` theory document)
   - Data search: Use `data/in/data_index.yaml` - comprehensive index of all input data files
     - **MUST** keep `data/in/data_index.yaml` up to date when adding new data files
     - **MUST** use this index to find available data before processing
     - **MUST** update index when data structure changes

### Quality Gates

Before merging to main:
- ✅ All tests pass
- ✅ No linter errors
- ✅ No type errors
- ✅ No issues in `code_issues.yaml`
- ✅ Code coverage > 80% (where applicable)
- ✅ All files < 400 lines

---

## Additional Rules

### Virtual Environment

- **Always** activate `.venv` before working: `source .venv/bin/activate`
- **Never** use `--break-system-packages` flag
- Install packages only in virtual environment

### Dependencies

- Pin major versions in `requirements.txt`
- Use `pyproject.toml` for project metadata
- Document all external dependencies

### Data Handling

- **ALL input/output data** must be in `data/` directory
- **Input data**: `data/in/` - all input files go here
- **Output data**: `data/out/` - all generated output files go here (gitignored)
- **Temporary data**: `data/tmp/` - temporary files during processing (gitignored)
- **NEVER** create data files outside `data/` directory
- Large data files (>10MB) should not be tracked in git
- Use `.gitignore` for `data/out/` and `data/tmp/`
- Document data sources and processing steps

### Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors appropriately
- Never use bare `except:` clauses

---

## Country and Symbol Standards

- **Country**: Ukraine
- **Symbols**: Only Ukrainian symbols (no Russian symbols)
- **Language**: Russian for communication, English for code

---

## Summary Checklist for Developers

Before starting work:
- [ ] Activate virtual environment
- [ ] Run code_mapper to check existing code
- [ ] Review `code_issues.yaml` for known issues
- [ ] Check `code_map.yaml` for existing functionality

During development:
- [ ] Follow file size limits (max 400 lines)
- [ ] Write docstrings for all public functions/classes
- [ ] Add type hints to all functions
- [ ] Write tests for new functionality

After completing work:
- [ ] Run black, flake8, mypy
- [ ] Run code_mapper
- [ ] Fix issues from `code_issues.yaml`
- [ ] Run tests
- [ ] Commit with descriptive message

---

**Last Updated**: 2024-01-15

