# CMB Verification Project

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

---

## Project Description

This project implements verification and analysis of CMB (Cosmic Microwave Background) microstructures as projections of Θ-field nodes, based on the theoretical framework described in the technical specification.

The project focuses on:
- CMB map reconstruction from Θ-field frequency spectrum
- C_l power spectrum generation
- Θ-node analysis and correlation with LSS (Large Scale Structure)
- Predictions for ACT/SPT/Simons Observatory
- Verification of the chain: CMB → LSS → clusters → galaxies

---

## Project Structure

```
cmb-verification/
├── cmb/                   # CMB verification module
│   ├── reconstruction/   # CMB map reconstruction
│   ├── spectrum/         # C_l spectrum generation
│   ├── nodes/            # Θ-node analysis
│   ├── correlation/      # CMB-LSS correlation
│   └── predictions/      # ACT/SPT predictions
├── theta/                # Θ-field model
│   ├── field/            # Θ-field calculations
│   ├── nodes/            # Node structure
│   └── evolution/        # Temporal evolution
├── utils/                # Utility functions
│   ├── io/               # Input/output operations
│   ├── math/             # Mathematical utilities
│   └── visualization/    # Plotting functions
├── config/               # Configuration management
├── tests/                # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test data and fixtures
├── scripts/              # Standalone scripts
│   ├── data_processing/  # Data processing scripts
│   └── analysis/         # Analysis scripts
├── data/                 # Data directory
│   ├── in/               # Input data
│   ├── out/              # Output data (gitignored)
│   └── tmp/              # Temporary data (gitignored)
├── docs/                 # Documentation
│   ├── ALL.md           # Complete theoretical framework and documentation
│   ├── ALL_index.yaml   # Index for search and analysis of theory content
│   ├── PROJECT_RULES.md # Project rules and guidelines
│   ├── tech_spec.md     # Technical specification
│   ├── implementation_plan/  # Implementation plan with phases and steps
│   ├── api/             # API documentation
│   └── reports/          # Generated reports
└── code_analysis/        # Code analysis results
    ├── code_map.yaml    # File and method index
    ├── method_index.yaml # Method descriptions
    └── code_issues.yaml # Code issues
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (`.venv/`)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cmb-verification
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Development

### Code Standards

See [docs/PROJECT_RULES.md](docs/PROJECT_RULES.md) for detailed coding standards, naming conventions, and project guidelines.

Key points:
- **Code Style**: PEP 8, Black formatter, Flake8 linter, MyPy type checking
- **File Size**: Maximum 350-400 lines per file
- **Documentation**: All code, comments, and docstrings in English
- **One Class Per File**: Except for exceptions, enums, and error classes

### Workflow

After completing a logical unit of work:

1. Activate virtual environment: `source .venv/bin/activate`
2. Run formatter: `black .`
3. Run linter: `flake8 .`
4. Run type checker: `mypy .`
5. Run code_mapper: `code_mapper` (updates `code_analysis/`)
6. Fix issues from `code_issues.yaml`
7. Run tests: `pytest`
8. Commit changes

### Code Search

- **Code search**: Use `code_analysis/code_map.yaml` and `code_analysis/method_index.yaml`
- **Theory search and analysis**: Use `docs/ALL_index.yaml` (index for `docs/ALL.md` theory document)

---

## Technical Specification

See [docs/tech_spec.md](docs/tech_spec.md) for the complete technical specification and requirements.

---

## Data

Input data is stored in `data/in/`:
- ACT DR6.02 data
- CMB microcells data
- LSS correlation data
- Θ-field background data

Output data is generated in `data/out/` (not tracked in git).

---

## Documentation

- **Theory**: `docs/ALL.md` - Complete theoretical framework
- **Technical Specification**: `docs/tech_spec.md` - Technical specification and requirements
- **Project Rules**: `docs/PROJECT_RULES.md` - Project rules and guidelines
- **API**: `docs/api/` - API documentation (to be generated)
- **Reports**: `docs/reports/` - Generated analysis reports

---

## License

[To be specified]

---

## Contact

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com
