# Project Structure

This document describes the organization of the EMTL Solver project.

```
EMTL/
├── emtl_solver.py          # Main solver implementation
├── README.md               # Comprehensive documentation
├── requirements.txt        # Python dependencies
├── PROJECT_STRUCTURE.md   # This file
├── .gitignore            # Git ignore rules
│
├── images/                # All image files
│   ├── examples/         # Example visualizations (tracked in git)
│   │   └── *.png        # Pre-generated example graphs
│   └── output/          # Generated output (gitignored)
│       └── *.png        # User-generated visualizations
│
├── tests/                # Test suite
│   ├── __init__.py
│   └── test_emtl.py     # Comprehensive tests (55 tests)
│
├── notebooks/            # Jupyter notebooks
│   └── EMTL_Tutorial.ipynb  # Interactive tutorial
│
├── web/                  # Web interface
│   └── app.py           # Streamlit application
│
├── examples/             # Example scripts
│   ├── run_examples.py  # Comprehensive example runner
│   └── output/          # (Legacy, use images/output/)
│
└── venv/                 # Virtual environment (gitignored)
    └── ...
```

## Directory Descriptions

### Root Files
- **emtl_solver.py**: Main implementation with all solver components
- **README.md**: Complete documentation with theory, usage, and examples
- **requirements.txt**: All Python package dependencies
- **PROJECT_STRUCTURE.md**: This file

### images/
Contains all visualization files:
- **images/examples/**: Pre-generated example visualizations (tracked in git)
- **images/output/**: User-generated output files (gitignored)

### tests/
Comprehensive test suite using pytest:
- Unit tests for all components
- Integration tests
- Parametric tests for various configurations
- Edge case and boundary condition tests

### notebooks/
Interactive Jupyter notebooks for learning:
- **EMTL_Tutorial.ipynb**: Complete tutorial with exercises

### web/
Streamlit web application for interactive exploration:
- **app.py**: Full-featured web interface

### examples/
Example scripts and runners:
- **run_examples.py**: Comprehensive example runner with multiple modes

## File Naming Conventions

- Images: `emtl_m{m}_n{n}_k{k}_t{t}.png` or `emtl_example{N}.png`
- Python files: `snake_case.py`
- Test files: `test_*.py`
- Documentation: `*.md`

## Git Tracking

**Tracked:**
- All source code files
- Documentation (README.md, PROJECT_STRUCTURE.md)
- Example images in `images/examples/`
- Configuration files

**Ignored:**
- Virtual environment (`venv/`)
- Python cache (`__pycache__/`)
- Generated output (`images/output/*.png`)
- IDE settings
- Test coverage files

