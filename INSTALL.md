# Installation & Usage Guide

## Requirements

- Python 3.8+ (3.11 recommended)
- pip

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/BardiaYaghmaie/Graph-Edge-Magic-Total-Labeling.git
cd Graph-Edge-Magic-Total-Labeling

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running the Solver

### Command Line

```bash
python emtl_solver.py
```

### Python API

```python
from emtl_solver import solve_emtl

result = solve_emtl(m=2, n=3, k=2, t=2)

if result.exists:
    print(f"Magic constant: {result.magic_constant}")
```

## Web Interface

```bash
streamlit run web/app.py
```
Open http://localhost:8501 in your browser.

## Jupyter Notebook

```bash
jupyter notebook notebooks/EMTL_Tutorial.ipynb
```

## Running Tests

```bash
pytest tests/ -v
```

## Examples

```bash
# Run comprehensive examples
python examples/run_examples.py

# Interactive mode
python examples/run_examples.py -i

# Stress test (larger graphs)
python examples/run_examples.py --stress
```

## Parameters

| Parameter | Description | Constraints |
|-----------|-------------|-------------|
| m | Vertices in set A | m ≥ 1 |
| n | Vertices in sets B and C | n ≥ 1 |
| k | Vertices in set D | k ≥ 1 |
| t | B-C subgraph regularity | 0 ≤ t ≤ n |

