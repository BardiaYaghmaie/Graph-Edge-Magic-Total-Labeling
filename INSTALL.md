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

# 3. Upgrade pip (recommended)
python -m pip install --upgrade pip

# 4. Install the project (recommended)
# - installs core dependencies (OR-Tools, NetworkX, Matplotlib, Streamlit, ...)
pip install -e .

# 5. Install dev/testing/notebook extras (optional, but recommended if you will run tests)
pip install -e ".[dev]"
```

> ⚠️ **Important**: Always activate the virtual environment before running any commands:
> ```bash
> source venv/bin/activate  # macOS/Linux
> venv\Scripts\activate     # Windows
> ```

## Running the Solver

### Command Line

```bash
# Make sure venv is activated!
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
# Make sure venv is activated!
streamlit run web/app.py
```
Open http://localhost:8501 in your browser.

## Jupyter Notebook

```bash
# Make sure venv is activated!
jupyter notebook notebooks/EMTL_Tutorial.ipynb
```

## Running Tests

```bash
# Make sure venv is activated!
pytest tests/ -v
```

## Troubleshooting: pip timeouts / slow networks

If you see errors like `ReadTimeoutError` while installing (common on slow connections),
re-run installs with a larger timeout and more retries:

```bash
pip install --default-timeout=120 --retries=10 -e .
pip install --default-timeout=120 --retries=10 -e ".[dev]"
```

## Troubleshooting: SSL / corporate proxy networks

If you see SSL errors such as `SSLError`, `SSL_ERROR_SYSCALL`, or `UNEXPECTED_EOF_WHILE_READING`,
your network may be blocking or intercepting HTTPS traffic to PyPI.

Things to try:

- **Try a different network** (hotspot/home Wi‑Fi) or disable VPN/proxy temporarily.
- **Use your organization’s approved PyPI mirror** (if provided), e.g.:

```bash
pip install -e . --index-url <YOUR_MIRROR_URL>
pip install -e ".[dev]" --index-url <YOUR_MIRROR_URL>
```

- **Example public mirror** (if standard PyPI is blocked in your network):

```bash
pip install -e . --index-url https://mirror-pypi.runflare.com/simple
pip install -e ".[dev]" --index-url https://mirror-pypi.runflare.com/simple
```

- **If your proxy uses a custom CA certificate**, configure pip to trust it (ask IT for the CA bundle):

```bash
export REQUESTS_CA_BUNDLE=/path/to/your-ca-bundle.pem
pip install -e .
```

## Examples

```bash
# Make sure venv is activated!

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

