#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  EMTL Solver – one-click launcher  (Linux / macOS)
#  Creates a venv, installs deps, then starts the solver REPL
#  and the Streamlit web app side-by-side.
# ─────────────────────────────────────────────────────────────
set -e

REQUIRED_PYTHON_VERSION="3.11"
REQUIRED_PYTHON_FULL="3.11.8"
VENV_DIR=".venv"

# ── resolve project root (one level up from this script) ─────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║   EMTL Solver – Automated Setup & Launcher      ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""

# ── locate a suitable Python 3.11 interpreter ───────────────
find_python() {
    # Try common names in order of specificity
    for cmd in "python${REQUIRED_PYTHON_VERSION}" "python3.11" "python3" "python"; do
        if command -v "$cmd" &>/dev/null; then
            version=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+')
            if [ "$version" = "$REQUIRED_PYTHON_VERSION" ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_CMD=$(find_python) || {
    echo "  ERROR: Python $REQUIRED_PYTHON_VERSION is required but was not found."
    echo ""
    echo "  Please install Python $REQUIRED_PYTHON_FULL:"
    echo "    - macOS:  brew install python@3.11"
    echo "    - Ubuntu: sudo apt install python3.11 python3.11-venv"
    echo "    - Fedora: sudo dnf install python3.11"
    echo ""
    exit 1
}

PYTHON_FULL_VERSION=$("$PYTHON_CMD" --version 2>&1 | grep -oP '\d+\.\d+\.\d+')
echo "  [✓] Found Python $PYTHON_FULL_VERSION ($PYTHON_CMD)"

# ── create virtual environment if it doesn't exist ───────────
if [ ! -d "$VENV_DIR" ]; then
    echo "  [·] Creating virtual environment in $VENV_DIR ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "  [✓] Virtual environment created."
else
    echo "  [✓] Virtual environment already exists."
fi

# ── activate venv ────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── PyPI mirror (with fallback) ──────────────────────────────
PYPI_MIRROR="https://mirror-pypi.runflare.com/simple"
if curl -s -o /dev/null --connect-timeout 10 "$PYPI_MIRROR" 2>/dev/null; then
    PIP_INDEX="--index-url $PYPI_MIRROR"
    echo "  [✓] Using PyPI mirror: $PYPI_MIRROR"
else
    PIP_INDEX=""
    echo "  [!] Mirror unreachable, falling back to default PyPI."
fi

# ── install / update dependencies ────────────────────────────
if [ ! -f "$VENV_DIR/.deps_installed" ]; then
    echo "  [·] Installing dependencies (first run) ..."
    pip install --upgrade pip $PIP_INDEX
    pip install -e . $PIP_INDEX
    touch "$VENV_DIR/.deps_installed"
    echo "  [✓] Dependencies installed."
else
    echo "  [✓] Dependencies already installed (delete $VENV_DIR/.deps_installed to reinstall)."
fi

# ── cleanup on exit ──────────────────────────────────────────
STREAMLIT_PID=""
cleanup() {
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo ""
        echo "  [·] Shutting down Streamlit ..."
        kill "$STREAMLIT_PID" 2>/dev/null
        wait "$STREAMLIT_PID" 2>/dev/null
        echo "  [✓] Streamlit stopped."
    fi
}
trap cleanup EXIT INT TERM

# ── launch Streamlit in the background ───────────────────────
echo ""
echo "  [·] Starting Streamlit web app ..."
STREAMLIT_LOG=$(mktemp)
streamlit run web/app.py --server.headless true > "$STREAMLIT_LOG" 2>&1 &
STREAMLIT_PID=$!
sleep 3

if kill -0 "$STREAMLIT_PID" 2>/dev/null; then
    echo "  [✓] Streamlit started (PID $STREAMLIT_PID)"
else
    echo "  [!] Warning: Streamlit may not have started correctly."
fi
echo ""
echo "  ── Streamlit log ──────────────────────────────────────"
cat "$STREAMLIT_LOG"
echo "  ───────────────────────────────────────────────────────"
rm -f "$STREAMLIT_LOG"

# ── launch the interactive solver ────────────────────────────
echo ""
echo "  ─────────────────────────────────────────────────────"
echo "  Starting interactive solver ... (Ctrl+C to quit)"
echo "  ─────────────────────────────────────────────────────"
echo ""

python emtl_solver.py

echo ""
echo "  Done. Goodbye!"
echo ""
