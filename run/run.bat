@echo off
setlocal EnableDelayedExpansion
:: ─────────────────────────────────────────────────────────────
::  EMTL Solver – one-click launcher  (Windows)
::  Creates a venv, installs deps, then starts the solver REPL
::  and the Streamlit web app side-by-side.
:: ─────────────────────────────────────────────────────────────

set "REQUIRED_PYTHON_VERSION=3.11"
set "REQUIRED_PYTHON_FULL=3.11.8"
set "VENV_DIR=.venv"

:: ── resolve project root (one level up from this script) ─────
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."
set "PROJECT_DIR=%CD%"

echo.
echo   ╔══════════════════════════════════════════════════╗
echo   ║   EMTL Solver – Automated Setup ^& Launcher      ║
echo   ╚══════════════════════════════════════════════════╝
echo.

:: ── locate a suitable Python 3.11 interpreter ───────────────
set "PYTHON_CMD="

:: Try py launcher first (common on Windows)
where py >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=*" %%v in ('py -3.11 --version 2^>nul') do (
        echo %%v | findstr /C:"3.11" >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=py -3.11"
        )
    )
)

:: Try python3.11
if not defined PYTHON_CMD (
    where python3.11 >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python3.11"
    )
)

:: Try python and check version
if not defined PYTHON_CMD (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do (
            echo %%v | findstr /B /C:"3.11" >nul 2>&1
            if !errorlevel! equ 0 (
                set "PYTHON_CMD=python"
            )
        )
    )
)

if not defined PYTHON_CMD (
    echo   ERROR: Python %REQUIRED_PYTHON_VERSION% is required but was not found.
    echo.
    echo   Please install Python %REQUIRED_PYTHON_FULL% from:
    echo     https://www.python.org/downloads/release/python-3118/
    echo.
    echo   Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Get full version string
for /f "tokens=2 delims= " %%v in ('!PYTHON_CMD! --version 2^>^&1') do set "PYTHON_FULL_VERSION=%%v"
echo   [✓] Found Python !PYTHON_FULL_VERSION! (!PYTHON_CMD!)

:: ── create virtual environment if it doesn't exist ───────────
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo   [·] Creating virtual environment in %VENV_DIR% ...
    !PYTHON_CMD! -m venv "%VENV_DIR%"
    echo   [✓] Virtual environment created.
) else (
    echo   [✓] Virtual environment already exists.
)

:: ── activate venv ────────────────────────────────────────────
call "%VENV_DIR%\Scripts\activate.bat"

:: ── PyPI mirror (with fallback) ──────────────────────────────
set "PYPI_MIRROR=https://mirror-pypi.runflare.com/simple"
set "PIP_INDEX="
curl -s -o nul --connect-timeout 3 "%PYPI_MIRROR%" >nul 2>&1
if !errorlevel! equ 0 (
    set "PIP_INDEX=--index-url %PYPI_MIRROR%"
    echo   [✓] Using PyPI mirror: %PYPI_MIRROR%
) else (
    echo   [!] Mirror unreachable, falling back to default PyPI.
)

:: ── install / update dependencies ────────────────────────────
if not exist "%VENV_DIR%\.deps_installed" (
    echo   [·] Installing dependencies (first run^) ...
    pip install --upgrade pip !PIP_INDEX!
    pip install -e . !PIP_INDEX!
    echo. > "%VENV_DIR%\.deps_installed"
    echo   [✓] Dependencies installed.
) else (
    echo   [✓] Dependencies already installed (delete %VENV_DIR%\.deps_installed to reinstall^).
)

:: ── launch Streamlit in the background ───────────────────────
echo.
echo   [·] Starting Streamlit web app ...
start "" /B cmd /c "streamlit run web/app.py --server.headless true >nul 2>&1"
timeout /t 3 /nobreak >nul
echo   [✓] Streamlit is running at http://localhost:8501

:: ── launch the interactive solver ────────────────────────────
echo.
echo   ─────────────────────────────────────────────────────
echo   Starting interactive solver ... (Ctrl+C to quit)
echo   ─────────────────────────────────────────────────────
echo.

python emtl_solver.py

:: ── cleanup ──────────────────────────────────────────────────
echo.
echo   [·] Shutting down Streamlit ...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8501" ^| findstr "LISTENING"') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   [✓] Streamlit stopped.
echo.
echo   Done. Goodbye!
echo.

popd
endlocal
pause
