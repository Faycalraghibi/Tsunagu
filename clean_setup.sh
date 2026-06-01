#!/usr/bin/env bash
set -euo pipefail

VENV_NAME="tsunagu-venv"

echo "[1/4] Cleaning previous environment..."
rm -rf "$VENV_NAME" .pytest_cache .mypy_cache .ruff_cache
find . -type d -name __pycache__ -not -path "./$VENV_NAME/*" -exec rm -rf {} + 2>/dev/null || true

echo "[2/4] Creating virtual environment: $VENV_NAME (Python 3.12)"
if command -v py >/dev/null 2>&1; then py -3.12 -m venv "$VENV_NAME"; else python3.12 -m venv "$VENV_NAME"; fi
if [ -f "$VENV_NAME/Scripts/activate" ]; then source "$VENV_NAME/Scripts/activate"; else source "$VENV_NAME/bin/activate"; fi

echo "[3/4] Installing CPU torch, then the project with dev dependencies..."
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"

echo "[4/4] Installing pre-commit hooks..."
pre-commit install

echo ""
echo "Setup complete. Activate with: source $VENV_NAME/Scripts/activate"
