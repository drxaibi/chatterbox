#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Checking Python 3.11..."
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Python 3.11 is required. Please install python3.11 first."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if not (sys.version_info.major == 3 and sys.version_info.minor == 11):
    raise SystemExit("Python 3.11 is required")
print("Python version OK")
PY

echo "[2/4] Creating virtual environment..."
"$PYTHON_BIN" -m venv .venv

VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Virtual environment python not found at $VENV_PY"
  exit 1
fi

echo "[3/4] Installing project dependencies..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
"$VENV_PY" -m pip install -e .

echo "[4/4] Downloading model checkpoints (first run can take a while)..."
"$VENV_PY" tools/warmup_models.py --models all

echo "Bootstrap complete."
echo "Run apps using the corresponding Python commands in this repo."
