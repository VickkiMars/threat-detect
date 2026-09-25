#!/usr/bin/env bash
# ==============================================================================
# setup_env.sh - Automated Environment Setup for AI Threat Detection System
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== [1/4] Verifying Host Environment ==="
echo "Operating System : $(uname -s) $(uname -r)"
echo "Host Machine     : $(uname -m)"
echo "Python Executable: $(which python3)"
python3 --version

echo ""
echo "=== [2/4] Setting Up Python Virtual Environment ==="
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment at .venv..."
    python3 -m venv .venv
else
    echo "Existing virtual environment detected at .venv."
fi

echo ""
echo "=== [3/4] Installing Required Dependencies ==="
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo ""
echo "=== [4/4] Verifying Core Package Imports ==="
.venv/bin/python3 -c "
import ai_edge_litert
import sklearn
import pandas
import numpy
import flask
import psutil
import sqlite3
print('All core dependencies successfully imported!')
print('ai-edge-litert version:', ai_edge_litert.__version__ if hasattr(ai_edge_litert, '__version__') else 'Loaded')
print('scikit-learn version   :', sklearn.__version__)
print('pandas version         :', pandas.__version__)
print('numpy version          :', numpy.__version__)
print('flask version          :', flask.__version__)
print('psutil version         :', psutil.__version__)
print('SQLite library version :', sqlite3.sqlite_version)
"

echo ""
echo "======================================================================"
echo " Environment setup complete! Activate with: source .venv/bin/activate"
echo "======================================================================"
