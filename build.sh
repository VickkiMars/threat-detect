#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "[Render Build] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[Render Build] Warming SQLite schema and reference model registry..."
python scripts/seed_demo_data.py

echo "[Render Build] Build completed successfully."
