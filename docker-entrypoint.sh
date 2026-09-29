#!/bin/bash
set -e

echo "[GRACE] Starting AI threat detection container on PORT ${PORT:-8000}..."

# Initialize SQLite schema and reference model registry
echo "[GRACE] Applying database schema and model registry initialization..."
if python scripts/seed_demo_data.py; then
    echo "[GRACE] Database schema and model registry initialized successfully."
else
    echo "[GRACE Warning] Schema initialization failed or storage is unreachable. Proceeding with web server startup..."
fi

echo "[GRACE] Executing command: $@"
exec "$@"
