#!/bin/bash
set -e

echo "[GRACE] Starting AI threat detection container on PORT ${PORT:-8000}..."

echo "[GRACE] Executing command: $@"
exec "$@"
