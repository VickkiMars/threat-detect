#!/usr/bin/env bash
# ==============================================================================
# scripts/run_simulated.sh - Workstation Simulation Launcher
# Enforces CPU Affinity, Cgroup Memory Ceiling, and Single-Threaded Execution
# Corresponding to Chapter 4.10.2 of University of Uyo Dissertation
# ==============================================================================
set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command to run under simulation> [args...]"
    echo "Example: $0 .venv/bin/python3 -m src.benchmark_runtime"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=-1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# Log simulation configuration
echo "=== [Simulated Edge Platform Configuration] ==="
echo "Host Machine         : $(uname -m) Linux"
echo "Logical CPU Pinning  : Core 0 (taskset -c 0)"
echo "Memory Ceiling Target: 2.0 GB (512 MB service budget)"
echo "GPU Acceleration     : DISABLED (CUDA_VISIBLE_DEVICES=-1)"
echo "TFLite Concurrency   : 1 Thread (intra/inter-op = 1)"
echo "Command to Execute   : $@"
echo "================================================"

# Check if systemd-run --user works in this environment
USE_SYSTEMD=0
if command -v systemd-run >/dev/null 2>&1; then
    if systemd-run --user --scope -p MemoryMax=2G -p OOMPolicy=stop true >/dev/null 2>&1; then
        USE_SYSTEMD=1
    fi
fi

if [ "$USE_SYSTEMD" -eq 1 ]; then
    echo "[Simulation] Executing with systemd-run cgroups v2..."
    exec taskset --cpu-list 0 systemd-run --user --scope \
      -p MemoryMax=2G \
      -p OOMPolicy=stop \
      env CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
      "$@"
else
    echo "[Simulation] Executing with ulimit fallback (2GB virtual memory ceiling)..."
    ulimit -v 2097152 2>/dev/null || true
    exec taskset --cpu-list 0 env CUDA_VISIBLE_DEVICES=-1 \
      TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
      "$@"
fi
