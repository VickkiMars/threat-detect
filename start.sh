#!/usr/bin/env bash
# ==============================================================================
# start.sh - Single-Command Launcher for AI Network Threat Detection System
# University of Uyo - B.Sc. Computer Science Research Project
# ==============================================================================
set -e

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
MODE="standard"
HOST="127.0.0.1"
PORT=5000
START_WORKER=0

# Parse arguments
print_usage() {
    echo "Usage: ./start.sh [options]"
    echo ""
    echo "Options:"
    echo "  --simulated, -s     Run under Raspberry Pi resource simulation (CPU Core 0, 2GB cgroup)"
    echo "  --with-worker, -w   Launch decoupled edge detection worker process alongside dashboard"
    echo "  --host <ip>         Bind dashboard to custom host (default: 127.0.0.1)"
    echo "  --port <port>       Bind dashboard to custom port (default: 5000)"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./start.sh                    # Standard dashboard startup"
    echo "  ./start.sh --simulated        # Edge-constrained benchmark mode"
    echo "  ./start.sh -s -w              # Simulated edge mode with decoupled background worker"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --simulated|-s)
            MODE="simulated"
            shift
            ;;
        --with-worker|-w)
            START_WORKER=1
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run './start.sh --help' for available options."
            exit 1
            ;;
    esac
done

echo "=================================================================="
echo "    AI NETWORK THREAT DETECTION GATEWAY (GRACE NIDS)"
echo "    Department of Computer Science - University of Uyo"
echo "=================================================================="

# Step 1: Ensure Python virtual environment exists
if [ ! -d ".venv" ]; then
    echo "[1/4] Virtual environment (.venv) not found. Initializing..."
    python3 -m venv .venv
    echo "[1/4] Installing dependencies from requirements.txt..."
    .venv/bin/pip install --upgrade pip --quiet
    .venv/bin/pip install -r requirements.txt --quiet
    echo "[1/4] Virtual environment configured successfully."
else
    echo "[1/4] Virtual environment detected at .venv."
fi

# Step 2: Ensure demo models and datasets exist
if [ ! -f "models/reference/hybrid_model.tflite" ] || [ ! -f "data/sample_flows.csv" ]; then
    echo "[2/4] Initializing calibrated reference models and sample flows..."
    .venv/bin/python3 scripts/seed_demo_data.py
    echo "[2/4] Calibration and demo data initialized."
else
    echo "[2/4] Quantized TFLite model and sample flow data verified."
fi

# Step 3: Ensure logs directory exists
mkdir -p logs

# Cleanup handler for graceful shutdown
WORKER_PID=""
cleanup() {
    echo ""
    echo "Shutting down AI Threat Detection Gateway..."
    if [ -n "$WORKER_PID" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "Stopping background inference worker (PID $WORKER_PID)..."
        kill "$WORKER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
    fi
    echo "Services stopped cleanly."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Step 4: Launch worker (optional) and web dashboard
echo "[3/4] Execution Mode: $([ "$MODE" = "simulated" ] && echo "Simulated Edge (Pinned Core 0, 2GB Memory Ceiling)" || echo "Standard Local Host")"

if [ "$START_WORKER" -eq 1 ]; then
    echo "[3/4] Launching decoupled edge inference worker process..."
    if [ "$MODE" = "simulated" ]; then
        ./scripts/run_simulated.sh .venv/bin/python3 -m src.inference_worker &
    else
        .venv/bin/python3 -m src.inference_worker &
    fi
    WORKER_PID=$!
    echo "[3/4] Inference worker started (PID $WORKER_PID)."
fi

echo "[4/4] Starting Flask Web Monitoring Dashboard..."
echo "------------------------------------------------------------------"
echo " Dashboard URL  : http://${HOST}:${PORT}"
echo " API Status     : http://${HOST}:${PORT}/api/status"
echo " SQLite DB      : data/threat_detection.db"
echo " Mode           : ${MODE}"
echo " Press Ctrl+C to stop all services."
echo "------------------------------------------------------------------"

export FLASK_RUN_HOST="$HOST"
export FLASK_RUN_PORT="$PORT"

if [ "$MODE" = "simulated" ]; then
    exec ./scripts/run_simulated.sh .venv/bin/python3 -m src.dashboard.app
else
    exec .venv/bin/python3 -m src.dashboard.app
fi
