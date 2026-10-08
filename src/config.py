"""
src/config.py - Central Configuration for Edge Threat Detection System
Defines paths, resource ceilings, model hyperparameters, and severity brackets.
"""

import os
from pathlib import Path

# Serverless Execution Detection (Vercel / AWS Lambda)
IS_SERVERLESS = bool(
    os.environ.get("VERCEL")
    or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    or os.environ.get("SERVERLESS")
)

# Persistent Container Detection (Docker / Render / Cloud Run)
# A long-lived container keeps its SQLite database on a writable volume, so it
# uses the repository data/ directory instead of the /tmp serverless scratch space.
IS_CONTAINER = bool(os.environ.get("GRACE_CONTAINER"))

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

if IS_SERVERLESS and not IS_CONTAINER:
    LOGS_DIR = Path("/tmp/logs")
    DB_PATH = Path("/tmp/threat_detection.db")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Automatically bootstrap ephemeral /tmp database from repository seed
    SEED_DB = DATA_DIR / "threat_detection.db"
    if not DB_PATH.exists() and SEED_DB.exists():
        try:
            import shutil
            shutil.copy2(SEED_DB, DB_PATH)
        except Exception as e:
            print(f"[GRACE Config] Notice copying seed DB to /tmp: {e}")
else:
    LOGS_DIR = PROJECT_ROOT / "logs"
    DB_PATH = DATA_DIR / "threat_detection.db"


SAMPLE_FLOWS_PATH = DATA_DIR / "sample_flows.csv"

# Preprocessing & Model Architecture Constants
WINDOW_SIZE = 10           # Sequence length: 10 consecutive flow records
PCA_COMPONENTS = 22        # Dimensionality reduction components (retains >=95% variance)
PCA_VARIANCE_TARGET = 0.95 # Minimum cumulative explained variance
CLASS_LABELS = {0: "Benign", 1: "Malicious Attack"}

# Model Artifact Paths
REFERENCE_MODEL_DIR = MODELS_DIR / "reference"
TFLITE_MODEL_PATH = REFERENCE_MODEL_DIR / "hybrid_model.tflite"
SCALER_PATH = REFERENCE_MODEL_DIR / "scaler.joblib"
PCA_PATH = REFERENCE_MODEL_DIR / "pca.joblib"
METADATA_PATH = REFERENCE_MODEL_DIR / "metadata.json"

# Alert Severity Classification Brackets (Confidence P)
ALERT_SEVERITY_LEVELS = [
    (0.95, "CRITICAL"),
    (0.85, "HIGH"),
    (0.70, "MEDIUM"),
    (0.50, "LOW"),
]

# Non-Functional Performance Ceilings (NFR Contracts)
NFR_TARGETS = {
    "NFR1_MAX_LATENCY_MS": 50.0,       # Max mean inference latency per 10-flow window
    "NFR2_MAX_MODEL_SIZE_KB": 1000.0,  # Max model file size (target <= 150 KB)
    "NFR3_MAX_MEMORY_RSS_MB": 512.0,   # Max sustained memory for detection service
    "NFR4_MIN_ACCURACY": 0.95,         # Target held-out test accuracy
    "NFR5_CPU_ONLY": True,             # Zero GPU acceleration
}

# Edge Simulation Settings
SIMULATION_CONFIG = {
    "CPU_CORE": 0,
    "MEMORY_CEILING_GB": 2.0,
    "DEFAULT_REPLAY_RATE": 30,         # Flows per second streamed by replay engine
    "TELEMETRY_INTERVAL_SEC": 1.0,     # psutil sampling frequency
    # Dashboard bind address / port honour $PORT (Vercel, Render, Fly.io) and
    # GRACE_DASHBOARD_HOST so a single container image serves cloud and edge.
    "DASHBOARD_PORT": int(os.environ.get("PORT", os.environ.get("FLASK_RUN_PORT", 5000))),
    "DASHBOARD_HOST": os.environ.get("GRACE_DASHBOARD_HOST", "127.0.0.1"),
}

# Container Concurrency Switch
# Gunicorn/uWSGI workers must not spawn the embedded replay thread alongside the
# standalone inference worker (double-consumption of the flow stream). Dashboard
# polling drives simulation steps instead, which is the verified serverless path.
DISABLE_EMBEDDED_THREADING = os.environ.get("GRACE_DISABLE_THREADING", "").lower() in ("1", "true", "yes")

# Trusted Host Allow-List (mirrors Django ALLOWED_HOSTS semantics)
_HOSTS_ENV = os.environ.get(
    "GRACE_ALLOWED_HOSTS",
    "127.0.0.1,localhost,0.0.0.0,.vercel.app",
)
ALLOWED_HOSTS = [host.strip() for host in _HOSTS_ENV.split(",") if host.strip()]

# Behind a cloud load balancer (Vercel/Render proxies) the Host header is the
# platform hostname, so host pinning is disabled in favour of wildcard suffixes.
HOST_ALLOW_ALL = os.environ.get("GRACE_HOST_ALLOW_ALL", "").lower() in ("1", "true", "yes")


def host_is_allowed(host_header: str) -> bool:
    """Validates an inbound HTTP Host header against ALLOWED_HOSTS conventions."""
    if HOST_ALLOW_ALL or not ALLOWED_HOSTS:
        return True
    hostname = (host_header or "").split(":")[0].strip().lower()
    if not hostname:
        return True
    for allowed in ALLOWED_HOSTS:
        candidate = allowed.lower()
        if candidate == hostname:
            return True
        # Leading-dot entries match the domain and any subdomain (.vercel.app)
        if candidate.startswith(".") and (
            hostname == candidate[1:] or hostname.endswith(candidate)
        ):
            return True
        if candidate == "*":
            return True
    return False


# Demo Data Bootstrap Switch
# Long-lived containers seed their SQLite schema at startup (docker-entrypoint.sh,
# Render startCommand); serverless functions rely on the pre-seeded bundled db.
SEED_ON_STARTUP = os.environ.get("GRACE_SEED_ON_STARTUP", "").lower() in ("1", "true", "yes")

