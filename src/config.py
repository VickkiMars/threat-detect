"""
src/config.py - Central Configuration for Edge Threat Detection System
Defines paths, resource ceilings, model hyperparameters, and severity brackets.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
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
    "DASHBOARD_PORT": 5000,
    "DASHBOARD_HOST": "127.0.0.1",
}
