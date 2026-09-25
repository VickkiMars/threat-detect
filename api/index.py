"""
api/index.py - Vercel Serverless Function Entry Point for Grace NIDS Dashboard
Initializes serverless SQLite environment in /tmp and exposes the Flask WSGI app.
"""

import os
import sys
import shutil
from pathlib import Path

# Explicitly declare serverless environment before importing project config
os.environ["VERCEL"] = "1"
os.environ["SERVERLESS"] = "1"

# Add repository root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Prepare writable /tmp directories for serverless runtime
TMP_DATA_DIR = Path("/tmp/data")
TMP_LOGS_DIR = Path("/tmp/logs")
TMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
TMP_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Copy initial pre-seeded SQLite database to /tmp if not already present
TMP_DB_PATH = Path("/tmp/threat_detection.db")
SEED_DB_PATH = ROOT_DIR / "data" / "threat_detection.db"

if not TMP_DB_PATH.exists() and SEED_DB_PATH.exists():
    try:
        shutil.copy2(SEED_DB_PATH, TMP_DB_PATH)
    except Exception as e:
        print(f"[Vercel Init] Notice: Could not copy seed database to /tmp: {e}")

# Import Flask app instance
from src.dashboard.app import app

# Vercel serverless handler expects 'app'
# Ready for HTTP request handling
