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
import re
from urllib.parse import parse_qs, urlencode

class VercelPathFixMiddleware:
    """Fixes PATH_INFO for Flask when running behind Vercel serverless rewrites."""
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        query_string = environ.get("QUERY_STRING", "")
        if "__vercel_original_path" in query_string:
            params = parse_qs(query_string, keep_blank_values=True)
            if "__vercel_original_path" in params:
                raw_path = params.pop("__vercel_original_path")[0]
                normalized_path = "/" + raw_path.lstrip("/")
                normalized_path = re.sub(r"/+", "/", normalized_path)
                environ["PATH_INFO"] = normalized_path
                environ["QUERY_STRING"] = urlencode(params, doseq=True)
        elif environ.get("PATH_INFO") in ("/api/index", "/api/index.py", ""):
            environ["PATH_INFO"] = "/"
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)
