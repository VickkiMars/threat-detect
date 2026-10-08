"""
wsgi.py - WSGI Application Entry Point for Grace NIDS Dashboard
Exposes the Flask WSGI application instance for Vercel Serverless and Gunicorn.
"""

import os
import sys
from pathlib import Path

# Explicitly declare serverless environment if executing on cloud platforms
if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    os.environ["SERVERLESS"] = "1"

# Add repository root to Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import Flask WSGI application instance
from src.dashboard.app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
