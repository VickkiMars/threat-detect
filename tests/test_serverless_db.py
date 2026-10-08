"""
tests/test_serverless_db.py - Regression Test for Serverless Database Bootstrap & Model Integrity
Verifies that on fresh/serverless environments, init_db ensures an active model exists
in model_registry so that log_detection never fails with a foreign key constraint violation.
"""

import sqlite3
import pytest
from pathlib import Path
from src.database import init_db, log_detection, get_active_model

def test_init_db_ensures_model_registry_integrity_on_fresh_db(tmp_path):
    """Verifies that a completely fresh database can log detections without FK failure."""
    fresh_db = tmp_path / "fresh_test.db"
    init_db(fresh_db)
    
    # Active model must be present
    model = get_active_model(fresh_db)
    assert model is not None, "A fresh database must have at least one registered reference model"
    assert model["model_id"] >= 1
    
    # log_detection must succeed without sqlite3.IntegrityError
    flow_id = log_detection(
        model_id=model["model_id"],
        src_ip="10.0.0.1",
        dst_ip="192.168.1.1",
        protocol="TCP",
        predicted_class=1,
        confidence=0.98,
        window_sequence_id=1,
        latency_ms=0.05,
        db_path=fresh_db
    )
    assert flow_id is not None
    assert flow_id > 0
