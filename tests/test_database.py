"""
tests/test_database.py - Unit Tests for SQLite Event & Telemetry Persistence
Verifies schema initialization, WAL pragma, CRUD queries, and alert triage.
"""

import pytest
import sqlite3
from pathlib import Path
import tempfile
import os

from src.database import (
    init_db, register_model, get_active_model,
    log_detection, create_alert, acknowledge_alert,
    get_recent_detections, get_unacknowledged_alerts,
    record_metrics, get_system_summary, get_connection
)

@pytest.fixture
def temp_db():
    """Provides a fresh temporary SQLite database path."""
    fd, path_str = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_path = Path(path_str)
    init_db(db_path)
    register_model(
        version="v1.0.0-fixture",
        model_format="TFLITE",
        file_path="/tmp/fixture.tflite",
        file_size_kb=50.0,
        test_accuracy=0.99,
        test_macro_f1=0.99,
        dataset_origin="UNSW-NB15",
        is_active=1,
        db_path=db_path
    )
    yield db_path
    if db_path.exists():
        db_path.unlink()
    # Clean WAL artifacts if present
    for extra in [f"{path_str}-wal", f"{path_str}-shm"]:
        if os.path.exists(extra):
            os.remove(extra)

def test_init_db_creates_tables_and_wal(temp_db):
    conn = get_connection(temp_db)
    try:
        # Verify WAL mode
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"
        
        # Verify tables exist
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        assert "detection_log" in tables
        assert "alert" in tables
        assert "system_metrics" in tables
        assert "model_registry" in tables
    finally:
        conn.close()

def test_model_registry_crud(temp_db):
    model_id = register_model(
        version="v1.0.0-test",
        model_format="TFLITE",
        file_path="/tmp/model.tflite",
        file_size_kb=56.88,
        test_accuracy=0.9984,
        test_macro_f1=0.9982,
        dataset_origin="UNSW-NB15",
        is_active=1,
        db_path=temp_db
    )
    assert model_id > 0
    active = get_active_model(temp_db)
    assert active is not None
    assert active["version"] == "v1.0.0-test"
    assert active["is_active"] == 1

def test_log_detection_and_query(temp_db):
    flow_id = log_detection(
        model_id=1,
        src_ip="192.168.1.50",
        dst_ip="192.168.1.1",
        protocol="TCP",
        predicted_class=1,
        confidence=0.985,
        window_sequence_id=42,
        latency_ms=0.035,
        db_path=temp_db
    )
    assert flow_id > 0
    
    detections = get_recent_detections(limit=10, db_path=temp_db)
    assert len(detections) == 1
    assert detections[0]["flow_id"] == flow_id
    assert detections[0]["src_ip"] == "192.168.1.50"
    assert detections[0]["predicted_class"] == 1

def test_alert_lifecycle(temp_db):
    flow_id = log_detection(
        model_id=1,
        src_ip="10.0.0.99",
        dst_ip="192.168.1.1",
        protocol="TCP",
        predicted_class=1,
        confidence=0.96,
        window_sequence_id=1,
        latency_ms=0.04,
        db_path=temp_db
    )
    
    alert_id = create_alert(
        flow_id=flow_id,
        severity="CRITICAL",
        message="Critical SYN Flood Detected",
        db_path=temp_db
    )
    assert alert_id > 0
    
    # Query unacknowledged
    unack = get_unacknowledged_alerts(db_path=temp_db)
    assert len(unack) == 1
    assert unack[0]["alert_id"] == alert_id
    assert unack[0]["severity"] == "CRITICAL"
    
    # Acknowledge
    success = acknowledge_alert(alert_id, db_path=temp_db)
    assert success is True
    
    # Query again
    unack_after = get_unacknowledged_alerts(db_path=temp_db)
    assert len(unack_after) == 0

def test_system_metrics_and_summary(temp_db):
    sample_id = record_metrics(
        cpu_percent=12.5,
        memory_rss_mb=49.9,
        throughput_fps=300.0,
        host_status="NORMAL",
        db_path=temp_db
    )
    assert sample_id > 0
    
    summary = get_system_summary(db_path=temp_db)
    assert "total_flows_processed" in summary
    assert "latest_telemetry" in summary
    assert summary["latest_telemetry"]["cpu_percent"] == 12.5
