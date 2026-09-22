"""
tests/test_api.py - Integration Tests for Flask REST API Endpoints & Dashboard
Verifies status queries, detection streams, alert retrieval, and alert acknowledgement.
"""

import pytest
import json
from src.dashboard.app import app
from src.database import log_detection, create_alert

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_get_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"AI Network Threat Detection Gateway" in res.data
    assert b"psutil" in res.data

def test_api_status_endpoint(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.get_json()
    assert "total_flows_processed" in data
    assert "active_model" in data
    assert "latest_telemetry" in data

def test_api_recent_detections(client):
    res = client.get("/api/detections/recent?limit=10")
    assert res.status_code == 200
    data = res.get_json()
    assert "detections" in data
    assert isinstance(data["detections"], list)

def test_api_alerts_and_acknowledgement(client):
    # Log flow and create alert
    flow_id = log_detection(
        model_id=1,
        src_ip="10.10.10.10",
        dst_ip="192.168.1.1",
        protocol="TCP",
        predicted_class=1,
        confidence=0.99,
        window_sequence_id=999,
        latency_ms=0.03
    )
    alert_id = create_alert(flow_id, "CRITICAL", "Test API Threat Alert")
    
    # Check unacknowledged list
    res = client.get("/api/alerts/unacknowledged")
    assert res.status_code == 200
    alerts = res.get_json()["alerts"]
    alert_ids = [a["alert_id"] for a in alerts]
    assert alert_id in alert_ids
    
    # Acknowledge via POST
    ack_res = client.post(f"/api/alerts/{alert_id}/acknowledge")
    assert ack_res.status_code == 200
    ack_data = ack_res.get_json()
    assert ack_data["status"] == "success"
    
    # Verify no longer in unacknowledged list
    res_after = client.get("/api/alerts/unacknowledged")
    alerts_after = res_after.get_json()["alerts"]
    assert alert_id not in [a["alert_id"] for a in alerts_after]

def test_api_system_metrics(client):
    res = client.get("/api/metrics/system?limit=5")
    assert res.status_code == 200
    data = res.get_json()
    assert "metrics" in data
    assert isinstance(data["metrics"], list)
