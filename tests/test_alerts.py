"""
tests/test_alerts.py - Comprehensive Unit & Integration Tests for Security Alert Lifecycle
Covers severity thresholding (LOW, MEDIUM, HIGH, CRITICAL), sorting priority,
message formatting, database persistence, and REST API acknowledgment triage.
"""

import pytest
from src.dashboard.app import app
from src.alert_manager import (
    evaluate_threat_severity,
    format_alert_message,
    handle_malicious_detection
)
from src.database import (
    init_db,
    log_detection,
    create_alert,
    acknowledge_alert,
    get_unacknowledged_alerts,
    get_system_summary
)

@pytest.fixture(autouse=True)
def setup_database():
    init_db()

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_severity_thresholding_brackets():
    """Verifies that confidence scores map accurately to severity tiers."""
    assert evaluate_threat_severity(1.00) == "CRITICAL"
    assert evaluate_threat_severity(0.95) == "CRITICAL"
    assert evaluate_threat_severity(0.949) == "HIGH"
    assert evaluate_threat_severity(0.85) == "HIGH"
    assert evaluate_threat_severity(0.849) == "MEDIUM"
    assert evaluate_threat_severity(0.70) == "MEDIUM"
    assert evaluate_threat_severity(0.699) == "LOW"
    assert evaluate_threat_severity(0.50) == "LOW"
    assert evaluate_threat_severity(0.40) == "LOW"

def test_format_alert_message():
    """Verifies format of incident alert text."""
    msg = format_alert_message("HIGH", "198.51.100.15", "192.168.1.1", "TCP", 0.885, 42)
    assert "[HIGH THREAT DETECTED]" in msg
    assert "#42" in msg
    assert "198.51.100.15" in msg
    assert "192.168.1.1" in msg
    assert "TCP" in msg
    assert "88.5%" in msg

def test_handle_malicious_detection_pipeline():
    """Verifies end-to-end alert creation from flow detection."""
    flow_id = log_detection(
        model_id=1,
        src_ip="203.0.113.5",
        dst_ip="192.168.1.10",
        protocol="UDP",
        predicted_class=1,
        confidence=0.975,
        window_sequence_id=101,
        latency_ms=0.024
    )
    alert_id, severity, msg = handle_malicious_detection(
        flow_id=flow_id,
        confidence=0.975,
        src_ip="203.0.113.5",
        dst_ip="192.168.1.10",
        protocol="UDP",
        window_id=101
    )
    assert alert_id > 0
    assert severity == "CRITICAL"
    assert "203.0.113.5" in msg

def test_severity_priority_sorting():
    """Verifies that unacknowledged alerts are sorted by severity: CRITICAL > HIGH > MEDIUM > LOW."""
    flow_id = log_detection(
        model_id=1,
        src_ip="10.0.0.1",
        dst_ip="192.168.1.1",
        protocol="TCP",
        predicted_class=1,
        confidence=0.99,
        window_sequence_id=200,
        latency_ms=0.02
    )
    # Insert in reverse order: LOW, MEDIUM, HIGH, CRITICAL
    id_low = create_alert(flow_id, "LOW", "Test Low Alert")
    id_med = create_alert(flow_id, "MEDIUM", "Test Medium Alert")
    id_high = create_alert(flow_id, "HIGH", "Test High Alert")
    id_crit = create_alert(flow_id, "CRITICAL", "Test Critical Alert")

    alerts = get_unacknowledged_alerts(limit=50)
    filtered = [a for a in alerts if a["alert_id"] in (id_low, id_med, id_high, id_crit)]
    severities = [a["severity"] for a in filtered]

    # Highest priority should appear first
    assert severities[0] == "CRITICAL"
    assert "HIGH" in severities
    assert "MEDIUM" in severities
    assert "LOW" in severities

    # Clean up by acknowledging
    for aid in (id_low, id_med, id_high, id_crit):
        acknowledge_alert(aid)

def test_rest_api_acknowledgement_lifecycle(client):
    """Verifies full REST API alert lifecycle: creation, triage query, and acknowledgment."""
    flow_id = log_detection(
        model_id=1,
        src_ip="10.0.0.99",
        dst_ip="192.168.1.1",
        protocol="TCP",
        predicted_class=1,
        confidence=0.96,
        window_sequence_id=300,
        latency_ms=0.022
    )
    alert_id = create_alert(flow_id, "CRITICAL", "API Triage Verification Alert")

    # 1. Fetch unacknowledged alerts
    res = client.get("/api/alerts/unacknowledged")
    assert res.status_code == 200
    alerts = res.get_json()["alerts"]
    assert any(a["alert_id"] == alert_id for a in alerts)

    # 2. Acknowledge alert via POST
    ack_res = client.post(f"/api/alerts/{alert_id}/acknowledge")
    assert ack_res.status_code == 200
    ack_json = ack_res.get_json()
    assert ack_json["status"] == "success"
    assert ack_json["alert_id"] == alert_id

    # 3. Duplicate acknowledgment must return 404 (already acknowledged)
    dup_res = client.post(f"/api/alerts/{alert_id}/acknowledge")
    assert dup_res.status_code == 404
    assert dup_res.get_json()["status"] == "error"

    # 4. Non-existent alert ID must return 404
    non_res = client.post("/api/alerts/9999999/acknowledge")
    assert non_res.status_code == 404

    # 5. Alert should no longer appear in unacknowledged list
    res_after = client.get("/api/alerts/unacknowledged")
    alerts_after = res_after.get_json()["alerts"]
    assert not any(a["alert_id"] == alert_id for a in alerts_after)

def test_rest_api_acknowledge_all_lifecycle(client):
    """Verifies bulk acknowledgment of all pending security threats."""
    flow_id = log_detection(
        model_id=1,
        src_ip="10.0.0.55",
        dst_ip="192.168.1.1",
        protocol="TCP",
        predicted_class=1,
        confidence=0.98,
        window_sequence_id=400,
        latency_ms=0.021
    )
    # Create multiple alerts
    id1 = create_alert(flow_id, "CRITICAL", "Bulk Test 1")
    id2 = create_alert(flow_id, "HIGH", "Bulk Test 2")

    # Verify unacknowledged
    res = client.get("/api/alerts/unacknowledged")
    assert res.status_code == 200
    ids_before = [a["alert_id"] for a in res.get_json()["alerts"]]
    assert id1 in ids_before and id2 in ids_before

    # Call acknowledge_all
    ack_res = client.post("/api/alerts/acknowledge_all")
    assert ack_res.status_code == 200
    ack_data = ack_res.get_json()
    assert ack_data["status"] == "success"
    assert ack_data["acknowledged_count"] >= 2

    # Verify unacknowledged is now empty of these alerts
    res_after = client.get("/api/alerts/unacknowledged")
    ids_after = [a["alert_id"] for a in res_after.get_json()["alerts"]]
    assert id1 not in ids_after and id2 not in ids_after
