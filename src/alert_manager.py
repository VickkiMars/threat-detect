"""
src/alert_manager.py - Security Alert Generation & Triage Logic
Classifies threat severity and generates prioritized security operations alerts.
"""

from typing import Tuple, Optional
from src.config import ALERT_SEVERITY_LEVELS
from src.database import create_alert

def evaluate_threat_severity(confidence: float) -> str:
    """Categorizes threat severity based on prediction confidence."""
    for threshold, level in ALERT_SEVERITY_LEVELS:
        if confidence >= threshold:
            return level
    return "LOW"

def format_alert_message(
    severity: str,
    src_ip: str,
    dst_ip: str,
    protocol: str,
    confidence: float,
    window_id: int
) -> str:
    """Generates a human-readable security incident alert description."""
    return (
        f"[{severity} THREAT DETECTED] Malicious sequence window #{window_id} "
        f"originating from {src_ip} targeting {dst_ip} ({protocol.upper()}) "
        f"with {confidence:.1%} model confidence."
    )

def handle_malicious_detection(
    flow_id: int,
    confidence: float,
    src_ip: str,
    dst_ip: str,
    protocol: str,
    window_id: int
) -> Tuple[int, str, str]:
    """Generates severity, constructs message, and persists alert to database."""
    severity = evaluate_threat_severity(confidence)
    message = format_alert_message(severity, src_ip, dst_ip, protocol, confidence, window_id)
    alert_id = create_alert(flow_id, severity, message)
    return alert_id, severity, message
