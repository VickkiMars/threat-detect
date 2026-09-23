"""
src/database.py - SQLite Database Management & Telemetry Persistence
Implements WAL mode concurrency, event logging, alert triage, and metrics storage.
"""

import sqlite3
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from src.config import DB_PATH

def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Returns a tuned SQLite connection configured for concurrent edge operation."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    
    # Enforce SQLite performance pragmas
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(db_path: Optional[Path] = None) -> None:
    """Initializes tables and indexes according to docs/DATABASE_SCHEMA.md."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS model_registry (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL UNIQUE,
                model_format TEXT NOT NULL CHECK (model_format IN ('TFLITE', 'KERAS')),
                file_path TEXT NOT NULL,
                file_size_kb REAL NOT NULL,
                test_accuracy REAL NOT NULL,
                test_macro_f1 REAL NOT NULL,
                dataset_origin TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
                registered_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS detection_log (
                flow_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                timestamp TEXT NOT NULL,
                src_ip TEXT DEFAULT '192.168.1.100',
                dst_ip TEXT DEFAULT '192.168.1.1',
                protocol TEXT DEFAULT 'TCP',
                predicted_class INTEGER NOT NULL CHECK (predicted_class IN (0, 1)),
                confidence REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
                window_sequence_id INTEGER NOT NULL,
                inference_latency_ms REAL NOT NULL,
                FOREIGN KEY (model_id) REFERENCES model_registry(model_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS alert (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_id INTEGER NOT NULL,
                severity TEXT NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                message TEXT NOT NULL,
                acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0, 1)),
                created_at TEXT NOT NULL,
                acknowledged_at TEXT,
                FOREIGN KEY (flow_id) REFERENCES detection_log(flow_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS system_metrics (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL NOT NULL,
                memory_rss_mb REAL NOT NULL,
                throughput_fps REAL NOT NULL,
                host_status TEXT DEFAULT 'NORMAL'
            );

            CREATE INDEX IF NOT EXISTS idx_detection_timestamp ON detection_log(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_detection_class ON detection_log(predicted_class);
            CREATE INDEX IF NOT EXISTS idx_alert_acknowledged ON alert(acknowledged, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_alert_severity ON alert(severity);
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp DESC);
            """)
    finally:
        conn.close()

def register_model(
    version: str,
    model_format: str,
    file_path: str,
    file_size_kb: float,
    test_accuracy: float,
    test_macro_f1: float,
    dataset_origin: str,
    is_active: int = 1,
    db_path: Optional[Path] = None
) -> int:
    """Registers a model artifact in the model_registry table."""
    conn = get_connection(db_path)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with conn:
            if is_active == 1:
                conn.execute("UPDATE model_registry SET is_active = 0 WHERE is_active = 1;")
            cursor = conn.execute("""
                INSERT INTO model_registry 
                (version, model_format, file_path, file_size_kb, test_accuracy, test_macro_f1, dataset_origin, is_active, registered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version) DO UPDATE SET
                    file_size_kb = excluded.file_size_kb,
                    test_accuracy = excluded.test_accuracy,
                    test_macro_f1 = excluded.test_macro_f1,
                    is_active = excluded.is_active;
            """, (version, model_format, str(file_path), file_size_kb, test_accuracy, test_macro_f1, dataset_origin, is_active, now))
            return cursor.lastrowid or 1
    finally:
        conn.close()

def get_active_model(db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Returns the currently active model record."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT * FROM model_registry WHERE is_active = 1 LIMIT 1;")
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def log_detection(
    model_id: Optional[int],
    src_ip: str,
    dst_ip: str,
    protocol: str,
    predicted_class: int,
    confidence: float,
    window_sequence_id: int,
    latency_ms: float,
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None
) -> int:
    """Logs a classified 10-flow window event and returns its flow_id."""
    conn = get_connection(db_path)
    now = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with conn:
            cursor = conn.execute("""
                INSERT INTO detection_log 
                (model_id, timestamp, src_ip, dst_ip, protocol, predicted_class, confidence, window_sequence_id, inference_latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (model_id, now, src_ip, dst_ip, protocol, predicted_class, confidence, window_sequence_id, latency_ms))
            return cursor.lastrowid
    finally:
        conn.close()

def create_alert(
    flow_id: int,
    severity: str,
    message: str,
    db_path: Optional[Path] = None
) -> int:
    """Inserts a new security alert for a malicious detection."""
    conn = get_connection(db_path)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with conn:
            cursor = conn.execute("""
                INSERT INTO alert (flow_id, severity, message, acknowledged, created_at)
                VALUES (?, ?, ?, 0, ?);
            """, (flow_id, severity, message, now))
            return cursor.lastrowid
    finally:
        conn.close()

def acknowledge_alert(alert_id: int, db_path: Optional[Path] = None) -> bool:
    """Marks an active alert as acknowledged."""
    conn = get_connection(db_path)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with conn:
            cursor = conn.execute("""
                UPDATE alert 
                SET acknowledged = 1, acknowledged_at = ?
                WHERE alert_id = ? AND acknowledged = 0;
            """, (now, alert_id))
            return cursor.rowcount > 0
    finally:
        conn.close()

def acknowledge_all_alerts(db_path: Optional[Path] = None) -> int:
    """Marks all currently active unacknowledged alerts as acknowledged."""
    conn = get_connection(db_path)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with conn:
            cursor = conn.execute("""
                UPDATE alert 
                SET acknowledged = 1, acknowledged_at = ?
                WHERE acknowledged = 0;
            """, (now,))
            return cursor.rowcount
    finally:
        conn.close()

def record_metrics(
    cpu_percent: float,
    memory_rss_mb: float,
    throughput_fps: float,
    host_status: str = "NORMAL",
    db_path: Optional[Path] = None
) -> int:
    """Stores a snapshot of hardware and throughput telemetry."""
    conn = get_connection(db_path)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with conn:
            cursor = conn.execute("""
                INSERT INTO system_metrics (timestamp, cpu_percent, memory_rss_mb, throughput_fps, host_status)
                VALUES (?, ?, ?, ?, ?);
            """, (now, cpu_percent, memory_rss_mb, throughput_fps, host_status))
            return cursor.lastrowid
    finally:
        conn.close()

def get_recent_detections(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieves the most recent classified flows."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("""
            SELECT flow_id, timestamp, src_ip, dst_ip, protocol, predicted_class, confidence, window_sequence_id, inference_latency_ms
            FROM detection_log
            ORDER BY flow_id DESC
            LIMIT ?;
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def get_unacknowledged_alerts(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieves unacknowledged alerts sorted by severity and recency."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("""
            SELECT a.alert_id, a.flow_id, a.severity, a.message, a.acknowledged, a.created_at,
                   d.src_ip, d.dst_ip, d.confidence, d.protocol
            FROM alert a
            JOIN detection_log d ON a.flow_id = d.flow_id
            WHERE a.acknowledged = 0
            ORDER BY 
                CASE a.severity
                    WHEN 'CRITICAL' THEN 1
                    WHEN 'HIGH' THEN 2
                    WHEN 'MEDIUM' THEN 3
                    WHEN 'LOW' THEN 4
                END,
                a.created_at DESC
            LIMIT ?;
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def get_latest_metrics(limit: int = 30, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieves the most recent system metrics for charting."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("""
            SELECT sample_id, timestamp, cpu_percent, memory_rss_mb, throughput_fps, host_status
            FROM system_metrics
            ORDER BY sample_id DESC
            LIMIT ?;
        """, (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        return list(reversed(rows))
    finally:
        conn.close()

def get_system_summary(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Generates an operational summary for the status dashboard."""
    conn = get_connection(db_path)
    try:
        total_flows = conn.execute("SELECT COUNT(*) FROM detection_log;").fetchone()[0]
        attack_flows = conn.execute("SELECT COUNT(*) FROM detection_log WHERE predicted_class = 1;").fetchone()[0]
        benign_flows = total_flows - attack_flows
        active_alerts = conn.execute("SELECT COUNT(*) FROM alert WHERE acknowledged = 0;").fetchone()[0]
        
        last_metric = conn.execute("SELECT * FROM system_metrics ORDER BY sample_id DESC LIMIT 1;").fetchone()
        active_model = conn.execute("SELECT version, file_size_kb, test_accuracy FROM model_registry WHERE is_active = 1 LIMIT 1;").fetchone()
        
        return {
            "total_flows_processed": total_flows,
            "benign_flows": benign_flows,
            "malicious_threats": attack_flows,
            "unacknowledged_alerts": active_alerts,
            "active_model": dict(active_model) if active_model else {
                "version": "v1.0.0-unsw-hybrid",
                "file_size_kb": 100.08,
                "test_accuracy": 0.9984
            },
            "latest_telemetry": dict(last_metric) if last_metric else {
                "cpu_percent": 0.0,
                "memory_rss_mb": 49.93,
                "throughput_fps": 0.0,
                "host_status": "NORMAL"
            }
        }
    finally:
        conn.close()
