"""
src/dashboard/app.py - Decoupled Flask Web Dashboard & Real-Time REST API
Serves interactive SecOps monitoring interface and telemetry polling endpoints.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
import sys
import json

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SIMULATION_CONFIG, NFR_TARGETS, DB_PATH
from src.database import (
    get_system_summary, get_recent_detections,
    get_unacknowledged_alerts, acknowledge_alert, acknowledge_all_alerts,
    get_latest_metrics
)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static")
)

@app.route("/")
def index():
    """Renders the main SecOps threat detection dashboard."""
    summary = get_system_summary()
    return render_template(
        "index.html",
        summary=summary,
        nfr=NFR_TARGETS,
        sim=SIMULATION_CONFIG
    )

@app.route("/api/status")
def api_status():
    """Returns aggregated system status and counter metrics."""
    return jsonify(get_system_summary())

@app.route("/api/detections/recent")
def api_recent_detections():
    """Returns the latest 50 classified 10-flow windows."""
    limit = min(int(request.args.get("limit", 50)), 100)
    return jsonify({"detections": get_recent_detections(limit=limit)})

@app.route("/api/alerts/unacknowledged")
def api_unacknowledged_alerts():
    """Returns all active, unacknowledged security threat alerts."""
    limit = min(int(request.args.get("limit", 50)), 100)
    return jsonify({"alerts": get_unacknowledged_alerts(limit=limit)})

@app.route("/api/alerts/<int:alert_id>/acknowledge", methods=["POST"])
def api_acknowledge_alert(alert_id: int):
    """Marks an active security alert as acknowledged by the operator."""
    success = acknowledge_alert(alert_id)
    if success:
        return jsonify({"status": "success", "alert_id": alert_id, "message": "Alert acknowledged."})
    return jsonify({"status": "error", "message": "Alert not found or already acknowledged."}), 404

@app.route("/api/alerts/acknowledge_all", methods=["POST"])
def api_acknowledge_all_alerts():
    """Marks all unacknowledged security alerts as acknowledged in bulk."""
    count = acknowledge_all_alerts()
    return jsonify({"status": "success", "acknowledged_count": count, "message": f"{count} alerts acknowledged."})

@app.route("/api/metrics/system")
def api_system_metrics():
    """Returns rolling CPU %, Memory RSS MB, and throughput samples."""
    limit = min(int(request.args.get("limit", 30)), 60)
    return jsonify({"metrics": get_latest_metrics(limit=limit)})

@app.route("/api/benchmarks")
def api_benchmarks():
    """Returns the dissertation academic evaluation benchmarks and model comparison matrix."""
    summary_path = REPORTS_DIR / "evaluation_summary.json"
    edge_lat_path = REPORTS_DIR / "edge_latency_benchmark.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
            if edge_lat_path.exists():
                try:
                    with open(edge_lat_path, "r") as f_edge:
                        data["edge_latency"] = json.load(f_edge)
                except Exception:
                    pass
            return jsonify({"status": "success", "data": data})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({
        "status": "pending",
        "message": "Benchmarks have not been generated yet. Run python3 -m src.evaluate."
    }), 404

@app.route("/api/figures/<path:filename>")
def api_figures(filename: str):
    """Serves generated confusion matrix figures and evaluation diagrams."""
    return send_from_directory(str(FIGURES_DIR), filename)

def run_dashboard(host: str = SIMULATION_CONFIG["DASHBOARD_HOST"], port: int = SIMULATION_CONFIG["DASHBOARD_PORT"]):
    """Starts the Flask development web server."""
    print("==================================================================")
    print(" AI NETWORK THREAT MONITORING DASHBOARD (FLASK DECOUPLED SERVICE)")
    print(f" URL             : http://{host}:{port}")
    print(f" SQLite Database : {DB_PATH}")
    print("==================================================================")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    run_dashboard()
