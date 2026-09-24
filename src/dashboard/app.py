"""
src/dashboard/app.py - Decoupled Flask Web Dashboard & Real-Time REST API
Serves interactive SecOps monitoring interface, telemetry polling endpoints,
and embedded background simulation controller.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
import sys
import json
import time
import threading

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SIMULATION_CONFIG, NFR_TARGETS, DB_PATH, TFLITE_MODEL_PATH, SAMPLE_FLOWS_PATH
from src.database import (
    init_db, get_system_summary, get_recent_detections,
    get_unacknowledged_alerts, get_alert_history,
    acknowledge_alert, acknowledge_all_alerts,
    get_latest_metrics, get_active_model, log_detection,
    clear_runtime_data
)
from src.inference_engine import EdgeInferenceEngine
from src.stream_simulator import FlowStreamSimulator
from src.alert_manager import handle_malicious_detection
from src.telemetry import TelemetrySampler

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SERVER_START_TIME = time.time()

def get_uptime_str() -> str:
    """Returns elapsed service uptime formatted as hh:mm:ss."""
    elapsed = int(time.time() - SERVER_START_TIME)
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class BackgroundSimulationManager:
    """Thread-safe simulation controller embedded within Flask backend."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.stream_generator = None
        self.simulator = None
        self.engine = None
        self.telemetry = None
        self.model_id = 1
        self.windows_processed = 0
        self.alerts_generated = 0
        self.thread = None
        self._init_components()

    def _init_components(self):
        try:
            init_db()
            active_rec = get_active_model()
            self.model_id = active_rec["model_id"] if active_rec else 1
            if TFLITE_MODEL_PATH.exists():
                self.engine = EdgeInferenceEngine(TFLITE_MODEL_PATH, num_threads=1)
            if SAMPLE_FLOWS_PATH.exists():
                self.simulator = FlowStreamSimulator(
                    csv_path=SAMPLE_FLOWS_PATH,
                    rate_flows_per_sec=SIMULATION_CONFIG["DEFAULT_REPLAY_RATE"],
                    loop=True
                )
                self.stream_generator = self.simulator.stream_windows()
            self.telemetry = TelemetrySampler()
        except Exception as e:
            print(f"[SimManager] Initialization notice: {e}")

    def start(self):
        with self.lock:
            if self.running:
                return {"status": "already_running", "running": True}
            self.running = True
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._run_loop, daemon=True)
                self.thread.start()
            return {"status": "started", "running": True}

    def pause(self):
        with self.lock:
            self.running = False
            return {"status": "paused", "running": False}

    def reset(self, clear_data: bool = True):
        with self.lock:
            self.running = False
            if self.simulator:
                self.stream_generator = self.simulator.stream_windows()
            self.windows_processed = 0
            self.alerts_generated = 0
            if clear_data:
                try:
                    clear_runtime_data(DB_PATH)
                except Exception as e:
                    print(f"[SimManager] Notice on clearing runtime data: {e}")
            return {
                "status": "reset",
                "running": False,
                "windows_processed": 0,
                "alerts_generated": 0,
                "message": "Simulation replay pointer and runtime logs reset successfully."
            }

    def status(self):
        with self.lock:
            return {
                "running": self.running,
                "windows_processed": self.windows_processed,
                "alerts_generated": self.alerts_generated,
                "stream_rate_fps": self.simulator.rate_fps if self.simulator else SIMULATION_CONFIG.get("DEFAULT_REPLAY_RATE", 100)
            }

    def _run_loop(self):
        last_telemetry_time = time.time()
        while True:
            if not self.running:
                time.sleep(0.2)
                continue

            if not self.engine or not self.simulator or not self.stream_generator:
                time.sleep(0.5)
                continue

            try:
                window_id, sequence_tensor, metadata = next(self.stream_generator)
                pred_class, confidence, latency_ms = self.engine.predict_window(sequence_tensor)
                
                with self.lock:
                    self.windows_processed += 1

                flow_id = log_detection(
                    model_id=self.model_id,
                    src_ip=metadata["src_ip"],
                    dst_ip=metadata["dst_ip"],
                    protocol=metadata["protocol"],
                    predicted_class=pred_class,
                    confidence=confidence,
                    window_sequence_id=window_id,
                    latency_ms=latency_ms
                )

                if pred_class == 1:
                    handle_malicious_detection(
                        flow_id=flow_id,
                        confidence=confidence,
                        src_ip=metadata["src_ip"],
                        dst_ip=metadata["dst_ip"],
                        protocol=metadata["protocol"],
                        window_id=window_id
                    )
                    with self.lock:
                        self.alerts_generated += 1

                now = time.time()
                if now - last_telemetry_time >= SIMULATION_CONFIG.get("TELEMETRY_INTERVAL_SEC", 1.0):
                    if self.telemetry:
                        self.telemetry.sample(current_flow_count=self.windows_processed * 10)
                    last_telemetry_time = now

                delay = float(self.simulator.window_size) / float(self.simulator.rate_fps)
                time.sleep(delay)
            except StopIteration:
                if self.simulator and self.simulator.loop:
                    self.stream_generator = self.simulator.stream_windows()
                else:
                    self.running = False
            except Exception as e:
                print(f"[SimManager] Loop warning: {e}")
                time.sleep(0.5)

sim_manager = BackgroundSimulationManager()

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static")
)

@app.route("/")
def index():
    """Renders the main SecOps threat detection dashboard."""
    summary = get_system_summary()
    summary["uptime"] = get_uptime_str()
    summary["uptime_seconds"] = int(time.time() - SERVER_START_TIME)
    summary["simulation"] = sim_manager.status()
    return render_template(
        "index.html",
        summary=summary,
        nfr=NFR_TARGETS,
        sim=SIMULATION_CONFIG
    )

@app.route("/api/status")
def api_status():
    """Returns aggregated system status, uptime, and simulation metrics."""
    summary = get_system_summary()
    summary["uptime"] = get_uptime_str()
    summary["uptime_seconds"] = int(time.time() - SERVER_START_TIME)
    summary["simulation"] = sim_manager.status()
    return jsonify(summary)

@app.route("/api/simulation/status")
def api_simulation_status():
    """Returns live playback state of the stream simulator."""
    return jsonify(sim_manager.status())

@app.route("/api/simulation/start", methods=["POST"])
def api_simulation_start():
    """Starts or resumes background network flow streaming."""
    return jsonify(sim_manager.start())

@app.route("/api/simulation/pause", methods=["POST"])
def api_simulation_pause():
    """Pauses background network flow streaming."""
    return jsonify(sim_manager.pause())

@app.route("/api/simulation/reset", methods=["POST"])
def api_simulation_reset():
    """Resets network flow playback position, counters, and clears operational logs."""
    clear_db = request.args.get("clear_db", "true").lower() in ("true", "1", "yes")
    if request.is_json and request.json:
        clear_db = request.json.get("clear_db", clear_db)
    return jsonify(sim_manager.reset(clear_data=clear_db))

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

@app.route("/api/alerts/history")
def api_alert_history():
    """Returns acknowledged security threat alerts for audit trail."""
    limit = min(int(request.args.get("limit", 50)), 100)
    return jsonify({"alerts": get_alert_history(limit=limit)})

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
