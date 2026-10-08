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

from src.config import (
    SIMULATION_CONFIG, NFR_TARGETS, DB_PATH, TFLITE_MODEL_PATH, SAMPLE_FLOWS_PATH,
    IS_SERVERLESS, IS_CONTAINER, DISABLE_EMBEDDED_THREADING, SEED_ON_STARTUP,
    host_is_allowed,
)
from src.database import (
    init_db, get_system_summary, get_recent_detections,
    get_unacknowledged_alerts, get_alert_history,
    acknowledge_alert, acknowledge_all_alerts,
    get_latest_metrics, get_active_model, log_detection,
    clear_runtime_data, get_simulation_state, update_simulation_state,
    get_max_window_sequence_id
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

import logging

logger = logging.getLogger("grace.dashboard")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [GRACE] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

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
        self.last_step_time = 0.0
        self.last_error = None
        self._init_components()

    def _init_components(self):
        try:
            logger.info("Initializing components. DB_PATH=%s, IS_SERVERLESS=%s, IS_CONTAINER=%s",
                        DB_PATH, IS_SERVERLESS, IS_CONTAINER)
            init_db()
            active_rec = get_active_model()
            self.model_id = active_rec["model_id"] if active_rec else 1
            logger.info("Active model: %s (model_id=%d)",
                        active_rec["version"] if active_rec else "None (default=1)", self.model_id)

            if TFLITE_MODEL_PATH.exists():
                self.engine = EdgeInferenceEngine(TFLITE_MODEL_PATH, num_threads=1)
                logger.info("Loaded EdgeInferenceEngine from %s", TFLITE_MODEL_PATH)
            else:
                logger.error("TFLite model not found at: %s", TFLITE_MODEL_PATH)

            if SAMPLE_FLOWS_PATH.exists():
                max_seq = get_max_window_sequence_id()
                self.simulator = FlowStreamSimulator(
                    csv_path=SAMPLE_FLOWS_PATH,
                    rate_flows_per_sec=SIMULATION_CONFIG["DEFAULT_REPLAY_RATE"],
                    loop=True
                )
                if max_seq > 0:
                    self.simulator.window_counter = max_seq
                self.stream_generator = self.simulator.stream_windows(with_delay=False)
                logger.info("Loaded FlowStreamSimulator from %s (resuming at window #%d)",
                            SAMPLE_FLOWS_PATH, max_seq)
            else:
                logger.error("Sample flows CSV not found at: %s", SAMPLE_FLOWS_PATH)

            self.telemetry = TelemetrySampler()

            # Synchronize state from database
            state = get_simulation_state()
            self.running = state.get("is_running", True)
            self.windows_processed = state.get("windows_processed", 0)
            self.alerts_generated = state.get("alerts_generated", 0)
            self.last_step_time = state.get("last_step_time", 0.0)
            logger.info("Simulation state loaded: running=%s, processed=%d, alerts=%d",
                        self.running, self.windows_processed, self.alerts_generated)

            # Start background thread unless explicitly disabled.
            threading_enabled = not (IS_SERVERLESS or DISABLE_EMBEDDED_THREADING)
            if self.running and threading_enabled:
                self.thread = threading.Thread(target=self._run_loop, daemon=True)
                self.thread.start()
                logger.info("Started background simulation thread.")
            else:
                logger.info("Background thread disabled (IS_SERVERLESS=%s, DISABLE_EMBEDDED_THREADING=%s). Request-driven stepping active.",
                            IS_SERVERLESS, DISABLE_EMBEDDED_THREADING)
        except Exception as e:
            self.last_error = str(e)
            logger.exception("[SimManager] Initialization error: %s", e)

    def step(self, count: int = 1):
        """Advances stream replay by count windows and persists results immediately."""
        with self.lock:
            if not self.engine or not self.simulator or not self.stream_generator:
                logger.info("[SimManager] Components incomplete before step() - re-initializing...")
                self._init_components()
            if not self.engine or not self.simulator or not self.stream_generator:
                err_msg = f"Incomplete components: engine={bool(self.engine)}, sim={bool(self.simulator)}, gen={bool(self.stream_generator)}"
                self.last_error = err_msg
                logger.error("[SimManager] Step aborted: %s", err_msg)
                return []
            
            new_flow_ids = []
            for _ in range(count):
                try:
                    window_id, sequence_tensor, metadata = next(self.stream_generator)
                except StopIteration:
                    if self.simulator and self.simulator.loop:
                        self.stream_generator = self.simulator.stream_windows(with_delay=False)
                        window_id, sequence_tensor, metadata = next(self.stream_generator)
                    else:
                        logger.info("[SimManager] Stream completed without loop")
                        break
                except Exception as stream_err:
                    self.last_error = f"Generator next() error: {stream_err}"
                    logger.exception("[SimManager] Generator error: %s", stream_err)
                    break
                
                try:
                    pred_class, confidence, latency_ms = self.engine.predict_window(sequence_tensor)
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
                        self.alerts_generated += 1
                        
                    new_flow_ids.append(flow_id)
                    logger.info("[Step] Window #%d -> %s (conf=%.4f, lat=%.3fms) logged flow_id=%s",
                                window_id, "MALICIOUS" if pred_class == 1 else "BENIGN", confidence, latency_ms, flow_id)
                except Exception as step_err:
                    self.last_error = f"Inference/logging error: {step_err}"
                    logger.exception("[SimManager] Error predicting/logging window #%s: %s", window_id, step_err)
                    break

            now = time.time()
            self.last_step_time = now
            update_simulation_state(
                is_running=self.running,
                set_processed=self.windows_processed,
                set_alerts=self.alerts_generated,
                last_step_time=now
            )
            
            if self.telemetry:
                self.telemetry.sample(current_flow_count=self.windows_processed * 10)
                
            return new_flow_ids

    def start(self):
        with self.lock:
            self.running = True
            update_simulation_state(is_running=True)
            if not (IS_SERVERLESS or DISABLE_EMBEDDED_THREADING):
                if self.thread is None or not self.thread.is_alive():
                    self.thread = threading.Thread(target=self._run_loop, daemon=True)
                    self.thread.start()
            return {"status": "started", "running": True}

    def pause(self):
        with self.lock:
            self.running = False
            update_simulation_state(is_running=False)
            return {"status": "paused", "running": False}

    def reset(self, clear_data: bool = True):
        with self.lock:
            self.running = False
            if self.simulator:
                self.simulator.window_counter = 0
                self.stream_generator = self.simulator.stream_windows()
            self.windows_processed = 0
            self.alerts_generated = 0
            self.last_step_time = 0.0
            update_simulation_state(is_running=False, set_processed=0, set_alerts=0, last_step_time=0.0)
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
            state = get_simulation_state()
            self.running = state.get("is_running", self.running)
            if state.get("windows_processed", 0) > self.windows_processed:
                self.windows_processed = state["windows_processed"]
                self.alerts_generated = state.get("alerts_generated", self.alerts_generated)
            return {
                "running": self.running,
                "windows_processed": self.windows_processed,
                "alerts_generated": self.alerts_generated,
                "stream_rate_fps": self.simulator.rate_fps if self.simulator else SIMULATION_CONFIG.get("DEFAULT_REPLAY_RATE", 30)
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
                update_simulation_state(
                    is_running=self.running,
                    set_processed=self.windows_processed,
                    set_alerts=self.alerts_generated,
                    last_step_time=now
                )

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
                    update_simulation_state(is_running=False)
            except Exception as e:
                print(f"[SimManager] Loop warning: {e}")
                time.sleep(0.5)

sim_manager = BackgroundSimulationManager()

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static")
)

SERVICE_VERSION = "v2.0.0-keras-cnnlstm"

@app.before_request
def validate_request_host():
    """Rejects requests whose Host header is outside the trusted allow-list.
    Orchestrator health probes are always exempt from host validation."""
    if request.path.startswith("/health"):
        return None
    if not host_is_allowed(request.host):
        return jsonify({
            "status": "error",
            "message": "Invalid Host header. Add the hostname to GRACE_ALLOWED_HOSTS.",
        }), 400

@app.errorhandler(404)
def handle_404(e):
    return jsonify({
        "error": "Not Found",
        "path": request.path,
        "full_path": request.full_path,
        "environ_path_info": request.environ.get("PATH_INFO"),
        "environ_script_name": request.environ.get("SCRIPT_NAME"),
        "routes": [str(rule) for rule in app.url_map.iter_rules()]
    }), 404


@app.route("/health/")
@app.route("/health")
def health_check():
    """
    Lightweight health check probe for cloud orchestrators (Vercel, Render,
    Kubernetes, Docker). Verifies service liveness and SQLite query readiness.
    Returns HTTP 200 if the database is reachable, HTTP 503 if disconnected.
    """
    import sqlite3 as _sqlite3
    db_connected = False
    db_error = None
    try:
        conn = _sqlite3.connect(str(DB_PATH), timeout=5.0)
        try:
            db_connected = conn.execute("SELECT 1;").fetchone()[0] == 1
        finally:
            conn.close()
    except Exception as exc:
        db_error = str(exc)

    is_healthy = db_connected
    payload = {
        "status": "healthy" if is_healthy else "unhealthy",
        "database": "connected" if db_connected else f"disconnected ({db_error})",
        "service": "grace-threat-detect",
        "version": SERVICE_VERSION,
        "model_present": TFLITE_MODEL_PATH.exists(),
        "container": IS_CONTAINER,
        "serverless": IS_SERVERLESS,
        "uptime_seconds": int(time.time() - SERVER_START_TIME),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return jsonify(payload), (200 if is_healthy else 503)

# Container startup bootstrap: ensure schema/registry exist on a fresh volume
# (mirrors docker-entrypoint.sh for platforms that skip the image entrypoint).
if SEED_ON_STARTUP:
    try:
        init_db()
        if not get_active_model():
            print("[GRACE] No active model registered — running demo seeding...")
            import subprocess
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "seed_demo_data.py")],
                check=False,
            )
    except Exception as seed_exc:
        print(f"[GRACE Warning] Startup seeding notice: {seed_exc}")


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
        sim=SIMULATION_CONFIG,
        active_page="dashboard"
    )

@app.route("/benchmarks")
def benchmarks():
    """Renders the dedicated Academic Benchmarks and Dissertation Evaluation page."""
    summary = get_system_summary()
    summary["uptime"] = get_uptime_str()
    summary["uptime_seconds"] = int(time.time() - SERVER_START_TIME)
    summary["simulation"] = sim_manager.status()

    benchmark_data = None
    summary_path = REPORTS_DIR / "evaluation_summary.json"
    edge_lat_path = REPORTS_DIR / "edge_latency_benchmark.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                benchmark_data = json.load(f)
            if edge_lat_path.exists():
                try:
                    with open(edge_lat_path, "r") as f_edge:
                        benchmark_data["edge_latency"] = json.load(f_edge)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Benchmarks] Notice loading benchmarks: {e}")

    return render_template(
        "benchmarks.html",
        summary=summary,
        nfr=NFR_TARGETS,
        sim=SIMULATION_CONFIG,
        benchmarks=benchmark_data,
        active_page="benchmarks"
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
    state = get_simulation_state()
    if state.get("is_running", True):
        # In serverless or when background thread is not alive, step on demand
        thread_inactive = sim_manager.thread is None or not sim_manager.thread.is_alive()
        if IS_SERVERLESS or (DISABLE_EMBEDDED_THREADING and thread_inactive):
            try:
                new_ids = sim_manager.step(count=1)
                logger.info("[API] Stepped simulation on demand. New flow IDs: %s", new_ids)
            except Exception as e:
                logger.exception("[API] Simulation step error: %s", e)

    limit = min(int(request.args.get("limit", 50)), 100)
    detections = get_recent_detections(limit=limit)
    if not detections:
        logger.warning("[API] 0 detections returned from DB (%s). Last error: %s", DB_PATH, sim_manager.last_error)
    return jsonify({"detections": detections})

@app.route("/api/debug")
def api_debug():
    """Returns operational diagnostics and component status for troubleshooting."""
    import sqlite3 as _sqlite3
    db_counts = {}
    try:
        conn = _sqlite3.connect(str(DB_PATH), timeout=3.0)
        try:
            db_counts["detection_log"] = conn.execute("SELECT COUNT(*) FROM detection_log;").fetchone()[0]
            db_counts["alert"] = conn.execute("SELECT COUNT(*) FROM alert;").fetchone()[0]
            db_counts["model_registry"] = conn.execute("SELECT COUNT(*) FROM model_registry;").fetchone()[0]
            db_counts["system_metrics"] = conn.execute("SELECT COUNT(*) FROM system_metrics;").fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:
        db_counts["error"] = str(exc)

    return jsonify({
        "environment": {
            "is_serverless": IS_SERVERLESS,
            "is_container": IS_CONTAINER,
            "disable_threading": DISABLE_EMBEDDED_THREADING,
            "server_start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(SERVER_START_TIME)),
            "uptime": get_uptime_str(),
        },
        "storage": {
            "db_path": str(DB_PATH),
            "db_exists": DB_PATH.exists(),
            "db_size_bytes": DB_PATH.stat().st_size if DB_PATH.exists() else 0,
            "counts": db_counts,
        },
        "artifacts": {
            "tflite_model_path": str(TFLITE_MODEL_PATH),
            "tflite_model_exists": TFLITE_MODEL_PATH.exists(),
            "tflite_model_size_kb": round(TFLITE_MODEL_PATH.stat().st_size / 1024.0, 2) if TFLITE_MODEL_PATH.exists() else 0,
            "sample_flows_path": str(SAMPLE_FLOWS_PATH),
            "sample_flows_exists": SAMPLE_FLOWS_PATH.exists(),
        },
        "simulation_engine": {
            "running": sim_manager.running,
            "windows_processed": sim_manager.windows_processed,
            "alerts_generated": sim_manager.alerts_generated,
            "engine_loaded": sim_manager.engine is not None,
            "simulator_loaded": sim_manager.simulator is not None,
            "generator_loaded": sim_manager.stream_generator is not None,
            "thread_alive": sim_manager.thread.is_alive() if sim_manager.thread else False,
            "last_step_error": sim_manager.last_error,
        },
        "recent_detections_sample": get_recent_detections(limit=3),
    })

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

@app.route("/favicon.ico")
def api_favicon():
    """Serves brand favicon to eliminate browser 404 console errors."""
    return send_from_directory(str(Path(__file__).parent / "static"), "favicon.svg", mimetype="image/svg+xml")

def run_dashboard(host: str = None, port: int = None):
    """Starts the Flask development web server."""
    import os
    if host is None:
        host = os.environ.get("GRACE_DASHBOARD_HOST", SIMULATION_CONFIG["DASHBOARD_HOST"])
    if port is None:
        port = int(os.environ.get("PORT", os.environ.get("FLASK_RUN_PORT", SIMULATION_CONFIG["DASHBOARD_PORT"])))
    print("==================================================================")
    print(" AI NETWORK THREAT MONITORING DASHBOARD (FLASK DECOUPLED SERVICE)")
    print(f" URL             : http://{host}:{port}")
    print(f" SQLite Database : {DB_PATH}")
    print("==================================================================")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    run_dashboard()
