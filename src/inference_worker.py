"""
src/inference_worker.py - Autonomous Edge Detection Worker
Orchestrates stream ingestion, single-core inference, alert triage, and SQLite logging.
"""

import sys
import time
import signal
import argparse
from pathlib import Path
from typing import Optional

from src.config import (
    TFLITE_MODEL_PATH, DB_PATH, SAMPLE_FLOWS_PATH,
    SIMULATION_CONFIG, NFR_TARGETS, CLASS_LABELS
)
from src.database import init_db, log_detection, get_active_model
from src.inference_engine import EdgeInferenceEngine
from src.stream_simulator import FlowStreamSimulator
from src.alert_manager import handle_malicious_detection
from src.telemetry import TelemetrySampler

class DetectionWorker:
    """Autonomous detection daemon operating within the simulated resource ceiling."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
        stream_rate: int = SIMULATION_CONFIG["DEFAULT_REPLAY_RATE"],
        max_windows: Optional[int] = None
    ):
        self.model_path = Path(model_path or TFLITE_MODEL_PATH)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.stream_rate = stream_rate
        self.max_windows = max_windows
        self.running = True
        
        # Ensure database is initialized
        init_db()
        active_rec = get_active_model()
        self.model_id = active_rec["model_id"] if active_rec else 1
        
        # Initialize subcomponents
        print(f"[Worker] Initializing Edge Inference Engine ({self.model_path})...")
        self.engine = EdgeInferenceEngine(self.model_path, num_threads=1)
        self.simulator = FlowStreamSimulator(
            csv_path=self.dataset_path,
            rate_flows_per_sec=self.stream_rate,
            loop=True
        )
        self.telemetry = TelemetrySampler()
        
        self.windows_processed = 0
        self.alerts_generated = 0
        
        # Register signals for graceful termination
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def _handle_shutdown(self, signum, frame):
        print("\n[Worker] Graceful shutdown signal received. Halting detection worker...")
        self.running = False
        
    def run(self):
        """Main detection loop processing sequential 10-flow windows."""
        print("==================================================================")
        print(" AI-POWERED NETWORK THREAT DETECTION SERVICE (SIMULATION ACTIVE)  ")
        print(f" Execution Mode : Single-Core Worker (Target Latency <= 50ms)")
        print(f" Flow Stream    : {self.stream_rate} flows/sec ({self.stream_rate // 10} windows/sec)")
        print(f" Persistence    : SQLite WAL Mode ({DB_PATH})")
        print("==================================================================")
        
        last_telemetry_time = time.time()
        
        for window_id, sequence_tensor, metadata in self.simulator.stream_windows():
            if not self.running:
                break
                
            # Perform inference on 10-flow window
            pred_class, confidence, latency_ms = self.engine.predict_window(sequence_tensor)
            self.windows_processed += 1
            
            # Persist detection event to SQLite
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
            
            # Trigger alert if classified as an attack
            alert_tag = ""
            if pred_class == 1:
                self.alerts_generated += 1
                alert_id, severity, _ = handle_malicious_detection(
                    flow_id=flow_id,
                    confidence=confidence,
                    src_ip=metadata["src_ip"],
                    dst_ip=metadata["dst_ip"],
                    protocol=metadata["protocol"],
                    window_id=window_id
                )
                alert_tag = f" | [!] ALERT-{alert_id} ({severity})"
                
            # Telemetry sample every second
            current_time = time.time()
            if current_time - last_telemetry_time >= SIMULATION_CONFIG["TELEMETRY_INTERVAL_SEC"]:
                metrics = self.telemetry.sample(current_flow_count=self.windows_processed * 10)
                last_telemetry_time = current_time
                telemetry_str = f" | CPU: {metrics['cpu_percent']}% | RSS: {metrics['memory_rss_mb']}MB"
            else:
                telemetry_str = ""
                
            # Console telemetry output
            class_str = "MALICIOUS" if pred_class == 1 else "BENIGN"
            print(
                f"[Window #{window_id:05d}] {metadata['src_ip']} -> {metadata['dst_ip']} "
                f"({metadata['protocol']}) -> {class_str} ({confidence:.1%}) "
                f"| Latency: {latency_ms:.3f}ms{alert_tag}{telemetry_str}"
            )
            
            if self.max_windows and self.windows_processed >= self.max_windows:
                print(f"[Worker] Completed requested {self.max_windows} windows.")
                break
                
        print("\n=== Detection Service Summary ===")
        print(f"Total Windows Processed: {self.windows_processed} ({self.windows_processed * 10} flows)")
        print(f"Security Alerts Raised : {self.alerts_generated}")
        print("Service terminated safely.")

def main():
    parser = argparse.ArgumentParser(description="AI Network Threat Detection Worker")
    parser.add_argument("--model", type=str, default=str(TFLITE_MODEL_PATH), help="Path to .tflite model")
    parser.add_argument("--dataset", type=str, default=None, help="Path to benchmark CSV dataset")
    parser.add_argument("--rate", type=int, default=SIMULATION_CONFIG["DEFAULT_REPLAY_RATE"], help="Arrival rate in flows/sec")
    parser.add_argument("--max-windows", type=int, default=None, help="Stop after N windows")
    args = parser.parse_args()
    
    worker = DetectionWorker(
        model_path=Path(args.model),
        dataset_path=Path(args.dataset) if args.dataset else None,
        stream_rate=args.rate,
        max_windows=args.max_windows
    )
    worker.run()

if __name__ == "__main__":
    main()
