"""
src/stream_simulator.py - Network Flow Replay & Streaming Simulation Engine
Simulates real-time network flow arrival by reading benchmark datasets in 10-flow sequences.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Dict, Any, Optional
from src.config import SAMPLE_FLOWS_PATH, WINDOW_SIZE, SIMULATION_CONFIG
from src.preprocessing import load_preprocessors, transform_flow_records

class FlowStreamSimulator:
    """Streams flow records from CSV files in sliding/tumbling windows of 10 records."""
    
    def __init__(
        self,
        csv_path: Optional[Path] = None,
        rate_flows_per_sec: int = SIMULATION_CONFIG["DEFAULT_REPLAY_RATE"],
        loop: bool = True
    ):
        self.csv_path = Path(csv_path or SAMPLE_FLOWS_PATH)
        self.rate_fps = max(rate_flows_per_sec, 1)
        self.loop = loop
        self.window_size = WINDOW_SIZE
        
        # Load fitted transformers
        self.scaler, self.pca, self.feature_cols = load_preprocessors()
        self._load_data()
        
    def _load_data(self) -> None:
        """Loads and pre-validates dataset."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Sample flows dataset not found at {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        if len(self.df) < self.window_size:
            raise ValueError(f"Dataset has {len(self.df)} rows, minimum required is {self.window_size}")
            
    def stream_windows(self) -> Generator[Tuple[int, np.ndarray, Dict[str, Any]], None, None]:
        """
        Yields sequential 10-flow windows with transformed feature tensors and stream metadata.
        
        Yields:
            Tuple: (window_id, sequence_tensor of shape (1, 10, PCA_COMPONENTS), metadata_dict)
        """
        window_id = 1
        delay_per_window = self.window_size / float(self.rate_fps)
        total_rows = len(self.df)
        idx = 0
        
        while True:
            # Check boundary
            if idx + self.window_size > total_rows:
                if self.loop:
                    idx = 0
                else:
                    break
                    
            df_window = self.df.iloc[idx:idx + self.window_size]
            idx += self.window_size
            
            # Transform features to PCA representation: shape (10, PCA_COMPONENTS)
            features_pca = transform_flow_records(df_window, self.scaler, self.pca, self.feature_cols)
            sequence_tensor = np.expand_dims(features_pca, axis=0)  # (1, 10, Features)
            
            # Extract representative metadata (prioritize attack if present in window)
            attack_rows = df_window[df_window["label"] == 1] if "label" in df_window.columns else pd.DataFrame()
            if not attack_rows.empty:
                rep_row = attack_rows.iloc[0]
            else:
                rep_row = df_window.iloc[-1]
                
            is_attack_ground_truth = int(rep_row.get("label", 0)) == 1

            # Robust IP resolution for datasets without explicit packet IP headers
            raw_src = rep_row.get("srcip")
            if pd.isna(raw_src) or not str(raw_src).strip() or str(raw_src).lower() in ("nan", "none", "-"):
                if is_attack_ground_truth:
                    src_ip = f"198.51.100.{(window_id * 7) % 250 + 1}"
                else:
                    src_ip = f"10.0.0.{(window_id * 3) % 250 + 1}"
            else:
                src_ip = str(raw_src).strip()

            raw_dst = rep_row.get("dstip")
            if pd.isna(raw_dst) or not str(raw_dst).strip() or str(raw_dst).lower() in ("nan", "none", "-"):
                dst_ip = "192.168.1.1" if is_attack_ground_truth else f"172.16.0.{(window_id % 12) + 1}"
            else:
                dst_ip = str(raw_dst).strip()

            metadata = {
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "protocol": str(rep_row.get("proto", "TCP")).upper(),
                "service": str(rep_row.get("service", "-")),
                "attack_cat": str(rep_row.get("attack_cat", "Normal")),
                "ground_truth_attack": int(is_attack_ground_truth)
            }
            
            yield window_id, sequence_tensor, metadata
            window_id += 1
            
            # Rate-limiting sleep to simulate arrival rate
            if delay_per_window > 0:
                time.sleep(delay_per_window)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Network Flow Replay Stream Simulator")
    parser.add_argument("--dataset", type=str, default=None, help="Path to benchmark CSV dataset")
    parser.add_argument("--rate", type=int, default=SIMULATION_CONFIG["DEFAULT_REPLAY_RATE"], help="Flows per second")
    parser.add_argument("--max-windows", type=int, default=10, help="Maximum windows to stream before stopping")
    args = parser.parse_args()

    simulator = FlowStreamSimulator(
        csv_path=Path(args.dataset) if args.dataset else None,
        rate_flows_per_sec=args.rate,
        loop=True
    )
    print(f"Streaming from: {simulator.csv_path} at {simulator.rate_fps} flows/s")
    for wid, tensor, meta in simulator.stream_windows():
        print(f"[Window #{wid:04d}] Shape: {tensor.shape} | {meta['src_ip']} -> {meta['dst_ip']} ({meta['protocol']}) | Attack: {meta['ground_truth_attack']}")
        if args.max_windows and wid >= args.max_windows:
            break

if __name__ == "__main__":
    main()
