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
                
            metadata = {
                "src_ip": str(rep_row.get("srcip", "192.168.1.100")),
                "dst_ip": str(rep_row.get("dstip", "192.168.1.1")),
                "protocol": str(rep_row.get("proto", "TCP")).upper(),
                "service": str(rep_row.get("service", "-")),
                "attack_cat": str(rep_row.get("attack_cat", "Normal")),
                "ground_truth_attack": int(rep_row.get("label", 0))
            }
            
            yield window_id, sequence_tensor, metadata
            window_id += 1
            
            # Rate-limiting sleep to simulate arrival rate
            if delay_per_window > 0:
                time.sleep(delay_per_window)
