"""
src/benchmark_runtime.py - 1,000-Iteration Edge Runtime Latency & Memory Benchmark
Executes formal simulation acceptance procedure (Chapter 4.10.2/4.10.3) under taskset -c 0.
"""

import sys
import time
import psutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import (
    TFLITE_MODEL_PATH, LOGS_DIR, NFR_TARGETS, WINDOW_SIZE, PCA_COMPONENTS
)
from src.inference_engine import EdgeInferenceEngine

def run_benchmark(
    model_path: Optional[Path] = None,
    warmup: int = 20,
    iterations: int = 1000,
    output_csv: Optional[Path] = None
) -> dict:
    """Runs 1,000 warm iterations and computes latency percentiles and memory RSS."""
    m_path = Path(model_path or TFLITE_MODEL_PATH)
    out_csv = Path(output_csv or (LOGS_DIR / "runtime_benchmark.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print("==================================================================")
    print(" FORMAL CONSTRAINED-ENVIRONMENT RUNTIME BENCHMARK                ")
    print(f" Target Model        : {m_path.name}")
    print(f" Warmup Runs         : {warmup}")
    print(f" Timed Iterations    : {iterations}")
    print(f" Concurrency         : Single-Thread CPU (CUDA_VISIBLE_DEVICES=-1)")
    print("==================================================================")
    
    engine = EdgeInferenceEngine(m_path, num_threads=1)
    process = psutil.Process()
    
    # Generate calibrated synthetic sequence batch (1, 10, 22)
    np.random.seed(42)
    sample_sequence = np.random.randn(1, WINDOW_SIZE, PCA_COMPONENTS).astype(np.float32)
    
    # 1. Warmup runs
    print(f"Executing {warmup} warmup iterations...")
    for _ in range(warmup):
        engine.predict_window(sample_sequence)
        
    # 2. Timed benchmark loop
    print(f"Benchmarking {iterations} sequential inferences...")
    latencies_ms = []
    t_start_total = time.perf_counter()
    
    for i in range(iterations):
        t_start = time.perf_counter_ns()
        engine.predict_window(sample_sequence)
        t_end = time.perf_counter_ns()
        latencies_ms.append((t_end - t_start) / 1_000_000.0)
        
    t_end_total = time.perf_counter()
    total_time_sec = t_end_total - t_start_total
    
    # 3. Memory & Hardware Telemetry
    mem_info = process.memory_info()
    peak_rss_mb = mem_info.rss / (1024.0 * 1024.0)
    file_size_kb = m_path.stat().st_size / 1024.0
    
    # 4. Statistical Calculations
    latencies = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    std_lat = float(np.std(latencies))
    throughput_seq = iterations / max(total_time_sec, 0.001)
    throughput_flows = throughput_seq * WINDOW_SIZE
    
    # Check compliance with NFR contracts
    nfr1_pass = mean_lat <= NFR_TARGETS["NFR1_MAX_LATENCY_MS"]
    nfr2_pass = file_size_kb <= NFR_TARGETS["NFR2_MAX_MODEL_SIZE_KB"]
    nfr3_pass = peak_rss_mb <= NFR_TARGETS["NFR3_MAX_MEMORY_RSS_MB"]
    
    # Save CSV
    df_results = pd.DataFrame([{
        "model": m_path.name,
        "iterations": iterations,
        "mean_latency_ms": round(mean_lat, 6),
        "median_latency_ms": round(median_lat, 6),
        "p95_latency_ms": round(p95_lat, 6),
        "min_latency_ms": round(min_lat, 6),
        "max_latency_ms": round(max_lat, 6),
        "std_latency_ms": round(std_lat, 6),
        "throughput_seq_per_sec": round(throughput_seq, 2),
        "throughput_flows_per_sec": round(throughput_flows, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "model_size_kb": round(file_size_kb, 2),
        "nfr1_latency_pass": nfr1_pass,
        "nfr2_size_pass": nfr2_pass,
        "nfr3_memory_pass": nfr3_pass
    }])
    df_results.to_csv(out_csv, index=False)
    
    print("\n================ BENCHMARK RESULTS ================")
    print(f"Mean Latency (per 10-flow window): {mean_lat:.6f} ms (Target <= 50.0 ms: {'PASSED' if nfr1_pass else 'FAILED'})")
    print(f"Median Latency                   : {median_lat:.6f} ms")
    print(f"95th-Percentile Latency (p95)    : {p95_lat:.6f} ms")
    print(f"Min / Max Latency                : {min_lat:.6f} ms / {max_lat:.6f} ms")
    print(f"Throughput                       : {throughput_seq:.2f} windows/s ({throughput_flows:.2f} flows/s)")
    print(f"Peak Resident Memory (RSS)       : {peak_rss_mb:.2f} MB (Target <= 512.0 MB: {'PASSED' if nfr3_pass else 'FAILED'})")
    print(f"Model Storage Size               : {file_size_kb:.2f} KB (Target <= 1000.0 KB: {'PASSED' if nfr2_pass else 'FAILED'})")
    print(f"Results saved to                 : {out_csv}")
    print("===================================================\n")
    print("Academic Note: Reported measurements reflect workstation simulation under taskset -c 0.")
    print("Power draw is recorded as: NOT MEASURED.")
    
    return {
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "peak_rss_mb": peak_rss_mb,
        "file_size_kb": file_size_kb,
        "nfr1_pass": nfr1_pass,
        "nfr2_pass": nfr2_pass,
        "nfr3_pass": nfr3_pass
    }

def main():
    parser = argparse.ArgumentParser(description="Run 1,000-iteration simulated benchmark")
    parser.add_argument("--model", type=str, default=str(TFLITE_MODEL_PATH), help="Path to .tflite model")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=1000, help="Benchmark iterations")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()
    
    run_benchmark(
        model_path=Path(args.model),
        warmup=args.warmup,
        iterations=args.iterations,
        output_csv=Path(args.output) if args.output else None
    )

if __name__ == "__main__":
    main()
