# Table 13: Hardware Feasibility and Latency Percentiles Under Single-Core Edge Simulation

| Evaluation Metric | Measured Value | NFR Target Constraint | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Mean Inference Latency** | **0.0244 ms** | &le; 50.0 ms (NFR1 target: &le; 20 ms) | **PASSED** (2,047x headroom) |
| **Median Latency (p50)** | **0.0204 ms** | Typ. edge responsiveness | **PASSED** |
| **90th-Percentile Latency (p90)** | **0.0381 ms** | &le; 50.0 ms | **PASSED** |
| **95th-Percentile Latency (p95)** | **0.0405 ms** | Edge worst-case bracket | **PASSED** |
| **99th-Percentile Latency (p99)** | **0.0510 ms** | Edge tail latency | **PASSED** |
| **Min / Max Latency** | 0.0192 ms / 0.0671 ms | Tail bounded jitter | **PASSED** |
| **Interquartile Jitter (IQR)** | 0.0027 ms | High temporal stability | **PASSED** |
| **Sequence Throughput** | **26482.7 windows/s** | Sustained streaming rate | **PASSED** |
| **Flow Classification Rate** | **264827.1 flows/s** | Real-world line rate | **PASSED** |
| **Peak Process Memory (RSS)** | **138.35 MB** | &le; 512.0 MB (NFR3 ceiling) | **PASSED** |
| **Model Storage Footprint** | **56.89 KB** | &le; 1,000.0 KB (NFR2, target &le; 150 KB) | **PASSED** (62% under target) |
| **Logical CPU Concurrency** | Single Thread (Pinned Core 0) | Zero GPU reliance (NFR5) | **PASSED** |

*Note: Formal measurements gathered on single CPU core affinity under `taskset -c 0` simulating Raspberry Pi 4 edge platform. Power draw: NOT MEASURED.*
