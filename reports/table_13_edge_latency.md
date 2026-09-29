# Table 13: Hardware Feasibility and Latency Percentiles Under Single-Core Edge Simulation

| Evaluation Metric | Measured Value | NFR Target Constraint | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Mean Inference Latency** | **0.0999 ms** | &le; 50.0 ms (NFR1 target: &le; 20 ms) | **PASSED** (500x headroom) |
| **Median Latency (p50)** | **0.0856 ms** | Typ. edge responsiveness | **PASSED** |
| **90th-Percentile Latency (p90)** | **0.1301 ms** | &le; 50.0 ms | **PASSED** |
| **95th-Percentile Latency (p95)** | **0.1489 ms** | Edge worst-case bracket | **PASSED** |
| **99th-Percentile Latency (p99)** | **0.2269 ms** | Edge tail latency | **PASSED** |
| **Min / Max Latency** | 0.0824 ms / 0.4113 ms | Tail bounded jitter | **PASSED** |
| **Interquartile Jitter (IQR)** | 0.0232 ms | High temporal stability | **PASSED** |
| **Sequence Throughput** | **8598.7 windows/s** | Sustained streaming rate | **PASSED** |
| **Flow Classification Rate** | **85987.0 flows/s** | Real-world line rate | **PASSED** |
| **Peak Process Memory (RSS)** | **142.32 MB** | &le; 512.0 MB (NFR3 ceiling) | **PASSED** |
| **Model Storage Footprint** | **99.59 KB** | &le; 1,000.0 KB (NFR2, target &le; 150 KB) | **PASSED** (34% under the 150 KB target) |
| **Logical CPU Concurrency** | Single Thread (Pinned Core 0) | Zero GPU reliance (NFR5) | **PASSED** |

*Note: Formal measurements gathered on single CPU core affinity under `taskset -c 0` simulating Raspberry Pi 4 edge platform. Power draw: NOT MEASURED.*
