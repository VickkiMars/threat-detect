# Architecture Decision Record (ADR-001)
## Workstation Simulation of Raspberry Pi-Class Edge Execution Environment

**Status:** Accepted  
**Date:** September 2026  
**Author:** Imeh Grace Mfon (21/SC/CO/1117)  
**Academic Context:** Department of Computer Science, University of Uyo, Nigeria  
**Cross-Reference:** Chapter 3.11, Chapter 4.10, Table 4.7 of B.Sc. Project Report  

---

## 1. Context and Problem Statement

The intended deployment target for this AI-powered network threat detection system is an affordable, edge-class single-board computer—specifically the **Raspberry Pi 4 Model B** (Quad-Core ARM Cortex-A72 @ 1.5GHz, 2GB–4GB RAM). In educational and resource-constrained environments, physical hardware units are frequently unavailable, in short supply, or delayed in acquisition. 

To ensure complete, verifiable, and scientifically rigorous software implementation without waiting for physical hardware, the project requires an authoritative method to evaluate the model and software stack under equivalent computational constraints on a standard development workstation (running Ubuntu 22.04 LTS on an Intel Core i7-8665U x86_64 processor).

The challenge is to establish a **controlled, reproducible simulation harness** that accurately enforces single-core execution, memory budgets, and single-threaded edge runtimes, while transparently documenting the methodological boundary between workstation simulation and physical ARM deployment.

---

## 2. Decision

We accept and formalize a **five-pillar OS-level simulation methodology** executing directly on the Ubuntu Linux host:

### Pillar 1: Single-Core CPU Affinity Pinning
The inference and detection processes are strictly bound to a single logical CPU core using Linux CPU affinity scheduling:
```bash
taskset --cpu-list 0 <command>
```
This prevents the operating system scheduler from distributing matrix multiplication and tensor convolutions across multiple physical or virtual CPU cores.

### Pillar 2: Linux Cgroup / Address Space Memory Ceiling
To simulate the physical memory limitations of an entry-level Raspberry Pi and protect the host from runaway memory usage, execution is confined to a **2 GB memory envelope** using Linux `cgroups v2` via `systemd-run`:
```bash
systemd-run --user --scope \
  -p MemoryMax=2G \
  -p OOMPolicy=stop \
  <command>
```
If `systemd-run` is unavailable (e.g. within nested container or restricted shell environments), the POSIX shell virtual memory limit is applied as a verified fallback:
```bash
ulimit -v 2097152
```

### Pillar 3: Hardware Acceleration Disablement & Single-Thread Enforcement
Edge microprocessors like the Broadcom BCM2711 lack discrete CUDA GPUs. All GPU access is unconditionally severed:
```bash
export CUDA_VISIBLE_DEVICES=-1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
```
This forces all convolution, pooling, recurrent LSTM, and fully connected tensor math through a single-threaded CPU execution path.

### Pillar 4: Lightweight Standalone Runtime (`ai-edge-litert`)
Standard `import tensorflow` introduces massive framework bloat, allocating 687.41 MB peak RSS merely to initialize the TensorFlow runtime—violating the project's 512 MB sustained service memory ceiling (NFR3). 
Instead, the edge inference service uses the standalone, lightweight **`ai-edge-litert`** (or `tflite-runtime`) package available for Python 3.10–3.12:
- Bypasses full TensorFlow compilation graph initialization.
- Slashes peak process Resident Set Size (RSS) down to **49.93 MB**.
- Provides identical numerical inference outputs to standard TensorFlow Lite.

### Pillar 5: Decoupled Process Architecture
The web dashboard (Flask) and edge inference engine run as separate OS processes:
- **Process A (Inference Worker):** Runs under the single-core simulation ceiling (`taskset -c 0`) and writes detection logs, alerts, and system metrics directly to `data/threat_detection.db`.
- **Process B (Web Dashboard):** Runs as an independent process serving web requests and charting metrics from SQLite.
- *Rationale:* Web client traffic (page reloads, static asset requests) does not compete with real-time packet flow classification or skew inference latency measurements.

---

## 3. Explicit Differences & Simulation Boundaries

To maintain strict academic integrity (as emphasized in University of Uyo dissertation Chapter 4.10 and Chapter 5.5), the exact methodological boundaries of this simulation must be stated in all project publications and logs:

| Evaluation Dimension | Physical Raspberry Pi 4B | Workstation Simulation (This System) | Methodological Status |
| :--- | :--- | :--- | :--- |
| **Processor Architecture** | ARMv8-A (64-bit ARM Cortex-A72) | x86_64 (Intel Core i7-8665U) | **SIMULATED:** Faster IPC and wider vector registers on host. |
| **Core Allocation** | 1 Physical ARM Core | 1 Logical x86 Core (`taskset -c 0`) | **VALIDATED:** Multi-core parallelism eliminated. |
| **System Memory Limit** | 2 GB / 4 GB Physical RAM | 2 GB Cgroup Ceiling (`MemoryMax=2G`) | **VALIDATED:** Memory bound strictly enforced. |
| **Service Memory Ceiling** | < 512 MB RSS | < 512 MB RSS (Measured: 49.93 MB) | **PASSED:** Passes NFR3 criteria. |
| **Inference Latency** | Anticipated: 5–25 ms / sequence | Measured: 0.0095 ms / sequence | **SIMULATED:** Workstation latency reflects software efficiency. |
| **Thermal Throttling** | Present at 80°C under load | Minimal on workstation | **UNVERIFIED:** Physical cooling not simulated. |
| **Power Consumption** | Measurable via physical shunt (Watts)| **NOT MEASURED** | **EXPLICIT LIMITATION:** Recorded as NOT MEASURED. |

> [!IMPORTANT]
> **Academic Integrity Statement:**  
> All benchmark tables, dashboard panels, and evaluation scripts must label findings as **"Workstation Simulation Performance"** and never represent them as physical Raspberry Pi hardware measurements.

---

## 4. Execution Harness Reference Commands

### 4.1 Automated Benchmark Execution (Chapter 4.10.2 Command)
```bash
#!/usr/bin/env bash
# run_simulated_benchmark.sh

taskset --cpu-list 0 systemd-run --user --scope \
  -p MemoryMax=2G \
  -p OOMPolicy=stop \
  env CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  python3 -m src.benchmark_runtime \
  --model models/unsw_nb15/hybrid/model.tflite \
  --data-dir data/processed/unsw_nb15 \
  --output logs/unsw_nb15_runtime_benchmark.csv \
  --warmup 20 \
  --iterations 1000
```

### 4.2 Streaming Ingestion & Detection Service Execution
```bash
#!/usr/bin/env bash
# run_simulated_detector.sh

taskset --cpu-list 0 systemd-run --user --scope \
  -p MemoryMax=2G \
  -p OOMPolicy=stop \
  env CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  python3 -m src.inference_worker \
  --model models/unsw_nb15/hybrid/model.tflite \
  --db data/threat_detection.db \
  --rate 50
```

---

## 5. Consequences and Trade-offs

### Positive Consequences
- **Zero Financial Cost:** Enables immediate, comprehensive verification of the entire software and deep learning stack without procuring hardware.
- **Strict Software Bounding:** Proves that the model memory footprint (100.08 KB) and runtime heap (49.93 MB) easily fit within a 512 MB embedded budget.
- **Repeatability:** Eliminates hardware thermal variance and SD-card I/O bottlenecks during software CI/CD validation.
- **Seamless Future Portability:** Because TensorFlow Lite and Python SQLite are architecture-neutral, the exact same `.tflite` model and codebase will run without code modification when transferred to an actual Raspberry Pi.

### Negative Consequences / Accepted Limitations
- Latency measurements on the Intel host are faster than ARM Cortex-A72 execution.
- Power draw in watts cannot be measured and must remain documented as future physical hardware work.
