# AI-Powered Network Threat Detection System in Resource-Constrained Environments

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Target Platform: Raspberry Pi Simulation](https://img.shields.io/badge/platform-Raspberry%20Pi%20(Ubuntu%20Simulated)-green.svg)]()
[![Model: Quantized Hybrid CNN-LSTM](https://img.shields.io/badge/model-CNN--LSTM%20TFLite%20(100KB)-orange.svg)]()
[![Database: SQLite WAL](https://img.shields.io/badge/storage-SQLite%203%20(WAL)-lightgrey.svg)]()
[![License: Academic Research](https://img.shields.io/badge/license-Academic%20B.Sc.-purple.svg)]()

**Author:** Imeh Grace Mfon (Registration Number: 21/SC/CO/1117)  
**Supervisor:** Mr. U. J. Ntia  
**Department:** Department of Computer Science, Faculty of Computing  
**Institution:** University of Uyo, Uyo, Akwa Ibom State, Nigeria  
**Degree:** Bachelor of Science (B.Sc.) in Computer Science (August 2026)  

---

## 1. Project Overview

This repository houses the formal engineering specification, design architecture, and implementation blueprint for an **AI-Powered Network Threat Detection System** tailored for edge deployment in resource-constrained environments (such as network perimeter gateways and IoT aggregators).

The system addresses the dual challenges of high intrusion detection accuracy and strict computational efficiency by combining:
1. **PCA Feature Reduction:** Compresses high-dimensional network flow data (78 features) into 22 components while preserving $\ge 95.72\%$ variance.
2. **SMOTE + Tomek Links Balancing:** Solves acute benchmark class imbalance strictly on training splits.
3. **Hybrid CNN–LSTM Architecture:** Integrates 1D Convolutional layers (to extract intra-flow spatial relationships) and Long Short-Term Memory recurrent layers (to model inter-flow temporal progression across 10-flow sequences).
4. **Edge Quantization (TensorFlow Lite):** Converts 32-bit floating point weights into 8-bit dynamic-range integers, shrinking model storage to **100.08 KB**.
5. **Decoupled Embedded Stack:** Employs an embedded SQLite database in WAL mode and an ultra-lightweight Flask web dashboard, keeping steady-state memory utilization at **49.93 MB RSS** (well below the 512 MB Raspberry Pi ceiling).
6. **Controlled Workstation Simulation:** Evaluates and benchmarks edge performance on a host Ubuntu Linux workstation through single-core CPU pinning (`taskset -c 0`), memory cgroups (`systemd-run -p MemoryMax=2G`), and single-threaded execution (`CUDA_VISIBLE_DEVICES=-1`).

---

## 2. Master Engineering Documentation Suite

All agile engineering documents, formal specifications, and architecture decision records are compiled in the [`docs/`](docs/) directory:

| Document | File Link | Description |
| :--- | :--- | :--- |
| **Software Requirements Specification** | [`docs/SRS.md`](docs/SRS.md) | Formal, comprehensive blueprint aligned with IEEE 830 / ISO 29148 detailing the 6 system layers, FR1–FR10, and NFR1–NFR7. |
| **Requirements Analysis** | [`docs/REQUIREMENTS_ANALYSIS.md`](docs/REQUIREMENTS_ANALYSIS.md) | In-depth engineering analysis of dataset properties, leakage-controlled preprocessing, PCA reduction, and simulation boundaries. |
| **Product Backlog** | [`docs/PRODUCT_BACKLOG.md`](docs/PRODUCT_BACKLOG.md) | Prioritized, epic-based agile product backlog with T-shirt sizing, dependencies, and delivery increments. |
| **User Stories** | [`docs/USER_STORIES.md`](docs/USER_STORIES.md) | Comprehensive user stories organized across SecOps, Edge Admin, Academic Researcher, and ML Engineer personas. |
| **Acceptance Criteria & DoD** | [`docs/ACCEPTANCE_CRITERIA.md`](docs/ACCEPTANCE_CRITERIA.md) | Gherkin (Given-When-Then) test scenarios for all stories and the system-wide Definition of Done checklist. |
| **Simulation Architecture Record** | [`docs/SIMULATION_SPEC_ADR.md`](docs/SIMULATION_SPEC_ADR.md) | **ADR-001:** Workstation simulation specification formalizing OS-level CPU pinning, cgroup memory limits, and the ARM vs x86 boundary. |
| **Sprint 1 Backlog** | [`docs/SPRINT_BACKLOG_S1.md`](docs/SPRINT_BACKLOG_S1.md) | Committed Sprint 1 backlog establishing the immediate end-to-end simulated baseline. |
| **Database Schema & ERD** | [`docs/DATABASE_SCHEMA.md`](docs/DATABASE_SCHEMA.md) | Full SQLite relational schema, ER diagram, data dictionary, DDL script, and concurrency optimizations. |
| **Risks & Assumptions Log** | [`docs/RISKS_AND_ASSUMPTIONS.md`](docs/RISKS_AND_ASSUMPTIONS.md) | Formal risk register, explicit assumptions, mitigation strategies, and academic defense talking points. |

---

## 3. Six-Layer System Architecture

```
+-----------------------------------------------------------------------------------+
| 6. Constrained Deployment & Presentation Layer                                    |
|    - Flask Web Dashboard (Status, Live Flow Feed, Alert Queue, Telemetry Gauges)  |
|    - Embedded SQLite Database (detection_log, alert, system_metrics, registry)    |
+-----------------------------------------------------------------------------------+
                                         ^
                                         | Telemetry & Alert Events
+-----------------------------------------------------------------------------------+
| 5. Detection & Response Layer                                                     |
|    - Window Classification & Confidence Scoring (Argmax / Thresholding)           |
|    - Security Alert Triage (LOW, MEDIUM, HIGH, CRITICAL)                          |
+-----------------------------------------------------------------------------------+
                                         ^
                                         | Inferences
+-----------------------------------------------------------------------------------+
| 4. Model Optimisation & Edge Preparation Layer                                    |
|    - Post-Training 8-Bit Dynamic Range Quantization                               |
|    - TensorFlow Lite FlatBuffer Model Export (100.08 KB)                          |
|    - Standalone ai-edge-litert Single-Threaded Interpreter Runtime                |
+-----------------------------------------------------------------------------------+
                                         ^
                                         | Optimized Models
+-----------------------------------------------------------------------------------+
| 3. Hybrid Detection Engine Layer                                                  |
|    - Conv1D (Spatial Feature Extraction) + LSTM (Temporal Progression Modeling)   |
|    - Comparative Baselines: CNN-only, LSTM-only, Decision Tree, Random Forest, SVM|
+-----------------------------------------------------------------------------------+
                                         ^
                                         | 10-Flow Window Tensors (N, 10, Features)
+-----------------------------------------------------------------------------------+
| 2. Preprocessing & Feature Engineering Layer                                      |
|    - Leakage-Controlled Scaling & Categorical Encoding (Train split only)         |
|    - PCA Dimensionality Reduction (>= 95% Variance Retention)                     |
|    - SMOTE + Tomek Links Resampling (Train split only)                            |
|    - Fixed 10-Flow Temporal Windowing                                             |
+-----------------------------------------------------------------------------------+
                                         ^
                                         | Flow Records
+-----------------------------------------------------------------------------------+
| 1. Data Acquisition Layer                                                         |
|    - Benchmark Dataset Parsers: UNSW-NB15 & CICIDS2017 (200k Capped Subset)       |
|    - Configurable Flow Stream Playback & Replay Generator                         |
+-----------------------------------------------------------------------------------+
```

---

## 4. Key Performance Ceilings & Verified Results

| Metric | Target Specification (NFR) | Workstation Simulation Result | Status |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | $\le 50.0\text{ ms}$ per 10-flow window | **0.0095 ms** (p95: 0.0097 ms) | **PASSED** |
| **Model Binary Size** | $\le 1.0\text{ MB}$ | **100.08 KB** (Quantized TFLite) | **PASSED** |
| **Sustained Memory** | $\le 512.0\text{ MB RSS}$ | **49.93 MB RSS** (via `ai-edge-litert`)| **PASSED** |
| **Test Accuracy** | $\ge 95.0\%$ | **99.84%** (UNSW-NB15 Hybrid) | **PASSED** |
| **Attack Recall** | $\ge 95.0\%$ | **99.88%** (UNSW-NB15 Hybrid) | **PASSED** |
| **False Positive Rate**| $\le 1.0\%$ | **0.23%** (UNSW-NB15 Hybrid) | **PASSED** |
| **Macro-Averaged F1** | $\ge 95.0\%$ | **99.82%** (UNSW-NB15 Hybrid) | **PASSED** |
| **Execution Hardware**| 100% CPU-Only | Pinned single-core (`taskset -c 0`)| **PASSED** |
| **Power Consumption** | Not constrained | **NOT MEASURED** | **DOCUMENTED BOUNDARY** |

---

## 5. Quick Start: Running Under Simulation

### 5.1 Environment Setup
```bash
# Clone or navigate to project directory
cd /home/kami/Desktop/codebase/Grace

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install runtime dependencies
pip install --upgrade pip
pip install numpy pandas scikit-learn imbalanced-learn flask psutil
```

### 5.2 Launching the Simulated Benchmark (Chapter 4.10.2 Command)
```bash
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
