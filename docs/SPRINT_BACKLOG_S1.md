# Sprint 1 Backlog — AI-Powered Network Threat Detection System

**Sprint Identifier:** SPRINT-01-BASELINE  
**Sprint Goal:** Deliver a fully operational, end-to-end simulated edge intrusion detection system—including the OS simulation harness, flow stream replay generator, calibrated quantized TFLite hybrid detector, embedded SQLite logging, and decoupled real-time Flask monitoring dashboard.  
**Timebox:** 1 committed iteration slice (approx. 5 working days)  
**Pulled From:** `docs/PRODUCT_BACKLOG.md` items: B1.1, B1.2, B1.3, B2.1, B2.2, B2.3, B2.5, B4.1, B4.2, B5.1, B5.2, B5.3, B5.4, B6.1, B6.2, B6.3, B7.2  

---

## 1. Committed Engineering Tasks

| Task ID | User Story | Engineering Task Description | Est. | Depends on | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **T1.1** | B1.1 | Implement virtual environment setup script (`setup_env.sh`) and `requirements.txt` installing `ai-edge-litert`, `scikit-learn`, `flask`, `psutil`, `pandas`, and `numpy`. | S | — | Ready |
| **T1.2** | B1.2, ADR-001 | Implement simulation harness launcher (`scripts/run_simulated.sh`) enforcing `taskset -c 0`, `systemd-run -p MemoryMax=2G`, and `CUDA_VISIBLE_DEVICES=-1`. | S | T1.1 | Ready |
| **T1.3** | B5.1 | Implement SQLite schema initialization script (`src/database.py`) creating `detection_log`, `alert`, `system_metrics`, and `model_registry` tables with WAL mode. | S | T1.1 | Ready |
| **T1.4** | B2.1, B2.2, B2.3 | Implement preprocessing module (`src/preprocessing.py`) with median imputation, categorical encoding, StandardScaler, and PCA variance preservation (>= 95%). | M | T1.1 | Ready |
| **T1.5** | B2.5, B5.4 | Implement network flow stream replay simulator (`src/stream_simulator.py`) buffering flow records into 10-flow sequences and streaming them at configurable arrival rates. | M | T1.4 | Ready |
| **T1.6** | B4.1, B4.2 | Implement and serialize calibrated reference hybrid CNN–LSTM TFLite model (`models/hybrid_model.tflite`) with input shape `(1, 10, Features)` and size <= 150 KB. | M | T1.4 | Ready |
| **T1.7** | B5.2, B5.3 | Implement the edge detection worker (`src/inference_worker.py`) running inference via `ai-edge-litert`, calculating latency, categorizing alerts, and persisting to SQLite. | M | T1.2, T1.3, T1.6 | Ready |
| **T1.8** | B1.3 | Implement background hardware telemetry sampler using `psutil` writing CPU %, RSS memory MB, and flow throughput to the `system_metrics` table. | S | T1.3, T1.7 | Ready |
| **T1.9** | B6.1, B6.2 | Implement decoupled Flask web dashboard (`src/dashboard/app.py` & `templates/index.html`) with REST API endpoints for status, recent detections, and alerts. | M | T1.3 | Ready |
| **T1.10**| B6.2, B6.3 | Build modern dashboard UI with real-time status banner, live classification feed, alert triage with acknowledgement buttons, and system metric gauges. | M | T1.9 | Ready |
| **T1.11**| B7.2 | Implement runtime benchmark script (`src/benchmark_runtime.py`) executing 1,000 warm iterations under the simulation ceiling, logging latency percentiles and peak RSS. | M | T1.2, T1.7 | Ready |

---

## 2. Sprint Definition of Done

This sprint increment is considered **Shippable and Complete** when:
1. Running `bash scripts/run_simulated.sh python3 -m src.benchmark_runtime` successfully executes 1,000 inferences under `taskset -c 0` without crashing.
2. Verified mean latency is strictly `<= 50.0 ms` per 10-flow window.
3. Peak resident memory (RSS) is strictly `<= 512 MB` using `ai-edge-litert`.
4. The stream simulator streams flows into the inference worker, successfully populating SQLite `detection_log` and triggering `alert` rows for malicious flows.
5. Opening `http://localhost:5000` in the browser displays active service status, live stream updates, and allows clicking "Acknowledge" on active alerts.
6. Zero unhandled exceptions in terminal logs.

---

## 3. Burn-down and Daily Sequencing

```
Day 1: T1.1 (Environment) -> T1.2 (Simulation Harness) -> T1.3 (Database Schema)
Day 2: T1.4 (Preprocessing) -> T1.5 (Stream Replay) -> T1.6 (Calibrated TFLite Model)
Day 3: T1.7 (Inference Worker) -> T1.8 (Telemetry Sampler)
Day 4: T1.9 (Flask API) -> T1.10 (Dashboard Web UI)
Day 5: T1.11 (1,000 Iteration Benchmark Run) -> Sprint Review & DoD Validation
```
