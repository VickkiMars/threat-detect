# Software Requirements Specification (SRS)
## AI-Powered Network Threat Detection System in Resource-Constrained Environments

**Document Identifier:** SRS-GRACE-2026-V1.0  
**Project Title:** Design and Implementation of an AI-Powered Network Threat Detection System in Resource-Constrained Environments  
**Author:** Imeh Grace Mfon (Registration Number: 21/SC/CO/1117)  
**Institution:** Department of Computer Science, Faculty of Computing, University of Uyo, Uyo, Nigeria  
**Supervisor:** Mr. U. J. Ntia  
**Execution Target:** Workstation-Simulated Raspberry Pi-Class Environment (Ubuntu Linux / x86_64)  
**Standard Compliance:** Aligned with IEEE 830 / ISO/IEC/IEEE 29148  

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) establishes a formal, comprehensive blueprint defining the functional, non-functional, interface, and performance requirements for the **AI-Powered Network Threat Detection System in Resource-Constrained Environments**. It provides the authoritative specification for implementing, testing, verifying, and evaluating the software system on a controlled Ubuntu Linux workstation simulating a Raspberry Pi-class edge platform.

### 1.2 Scope of the System
The software system is a lightweight, edge-oriented Network Intrusion Detection System (NIDS) designed to protect resource-constrained network perimeters. The system:
- Ingests network flow records from standard benchmark datasets (UNSW-NB15 and CICIDS2017) and supports a streaming playback replay interface.
- Applies a rigorous, leakage-controlled data preprocessing pipeline: cleaning, numerical normalization, categorical encoding, dimensionality reduction via Principal Component Analysis (PCA >= 95% variance retention), sequence windowing (10 consecutive flow records), and class balancing (SMOTE + Tomek Links applied strictly to training partitions).
- Implements and evaluates comparative detection models (Classical baselines: Decision Tree, Random Forest, SVM; Deep Learning baselines: CNN-only, LSTM-only; Proposed Hybrid: CNN–LSTM).
- Converts and optimizes the hybrid model for edge execution via 8-bit dynamic-range quantization and TensorFlow Lite FlatBuffer serialization.
- Executes real-time inference within a single-threaded, CPU-only edge runtime (`ai-edge-litert` / TFLite interpreter).
- Persists all classification events, security alerts, and system health metrics to a local, serverless SQLite database.
- Provides a web-based, real-time monitoring dashboard (Flask, HTML5, CSS3, JavaScript) displaying service health, recent traffic classifications, active alerts, and resource utilization metrics.
- Enforces and measures performance against strict resource ceilings: single logical CPU core, memory ceiling of 512 MB (sustained detection service) / 2 GB (host execution container), and inference latency <= 50 ms per 10-flow window.

### 1.3 Definitions, Acronyms, and Abbreviations
| Term / Acronym | Definition |
| :--- | :--- |
| **ADR** | Architecture Decision Record |
| **CICIDS2017** | Canadian Institute for Cybersecurity Intrusion Detection System 2017 Dataset |
| **CNN** | Convolutional Neural Network (extracts local feature interactions across flows) |
| **DoD** | Definition of Done |
| **FlatBuffer** | Efficient cross-platform serialization format used by TensorFlow Lite |
| **FPR** | False Positive Rate ($FP / (FP + TN)$) |
| **LSTM** | Long Short-Term Memory Network (captures temporal sequential dependencies) |
| **NIDS** | Network Intrusion Detection System |
| **PCA** | Principal Component Analysis (orthogonal linear transformation for dimensionality reduction) |
| **QAT** | Quantization-Aware Training |
| **RSS** | Resident Set Size (actual physical memory occupied by a process) |
| **SMOTE** | Synthetic Minority Over-sampling Technique |
| **SRS** | Software Requirements Specification |
| **TFLite** | TensorFlow Lite (runtime engine optimized for on-device machine learning) |
| **Tomek Links** | Under-sampling algorithm that removes ambiguous/noisy boundary instances |
| **UNSW-NB15** | University of New South Wales Network Benchmark 2015 Dataset |

### 1.4 References
1. University of Uyo B.Sc. Project Report: *Design and Implementation of an AI-Powered Network Threat Detection System in Resource Constrained Environments*, by Imeh Grace Mfon (August 2026).
2. IEEE Std 830-1998: *IEEE Recommended Practice for Software Requirements Specifications*.
3. ISO/IEC/IEEE 29148:2018: *Systems and software engineering — Life cycle processes — Requirements engineering*.
4. Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *MilCIS 2015*.
5. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP 2018*.
6. Google TensorFlow: *TensorFlow Lite 8-Bit Dynamic Range Quantization Specification*.

---

## 2. Overall Description

### 2.1 Product Perspective & Six-Layer Architecture
The software is an autonomous, self-contained edge intrusion detection appliance. It is architected into six clearly decoupled layers to ensure strict separation of concerns, auditable data lineage, and predictable resource consumption:

```
+-----------------------------------------------------------------------------------+
| 6. Constrained Deployment & Presentation Layer                                    |
|    - Flask Monitoring Dashboard (Uptime, Alerts, Flow Telemetry, Resource Gauges)  |
|    - SQLite Embedded Database (detection_log, alert, system_metrics, registry)    |
+-----------------------------------------------------------------------------------+
                                         ^
                                         | Events, Alerts & Metrics
+-----------------------------------------------------------------------------------+
| 5. Detection & Response Layer                                                     |
|    - Window-level Classification & Confidence Scoring                             |
|    - Malicious Alert Generation & Security Operations Triage                      |
+-----------------------------------------------------------------------------------+
                                         ^
                                         | Inferences
+-----------------------------------------------------------------------------------+
| 4. Model Optimisation & Edge Preparation Layer                                    |
|    - 8-Bit Dynamic-Range Post-Training Quantization (Weight INT8, Activations FP32)|
|    - TensorFlow Lite FlatBuffer Export & Verification                             |
|    - Standalone ai-edge-litert / TFLite Single-Threaded Interpreter Runtime        |
+-----------------------------------------------------------------------------------+
                                         ^
                                         | Optimized Models
+-----------------------------------------------------------------------------------+
| 3. Hybrid Detection Engine Layer                                                  |
|    - Spatial Feature Extraction: 1D Convolutional Neural Network (Conv1D + Pool)  |
|    - Temporal Sequence Modelling: Long Short-Term Memory (LSTM)                   |
|    - Baseline Benchmarking: CNN-only, LSTM-only, Decision Tree, Random Forest, SVM|
+-----------------------------------------------------------------------------------+
                                         ^
                                         | 10-Flow Window Tensors (N, 10, Features)
+-----------------------------------------------------------------------------------+
| 2. Preprocessing & Feature Engineering Layer                                      |
|    - Missing Value & Infinite Number Imputation / Cleaning                        |
|    - Categorical Encoding & Standard Scaling (Z-score normalization)              |
|    - Dimensionality Reduction: PCA (retaining >= 95% training variance)           |
|    - Class Balancing: SMOTE + Tomek Links (Training partition only)               |
|    - Sequence Formatting: Fixed non-overlapping windows of 10 flow records        |
+-----------------------------------------------------------------------------------+
                                         ^
                                         | Raw Flow Records
+-----------------------------------------------------------------------------------+
| 1. Data Acquisition Layer                                                         |
|    - Benchmark Dataset Parsers: UNSW-NB15 & CICIDS2017 (200k capped subset)       |
|    - Stream Playback & Replay Engine (simulated live network flow generator)     |
+-----------------------------------------------------------------------------------+
```

### 2.2 Operational Environment & Simulation Constraints
The operational target is an edge computing device representative of the **Raspberry Pi 4 Model B (Quad-core Cortex-A72 @ 1.5GHz, 2GB–4GB RAM)**. Because physical hardware deployment is reserved for future hardware transfer, the authoritative operational environment is an **Ubuntu 22.04 LTS / compatible Linux workstation** operating under strict simulated constraints:
- **CPU Affinity:** Pinned to exactly **1 logical CPU core** (`taskset --cpu-list 0`).
- **Memory Ceiling:** Controlled host limit of **2 GB virtual memory** (`systemd-run --user --scope -p MemoryMax=2G -p OOMPolicy=stop` or `ulimit -v 2097152`).
- **Service Memory Budget:** Steady-state resident memory (RSS) must remain below **512 MB** during active inference.
- **Hardware Acceleration:** Strictly disabled (`CUDA_VISIBLE_DEVICES=-1`, no OpenCL/Vulkan delegates).
- **Execution Concurrency:** Interpreter intra-op and inter-op threads restricted to 1 (`TF_NUM_INTRAOP_THREADS=1`, `TF_NUM_INTEROP_THREADS=1`).
- **Edge Interpreter:** Python 3.10+ utilizing the standalone `ai-edge-litert` or lightweight TensorFlow Lite runtime to avoid full TensorFlow library overhead.

### 2.3 User Classes and Characteristics
1. **Network Security Analyst (SecOps):** Monitors the dashboard, investigates classified intrusion alerts, inspects flow confidence levels, and acknowledges active security events.
2. **Edge System Administrator:** Deploys and manages the inference service, configures traffic replay rates, monitors CPU and RAM telemetry, and verifies resource ceilings.
3. **Academic Researcher / Examiner (Student):** Audits experimental reproducibility, reviews preprocessing artifacts, inspects confusion matrices and training curves, and validates performance metrics against the published dissertation results.
4. **Machine Learning Engineer:** Retrains models, tunes hyperparameters, manages PCA transformations, and runs the quantization and serialization pipelines.

### 2.4 Design and Implementation Constraints
- **Zero Cloud / External Dependence:** All processing, feature scaling, model inference, and event logging must run locally without external API queries or internet egress.
- **Data Leakage Immunity:** Preprocessing objects (scalers, PCA transformers, label encoders) must fit strictly on training splits. Oversampling (SMOTE + Tomek Links) must **never** be applied to validation or test splits.
- **Sequence Temporal Coherence:** Detections are performed on chronological sequences of exactly 10 flow records; synthetic padding is applied only when flushing incomplete terminal streams.
- **Decoupled Architecture:** The web dashboard and detection worker must run as separate processes communicating through the SQLite database to avoid web request latency contaminating inference benchmarks.

---

## 3. Specific System Features & Functional Requirements

### FR1: Benchmark Dataset Ingestion and Live Stream Playback
- **Description:** The system shall ingest benchmark network intrusion datasets and provide a configurable stream replay mechanism to simulate real-time network flow arrival.
- **Inputs:** Raw CSV files from UNSW-NB15 and CICIDS2017 (200,000 class-capped subset).
- **Processing:**
  - Ingest raw CSV headers and validate schema consistency.
  - Replay flow records sequentially at configurable arrival frequencies (e.g., 10 to 500 flows/sec).
  - Buffer records into non-overlapping sliding/tumbling windows of 10 flow records.
- **Outputs:** Standardized structured dictionaries/tensors representing flow sequences.

### FR2: Leakage-Controlled Preprocessing and PCA Reduction
- **Description:** The system shall clean, encode, standardize, and reduce the dimensionality of flow records using frozen transformers.
- **Processing:**
  - Strip whitespace, impute null/missing entries with median values, and replace infinite values (`+inf`, `-inf`) with finite maximums.
  - Encode categorical attributes (`proto`, `service`, `state`) using one-hot or preserved integer index mappings.
  - Fit `StandardScaler` strictly on the training partition; apply the frozen scaler to validation, test, and live stream inputs.
  - Apply Principal Component Analysis (PCA) retaining >= 95% cumulative training variance (22 components for CICIDS2017; calibrated component set for UNSW-NB15).
  - Persist fitted scalers, encoders, and PCA transformers as serialized pipeline artifacts (`joblib` / `pickle`).
- **Outputs:** Dimension-reduced float32 feature vectors.

### FR3: Training Partition Class Balancing (SMOTE + Tomek Links)
- **Description:** The system shall balance attack and benign class distributions in the training split without distorting evaluation integrity.
- **Processing:**
  - Apply Synthetic Minority Over-sampling Technique (SMOTE) to synthetically balance minority attack classes.
  - Apply Tomek Links under-sampling to remove ambiguous, overlapping boundary pairs between attack and benign clusters.
  - Restrict balancing exclusively to the training partition (`X_train`, `y_train`).
- **Outputs:** Balanced training dataset ready for sequence windowing.

### FR4: Baseline and Hybrid Model Training Pipeline
- **Description:** The system shall support training and comparing baseline models against the proposed hybrid CNN–LSTM architecture.
- **Processing:**
  - Reshape training sequences into 3D tensors: `(batch_size, 10, num_features)`.
  - Train Classical Baselines: Decision Tree, Random Forest, Support Vector Machine (flattened sequence representation).
  - Train Deep Learning Baselines: CNN-only (1D Convolution + MaxPooling + Dense) and LSTM-only (LSTM cells + Dense).
  - Train Proposed Hybrid Model:
    - Input Layer: `(None, 10, num_features)`
    - Conv1D Layer: 64 filters, kernel size 3, ReLU activation, padding 'same'
    - MaxPooling1D Layer: pool size 2
    - Dropout Layer: rate 0.2
    - LSTM Layer: 64 units, returning sequences or terminal state
    - Dense Output Layer: 2 units with Softmax activation (Binary classification: Benign=0, Attack=1)
  - Train using Adam optimizer, Binary Cross-Entropy loss, Early Stopping, and Model Checkpointing.
- **Outputs:** Saved Keras HDF5/SavedModel artifacts and training logs.

### FR5: Dynamic Range Model Quantization and TFLite Conversion
- **Description:** The system shall compress the trained hybrid Keras model for resource-constrained execution.
- **Processing:**
  - Convert trained model to TensorFlow Lite FlatBuffer format.
  - Apply post-training dynamic-range quantization (weights converted to 8-bit integers, activations evaluated dynamically in float32).
  - Validate that model size is <= 1 MB (target ~100 KB for hybrid architecture).
  - Verify exported model tensor signatures: Input `(1, 10, num_features)`, Output `(1, 2)`.
  - Persist `.tflite` model alongside matching metadata and preprocessing artifacts.
- **Outputs:** Deployable `.tflite` model file.

### FR6: Windowed Inference and Confidence Scoring
- **Description:** The edge detection worker shall classify arriving 10-flow sequences using the lightweight TFLite runtime.
- **Processing:**
  - Allocate TFLite interpreter using single CPU thread (`ai-edge-litert`).
  - Feed prepared sequence tensor into the input tensor buffer.
  - Invoke interpreter and extract probability distribution over `[Benign, Attack]`.
  - Assign predicted label based on argmax / decision threshold (default 0.50).
  - Calculate inference latency via high-resolution wall-clock timer (`time.perf_counter_ns()`).
- **Outputs:** Binary classification label, confidence score (0.0 to 1.0), and latency measurement (ms).

### FR7: Security Alert Generation and Triage
- **Description:** The system shall immediately raise a security alert when a sequence is classified as malicious.
- **Processing:**
  - Check prediction against attack threshold.
  - Assign severity level based on prediction confidence:
    - `LOW`: Confidence 0.50 – 0.69
    - `MEDIUM`: Confidence 0.70 – 0.84
    - `HIGH`: Confidence 0.85 – 0.94
    - `CRITICAL`: Confidence 0.95 – 1.00
  - Construct alert message detailing timestamp, source/destination IP, protocol, confidence, and sequence ID.
  - Queue alert for database persistence and immediate dashboard notification.
- **Outputs:** Structured alert records.

### FR8: Embedded SQLite Telemetry and Event Persistence
- **Description:** The system shall persist all detection logs, security alerts, and system telemetry to a local SQLite database.
- **Processing:**
  - Maintain four normalized relational tables: `detection_log`, `alert`, `system_metrics`, and `model_registry`.
  - Sample CPU usage, memory RSS (MB), and throughput (flows/sec) every 1–5 seconds via `psutil`.
  - Execute database writes using indexed queries and write-ahead logging (WAL mode) to avoid database lock contention.
- **Outputs:** Persisted records in `data/threat_detection.db`.

### FR9: Real-Time Web Monitoring Dashboard
- **Description:** The system shall provide a lightweight web user interface for real-time threat visualization and telemetry monitoring.
- **Processing:**
  - Serve UI using Flask with HTML5, CSS3, and JavaScript.
  - Provide asynchronous JSON polling endpoints:
    - `GET /api/status`: Service uptime, active model version, total flows analyzed.
    - `GET /api/detections/recent`: Latest 50 flow classifications.
    - `GET /api/alerts/unacknowledged`: Active unacknowledged threat alerts.
    - `POST /api/alerts/<id>/acknowledge`: Mark alert as reviewed.
    - `GET /api/metrics/system`: Rolling CPU, memory, and throughput telemetry.
  - Render responsive dashboard panels: Status Banner, Live Feed, Prioritized Alert Queue, and Resource Utilization Charts.
- **Outputs:** Browser-accessible web interface (`http://localhost:5000`).

### FR10: Comprehensive Evaluation and Academic Benchmarking
- **Description:** The system shall calculate, tabulate, and plot formal intrusion detection metrics on held-out test datasets.
- **Processing:**
  - Calculate Accuracy, Precision, Recall (True Positive Rate), False Positive Rate (FPR), and Macro-averaged F1 Score.
  - Generate Confusion Matrices for all baseline and hybrid models.
  - Benchmark inference latency: mean, median, 95th-percentile (p95), min, max, and standard deviation over 1,000 iterations.
  - Measure peak Resident Set Size (RSS) memory consumption under simulated resource limits.
- **Outputs:** Tabular summary CSVs and high-resolution evaluation figures (PNG).

---

## 4. Non-Functional Requirements (NFR)

| ID | Category | Requirement Specification | Verification Method |
| :--- | :--- | :--- | :--- |
| **NFR1** | **Inference Latency** | Mean inference latency shall not exceed **50 ms** per 10-flow sequence under simulated single-core CPU execution. | Automated benchmark script measuring 1,000 warm iterations with `time.perf_counter_ns()`. |
| **NFR2** | **Model Storage Size** | Deployed TensorFlow Lite model file size shall not exceed **1.0 MB** (target <= 150 KB for quantized hybrid model). | File system binary size check (`ls -lh` / `os.path.getsize`). |
| **NFR3** | **Memory Consumption** | Detection service shall sustain steady-state memory utilization below **512 MB RSS**, operating within a 2 GB host ceiling. | Continuous `psutil.Process().memory_info().rss` profiling under cgroup/systemd-run containment. |
| **NFR4** | **Detection Quality** | Held-out test accuracy shall exceed **95.0%**, with attack recall >= 95.0% and False Positive Rate <= 1.0% on UNSW-NB15. | Held-out test evaluation against frozen test partition. |
| **NFR5** | **Platform Independence** | The detection service shall execute entirely on CPU architectures without requiring GPU hardware acceleration. | Executed with `CUDA_VISIBLE_DEVICES=-1` and verified via PyTorch/TensorFlow device enumeration. |
| **NFR6** | **Data Privacy & Locality** | All traffic analysis, classification, and event logging shall occur locally; zero captured payloads or telemetry shall egress. | Network socket inspection and firewall rules confirming zero outbound external network connections. |
| **NFR7** | **Artifact Co-Versioning** | Preprocessing objects (scaler, encoder, PCA) and label mappings shall be strictly versioned and bonded with the model. | Manifest validation checking cryptographic checksums and schema version compatibility at service startup. |

---

## 5. Interface Requirements

### 5.1 Software Interfaces
- **Operating System:** Ubuntu Linux 22.04 LTS or compatible POSIX Linux distribution.
- **Runtime Environment:** Python 3.10+ (specifically validated on Python 3.12.3).
- **Core Libraries:**
  - `ai-edge-litert` / `tflite-runtime`: Lightweight edge model execution.
  - `scikit-learn` (v1.3+): StandardScaler, PCA, metrics calculation.
  - `imbalanced-learn` (v0.11+): SMOTE and TomekLinks algorithms.
  - `pandas` & `numpy`: Data ingestion, tensor shaping, and array manipulation.
  - `flask`: Micro web framework serving REST API endpoints and dashboard UI.
  - `sqlite3`: Native relational persistence engine.
  - `psutil`: OS hardware metrics instrumentation.

### 5.2 Hardware and Simulation Interfaces
- **CPU Affinity Interface:** Linux `sched_setaffinity` syscall invoked via `taskset --cpu-list 0`.
- **Memory Control Interface:** Linux `cgroups v2` hierarchy enforced via `systemd-run --user --scope -p MemoryMax=2G -p OOMPolicy=stop` or `ulimit -v 2097152`.
- **Timer Subsystem:** POSIX monotonic clock via Python standard library `time.perf_counter_ns()`.

### 5.3 User Interfaces
- **Web UI:** Responsive single-page dashboard accessible over standard HTTP (port 5000), compatible with modern web browsers (Chrome, Firefox, Safari, Edge).
- **Command-Line Interface (CLI):** Command-line scripts for training (`src/train.py`), quantization (`src/convert_tflite.py`), benchmarking (`src/benchmark_runtime.py`), and replay simulation (`src/stream_simulator.py`).

---

## 6. Verification & Traceability Matrix

| Requirement ID | Description | Source / University Dissertation Section | Implementation Module | Verification Test |
| :--- | :--- | :--- | :--- | :--- |
| **FR1** | Flow Ingestion & Replay | Chapter 3.6, Table 3.4 (FR1) | `src/stream_simulator.py` | `tests/test_stream_simulator.py` |
| **FR2** | Preprocessing & PCA | Chapter 3.8, Table 3.4 (FR2) | `src/preprocessing.py` | `tests/test_preprocessing.py` |
| **FR3** | SMOTE + Tomek Balancing | Chapter 3.8, Table 3.4 (FR3) | `src/train.py` | `tests/test_balancing.py` |
| **FR4** | CNN–LSTM & Baselines | Chapter 3.9, Table 3.4 (FR4) | `src/models/` | `tests/test_models.py` |
| **FR5** | TFLite Quantization | Chapter 3.10, Table 3.4 (FR5) | `src/convert_tflite.py` | `tests/test_quantization.py` |
| **FR6** | Windowed Inference | Chapter 4.8, Table 3.4 (FR6) | `src/inference_engine.py` | `tests/test_inference.py` |
| **FR7** | Threat Alerting | Chapter 3.12, Table 3.4 (FR7) | `src/alert_manager.py` | `tests/test_alerts.py` |
| **FR8** | SQLite Telemetry Persistence| Chapter 3.12, Table 3.4 (FR8) | `src/database.py` | `tests/test_database.py` |
| **FR9** | Web Monitoring Dashboard | Chapter 4.8, Table 3.4 (FR9) | `src/dashboard/` | Browser & API functional test |
| **FR10** | Metrics & Benchmarking | Chapter 3.13, Table 3.4 (FR10) | `src/evaluate.py`, `src/benchmark_runtime.py`| Automated test suites |
| **NFR1** | Latency <= 50ms | Chapter 4.10, Table 3.5 (NFR1) | `src/benchmark_runtime.py` | Latency benchmark under taskset |
| **NFR2** | Model Size <= 1MB | Chapter 4.7, Table 3.5 (NFR2) | `src/convert_tflite.py` | Binary size validation check |
| **NFR3** | Memory RSS < 512MB | Chapter 4.10, Table 3.5 (NFR3) | `src/benchmark_runtime.py` | psutil peak RSS profiling |
| **NFR4** | Accuracy > 95% | Chapter 4.9, Table 3.5 (NFR4) | `src/evaluate.py` | Test partition evaluation run |
| **NFR5** | Zero GPU Dependency | Chapter 4.10, Table 3.5 (NFR5) | Runtime configuration | `CUDA_VISIBLE_DEVICES=-1` test |
| **NFR6** | Data Locality | Chapter 3.11, Table 3.5 (NFR6) | Core architecture | Socket listener audit |
| **NFR7** | Pipeline Co-Versioning | Chapter 3.10, Table 3.5 (NFR7) | `src/model_registry.py` | Checksum manifest validation |
