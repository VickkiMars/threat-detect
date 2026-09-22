# Requirements Analysis: AI-Powered Network Threat Detection System
## Detailed Engineering Specification & System Analysis

**Document Identifier:** REQ-ANALYSIS-GRACE-2026-V1.0  
**Project Title:** Design and Implementation of an AI-Powered Network Threat Detection System in Resource-Constrained Environments  
**Target Architecture:** Workstation Simulation of Raspberry Pi-Class Edge Node  
**Reference Document:** University of Uyo B.Sc. Project Report (Imeh Grace Mfon, 2026)  

---

## 1. Problem Domain & Engineering Motivation

Modern enterprise and campus network perimeters are increasingly distributed across remote branch offices, IoT gateways, and edge nodes. Traditional Network Intrusion Detection Systems (NIDS) like Snort or Suricata rely heavily on predefined rule signatures. While effective against known attack patterns, signature-based systems:
1. Fail completely against novel zero-day attacks and polymorphic malicious traffic.
2. Require massive, constantly expanding rule databases that saturate edge CPU and memory budgets.
3. Lack the contextual intelligence to correlate temporal traffic patterns across successive packet flows.

Deep learning architectures—specifically Convolutional Neural Networks (CNN) for extracting spatial feature correlations and Long Short-Term Memory (LSTM) networks for modeling temporal sequence dependencies—offer superior threat detection. However, deep neural networks are computationally expensive and memory-intensive, making direct deployment on low-cost single-board computers (like the Raspberry Pi) a major engineering challenge.

This system resolves this tension by combining **aggressive feature reduction (PCA >= 95% variance)**, **class imbalance treatment (SMOTE + Tomek Links)**, **hybrid spatial-temporal modeling (CNN–LSTM)**, **post-training dynamic range quantization (TensorFlow Lite)**, and **a decoupled embedded monitoring stack (SQLite + Flask)**.

---

## 2. In-Depth Functional Requirements Analysis

### 2.1 FR1: Benchmark Flow Ingestion & Stream Replay
- **Functional Need:** The system must process both historical benchmark data for offline training/evaluation and real-time streaming flows during active detection.
- **Analysis:** Live network packet capture (via raw sockets or `pcap`) requires root privileges and introduces heavy packet-parsing CPU overhead on edge devices. Furthermore, live benign campus traffic does not contain labelled attack distributions necessary to prove detector efficacy.
- **Architectural Solution:** Implement a dual-mode data ingestion module:
  1. *Batch Mode:* Direct ingestion of official benchmark CSV archives.
  2. *Stream Replay Mode:* A simulated network stream generator that reads preprocessed or raw flow CSVs, batches them into chronological 10-flow sequences, and streams them into the detection pipeline at configurable rates (10–500 flows/sec) via local IPC or SQLite polling.

### 2.2 FR2: Leakage-Controlled Preprocessing & Dimensionality Reduction
- **Functional Need:** Clean messy network flow features, standardize diverse numeric scales, and reduce dimensionality without information leakage.
- **Analysis:** Network flow datasets contain extreme anomalies: infinite values from zero-duration flow divisions (`Flow Bytes/s`, `Flow Packets/s`), missing values, high-cardinality categorical protocols, and high feature dimensionality (78 features in CICIDS2017, 45 in UNSW-NB15).
- **Processing Steps:**
  - *Cleaning:* Replace infinite values with column maximums; impute null values with column medians.
  - *Categorical Encoding:* Encode nominal features (`proto`, `service`, `state`) using one-hot encoding or preserved integer index mappings.
  - *Standardization:* Compute mean $\mu$ and standard deviation $\sigma$ strictly on the training partition ($X_{train}$). Apply $(x - \mu)/\sigma$ across train, validation, test, and streaming inputs.
  - *Principal Component Analysis (PCA):* Compute orthogonal principal components on scaled $X_{train}$ such that cumulative explained variance ratio $\sum \lambda_i \ge 0.95$. In CICIDS2017, this compresses 78 features into **22 principal components** while preserving 95.7233% of total data variance.

### 2.3 FR3: Training Class Imbalance Treatment (SMOTE + Tomek Links)
- **Functional Need:** Prevent the classifier from developing a majority-class bias toward benign traffic while ensuring minority attack patterns (e.g., Infiltration, Botnet, Shellcode) are reliably detected.
- **Analysis:** Intrusion datasets exhibit severe class imbalance (often >80% benign). Naive training results in models with deceptively high overall accuracy (>90%) but near-zero recall on rare, critical attacks.
- **Processing Steps:**
  - *SMOTE (Oversampling):* Synthesize new minority attack samples along the line segments connecting $k$-nearest neighbors in feature space.
  - *Tomek Links (Cleaning):* Identify pairs of nearest neighbors from opposite classes. If two samples are each other's nearest neighbors, the link is ambiguous/noisy; remove the majority instance (or both) to clarify decision boundaries.
  - *Boundary Rule:* SMOTE + Tomek Links must **only** be executed on the training partition. Applying oversampling or synthetic cleaning to validation or test splits introduces severe synthetic data leakage, invalidating academic evaluation.

### 2.4 FR4: Hybrid CNN–LSTM vs Baseline Model Architectures
- **Functional Need:** Detect complex attacks by learning both within-flow feature patterns and cross-flow temporal transitions.
- **Analysis:**
  - Individual flow records represent isolated snapshots. Advanced attacks (DDoS surges, slow port scans, multi-stage infiltration) exhibit temporal dependencies spanning multiple successive flows.
  - Grouping flows into non-overlapping windows of $W=10$ records allows a 3D input tensor $(Batch, 10, Features)$.
- **Model Architectural Specification:**
  - **Conv1D Layer:** 64 filters, kernel size 3, ReLU activation. Extracts local spatial correlations across adjacent features within each flow.
  - **MaxPooling1D Layer:** Pool size 2. Reduces dimensionality and preserves dominant feature activations.
  - **Dropout Layer:** Rate 0.20. Prevents over-fitting on training flow patterns.
  - **LSTM Layer:** 64 recurrent memory units. Models temporal progression, recurrence, and long-range dependencies across the 10 sequential flow timesteps.
  - **Dense Output Layer:** 2 units with Softmax activation providing calibrated class probabilities for `[0: Benign, 1: Attack]`.
- **Baseline Comparative Set:**
  - Classical: Decision Tree (CART), Random Forest (100 estimators), Support Vector Machine (RBF kernel) trained on flattened PCA sequences $(Batch, 10 \times Features)$.
  - Deep Learning: CNN-only model (Conv1D + Dense) and LSTM-only model (LSTM + Dense).

### 2.5 FR5: Post-Training Quantization & TensorFlow Lite Serialization
- **Functional Need:** Minimize storage footprint, memory consumption, and instruction execution cycles for edge environments.
- **Analysis:** A standard 32-bit floating-point (FP32) Keras model requires significant memory and relies on complex floating-point hardware execution units.
- **Optimization Strategy:**
  - Dynamic-Range Post-Training Quantization: Convert 32-bit floating-point weights to 8-bit integers (`INT8`). Activations remain dynamic FP32 at runtime.
  - Shaves model size by ~75% (from ~400 KB down to **100.08 KB**).
  - Enables execution via the standalone `ai-edge-litert` interpreter, bypassing the massive memory overhead of importing full TensorFlow.

### 2.6 FR6–FR8: Edge Inference, Alerting & SQLite Persistence
- **Functional Need:** Deliver real-time inference, security triage, and telemetry storage without saturating single-core edge resources.
- **Analysis:** Enterprise databases (PostgreSQL, MySQL) require independent daemon processes consuming hundreds of megabytes of RAM.
- **Architectural Solution:**
  - Use an embedded SQLite database (`threat_detection.db`) in WAL (Write-Ahead Logging) mode.
  - Single-core friendly: Zero background daemon overhead, zero network socket overhead, instant file-based lookups.
  - Record detections, classified alerts, and rolling hardware metrics (`psutil`: CPU %, Memory RSS MB, Throughput flows/sec).

### 2.7 FR9: Real-Time Web Monitoring Dashboard
- **Functional Need:** Provide network operators with actionable situational awareness.
- **Analysis:** The dashboard must be visually informative yet ultra-lightweight.
- **Architectural Solution:** Flask web application serving vanilla HTML5, CSS3, and JavaScript. Uses asynchronous AJAX polling (`fetch`) to query SQLite via JSON endpoints, completely decoupled from the inference worker process.

---

## 3. In-Depth Non-Functional Requirements Analysis

```
+-----------------------------------------------------------------------------------------+
|                               RESOURCE CEILINGS & BOUNDARIES                           |
+------------------------------------+----------------------------------------------------+
| Parameter                          | System Constraint / Target Value                  |
+------------------------------------+----------------------------------------------------+
| NFR1: Mean Inference Latency       | <= 50.0 ms per 10-flow sequence                    |
| NFR2: Model Storage Size           | <= 1.0 MB (Quantized TFLite target: ~100 KB)       |
| NFR3: Detection Service Memory RSS | <= 512 MB sustained (Host container ceiling: 2 GB) |
| NFR4: Test Set Accuracy & Recall   | >= 95.0% Accuracy; >= 95.0% Attack Recall          |
| NFR5: Execution Hardware           | 100% CPU-only (Zero GPU acceleration required)     |
| NFR6: Data Privacy & Locality      | Zero external network egress                       |
| NFR7: Preprocessor Co-Versioning   | Cryptographically validated artifact pairing       |
+------------------------------------+----------------------------------------------------+
```

### 3.1 NFR1: Latency Budget Breakdown
At 50 flows/sec arrival rate, a 10-flow window accumulates every 200 ms. An inference latency ceiling of **50 ms** provides a 4x safety margin, guaranteeing zero buffer backlog and zero flow dropping on a single CPU core. In actual testing, the quantized hybrid model achieved **0.0095 ms** simulated inference latency per sequence on the x86 workstation.

### 3.2 NFR3: Memory Architecture (LiteRT vs Full TensorFlow)
Importing full `tensorflow` into a Python process immediately loads CUDA stubs, compilation graphs, and monolithic runtime binaries, consuming **687.41 MB** peak RSS—violating the 512 MB service ceiling. By contrast, executing the model via the standalone `ai-edge-litert` interpreter consumes only **49.93 MB** peak RSS, easily satisfying NFR3.

---

## 4. Benchmark Dataset Specifications

### 4.1 Dataset Properties & Roles
| Dataset Property | UNSW-NB15 | CICIDS2017 (Capped Subset) |
| :--- | :--- | :--- |
| **Originating Institution** | Australian Defence Force Academy (ADFA) | Canadian Institute for Cybersecurity (CIC) |
| **Capture Period** | 2015 | 2017 |
| **Raw Features** | 45 features (numeric + categorical) | 78 features (all numeric flow stats) |
| **Class Distribution** | Binary (`attack_cat` mapped to 0/1) | Binary (Benign vs Malicious attacks) |
| **Subset Scope** | Standard train/test partition (3,865 test sequences) | 200,000-row class-capped subset |
| **PCA Components** | Retaining >= 95% variance | 22 components (95.7233% variance) |
| **Role in Study** | Primary benchmark for verified TFLite edge simulation | Cross-dataset generalizability validation |

### 4.2 Handling the CICIDS2017 Memory Boundary
The full raw CICIDS2017 dataset comprises over 2.8 million flow records across 8 separate CSV files totaling > 3 GB. Ingesting, concatenating, and running SMOTE on the entire dataset exceeds the memory budget of a standard 16 GB workstation and causes Out-Of-Memory (OOM) crashes. As documented in the dissertation, a stratified **200,000-row class-capped subset** is extracted to preserve rare attack classes (Heartbleed, Botnet, Infiltration, Web Attacks) while maintaining strict execution within memory limits.

---

## 5. Workstation Simulation Specification

### 5.1 Host System Hardware vs Target Hardware
| Attribute | Target Edge Hardware (Raspberry Pi 4B) | Workstation Simulation Environment |
| :--- | :--- | :--- |
| **Processor** | Broadcom BCM2711, Quad-core Cortex-A72 | Intel Core i7-8665U @ 1.90GHz (Host) |
| **Architecture** | ARMv8-A (64-bit ARM) | x86_64 |
| **Simulated Active Cores**| 1 Physical Core | **1 Logical Core** (Enforced via `taskset -c 0`) |
| **Memory Ceiling** | 2 GB / 4 GB LPDDR4 | **2 GB Ceiling** (`systemd-run` / `ulimit -v`) |
| **Service Target RAM** | < 512 MB | **< 512 MB RSS** (Verified with `psutil`) |
| **Operating System** | Raspberry Pi OS (Debian Linux) | Ubuntu 22.04 LTS Linux |
| **Power Measurement** | Hardware USB Multimeter / INA219 | **NOT MEASURED** (Explicit simulation boundary) |

### 5.2 Simulation Enforcement Commands
1. **CPU Pinning:** `taskset --cpu-list 0` restricts the execution thread to CPU core 0.
2. **Memory Containment:** `systemd-run --user --scope -p MemoryMax=2G -p OOMPolicy=stop` enforces an unyielding Linux cgroup memory boundary.
3. **Execution Flags:** `env CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1` guarantees single-threaded, CPU-only math kernel execution.
