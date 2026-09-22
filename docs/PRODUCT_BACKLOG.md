# Product Backlog — AI-Powered Network Threat Detection System

**Project:** AI-Powered Network Threat Detection System in Resource-Constrained Environments  
**Target Platform:** Ubuntu Linux Workstation Simulating a Raspberry Pi-Class Edge Device  
**Assumption:** Solo engineering implementation; reproducible software simulation on Ubuntu PC; benchmark evaluation on UNSW-NB15 and 200k CICIDS2017 subset.

---

## Epic 1: Simulation Harness & Edge Environment Setup
*Establishes the resource containment boundaries (single-core CPU affinity, memory cgroup ceiling, GPU disablement) on the host Ubuntu workstation.*

| ID | Story | Priority | Est. | Depends on | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B1.1** | As a system administrator, I want a standardized environment setup script (`setup_env.sh`), so that all dependencies and lightweight runtimes (`ai-edge-litert`, `scikit-learn`, `flask`) are installed into an isolated virtual environment. | High | S | — | Validated on Python 3.10–3.12 |
| **B1.2** | As an academic evaluator, I want an OS-level simulation harness script (`run_simulated.sh`) enforcing `taskset -c 0` and memory limits, so that the detector executes under strict Raspberry Pi-class resource budgets. | High | S | B1.1, ADR-001 | Implements Chapter 4.10.2 commands |
| **B1.3** | As an edge engineer, I want a hardware instrumentation utility using `psutil`, so that real-time process CPU utilization, Resident Set Size (RSS), and memory limits are reliably sampled and logged. | High | S | B1.1 | Samples every 1s |

---

## Epic 2: Data Acquisition & Preprocessing Pipeline
*Ingests benchmark datasets, enforces data leakage separation, applies PCA, balances classes via SMOTE-Tomek, and windows flows into 10-step sequences.*

| ID | Story | Priority | Est. | Depends on | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B2.1** | As an ML engineer, I want dataset loaders for UNSW-NB15 and CICIDS2017 (200k subset), so that raw benchmark flows are parsed and cleaned with consistent column schemas. | High | M | B1.1 | Imputes nulls, strips infinities |
| **B2.2** | As an ML engineer, I want a leakage-free preprocessing pipeline fitting scalers and categorical encoders strictly on training splits, so that data leakage does not contaminate evaluation splits. | High | M | B2.1 | Frozen transformers saved via joblib |
| **B2.3** | As an ML engineer, I want a PCA dimensionality reduction transformer retaining >= 95% cumulative variance, so that input features are compressed to 22 components without losing critical threat variance. | High | M | B2.2 | Enforces Chapter 3.8.1/3.8.2 |
| **B2.4** | As an ML engineer, I want SMOTE oversampling combined with Tomek Links under-sampling applied strictly to training partitions, so that severe attack class imbalance is resolved without skewing test splits. | High | L | B2.3 | Uses `imbalanced-learn` |
| **B2.5** | As an ML engineer, I want a sequence formatter grouping sequential flows into non-overlapping windows of 10 records, so that temporal relationships can be ingested by sequential deep neural networks. | High | S | B2.3 | Yields 3D tensors `(N, 10, Features)` |

---

## Epic 3: Baseline & Hybrid Model Training Pipeline
*Constructs, trains, and evaluates comparative intrusion detection models (Classical, CNN-only, LSTM-only, and Hybrid CNN–LSTM).*

| ID | Story | Priority | Est. | Depends on | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B3.1** | As an academic researcher, I want training pipelines for classical machine learning baselines (Decision Tree, Random Forest, SVM) on flattened PCA sequences, so that deep learning gains are benchmarked against traditional approaches. | Medium | M | B2.5 | Flattened `(N, 10 * Features)` |
| **B3.2** | As an academic researcher, I want training pipelines for standalone CNN-only and LSTM-only neural architectures, so that the individual spatial and temporal contributions can be isolated. | Medium | M | B2.5 | DL baselines from Chapter 4.5 |
| **B3.3** | As an ML engineer, I want to construct and train the proposed hybrid CNN–LSTM model (Conv1D 64 -> MaxPool 2 -> Dropout 0.2 -> LSTM 64 -> Dense 2 Softmax), so that spatial features and temporal sequences are jointly learned. | High | L | B2.5 | Binary cross-entropy, Adam optimizer |
| **B3.4** | As an ML engineer, I want early stopping and model checkpointing during training, so that optimal model weights are captured without overfitting. | Medium | S | B3.3 | Monitored on validation loss |

---

## Epic 4: Model Compression & Edge Quantization
*Compresses the trained hybrid neural network into an ultra-compact TensorFlow Lite FlatBuffer model using 8-bit dynamic-range quantization.*

| ID | Story | Priority | Est. | Depends on | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B4.1** | As an edge engineer, I want a quantization script converting trained Keras models to TensorFlow Lite with 8-bit post-training dynamic-range quantization, so that model file size is compressed below 150 KB. | High | M | B3.3 | Shrinks weights to INT8 |
| **B4.2** | As an edge engineer, I want an automated validation check verifying that the exported `.tflite` model produces identical output shapes `(1, 2)` and numerically valid probabilities, so that deployment stability is guaranteed. | High | S | B4.1 | Verifies NFR2 (<1 MB target) |

---

## Epic 5: Edge Inference Worker & Local Persistence
*Deploys the single-threaded detection worker, real-time flow stream replay engine, and embedded SQLite database.*

| ID | Story | Priority | Est. | Depends on | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B5.1** | As a system administrator, I want an embedded SQLite database schema initialized with tables for detections, alerts, system metrics, and model registry, so that all runtime data is persisted locally. | High | S | B1.1 | WAL mode enabled |
| **B5.2** | As an edge engineer, I want a decoupled inference engine executing on `ai-edge-litert` within the single-core simulation ceiling, so that incoming 10-flow windows are classified in < 50 ms. | High | M | B1.2, B4.2, B5.1 | Enforces NFR1 and NFR3 |
| **B5.3** | As a SecOps analyst, I want an automated alert generation module that assigns severity levels (LOW, MEDIUM, HIGH, CRITICAL) to malicious classifications, so that security events are prioritized. | High | S | B5.2 | Rules defined in FR7 |
| **B5.4** | As an evaluator, I want a configurable flow stream replay engine streaming benchmark records at adjustable arrival rates, so that real-world network traffic conditions can be simulated without packet capture hardware. | High | M | B2.1, B5.2 | Replays UNSW-NB15 / CICIDS2017 |

---

## Epic 6: Threat Monitoring Dashboard & Operational UI
*Delivers an ultra-lightweight, decoupled web dashboard for real-time visualization of threats, system health, and flow classifications.*

| ID | Story | Priority | Est. | Depends on | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B6.1** | As a SecOps analyst, I want a Flask web application serving a responsive single-page dashboard, so that network threat status is visible in any web browser. | High | M | B5.1 | Port 5000, decoupled from worker |
| **B6.2** | As a SecOps analyst, I want real-time dashboard panels displaying active service status, live classification feeds, and a prioritized alert queue with acknowledgement toggles, so that threats can be monitored and triaged. | High | M | B6.1, B5.3 | REST API JSON endpoints |
| **B6.3** | As a system administrator, I want a system metrics panel on the dashboard graphing CPU %, RAM RSS (MB), and throughput (flows/sec), so that resource compliance against Raspberry Pi ceilings is monitored in real-time. | Medium | S | B6.1, B1.3 | Visualizes psutil metrics |

---

## Epic 7: Empirical Benchmarking & Academic Evaluation
*Generates comprehensive performance metrics, confusion matrices, latency distributions, and thesis documentation artifacts.*

| ID | Story | Priority | Est. | Depends on | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B7.1** | As an academic examiner, I want an automated evaluation script calculating Accuracy, Precision, Attack Recall, Macro-F1, FPR, and generating confusion matrix plots, so that model performance is rigorously validated. | High | M | B3.3, B4.2 | Replicates Chapter 4.9 tables |
| **B7.2** | As an academic examiner, I want a simulated benchmark runner executing 1,000 warm iterations under `taskset -c 0`, recording latency percentiles (mean, median, p95) and peak memory, so that edge feasibility is formally proved. | High | M | B1.2, B5.2 | Validates NFR1 and NFR3 |
