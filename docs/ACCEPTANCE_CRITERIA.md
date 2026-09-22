# Acceptance Criteria & Definition of Done (DoD)
## AI-Powered Network Threat Detection System in Resource-Constrained Environments

**Project:** AI-Powered Network Threat Detection System in Resource-Constrained Environments  
**Target Platform:** Simulated Raspberry Pi-Class Edge Node (Ubuntu Workstation)  
**Standard Format:** Given-When-Then (Gherkin Syntax)  

---

## 1. System-Wide Definition of Done (DoD)

A backlog item, feature increment, or model artifact is considered **Done** if and only if all of the following criteria are verified:

1. **Code & Architecture Quality:**
   - [ ] Implementation is written in Python 3.10+ following PEP 8 conventions.
   - [ ] All functions and classes have explicit docstrings, parameter types, and return types.
   - [ ] Zero hardcoded file paths; all directories resolve relative to project root or environment variables.
2. **Resource & Performance Ceilings (NFR Compliance):**
   - [ ] **NFR1:** Mean inference latency is <= 50.0 ms per 10-flow window under `taskset -c 0` single-core execution.
   - [ ] **NFR2:** Exported `.tflite` model binary is <= 1.0 MB (target <= 150 KB).
   - [ ] **NFR3:** Steady-state Resident Set Size (RSS) memory consumption is <= 512 MB when executed with `ai-edge-litert`.
   - [ ] **NFR5:** System runs 100% CPU-only (`CUDA_VISIBLE_DEVICES=-1`); zero GPU calls.
3. **Data Integrity & Academic Rigor:**
   - [ ] Preprocessing transformers (`StandardScaler`, `PCA`) are fitted exclusively on training data ($X_{train}$); zero data leakage to validation or test splits.
   - [ ] SMOTE and Tomek Links are applied strictly to the training split.
   - [ ] PCA preserves >= 95% cumulative explained variance.
4. **Resilience & Testing:**
   - [ ] Unit tests pass for preprocessing, model conversion, SQLite operations, and REST endpoints.
   - [ ] Continuous flow replay does not crash or trigger unhandled Python exceptions.
   - [ ] Database access uses WAL mode and parameterized queries to prevent database locking and SQL injection.
5. **Documentation & Traceability:**
   - [ ] Every result, metric, or figure matches the published findings in the University of Uyo dissertation.
   - [ ] All simulation logs explicitly record the host processor, operating system version, memory limit, and CPU affinity.

---

## 2. Granular Acceptance Criteria by Backlog Item

### Epic 1: Simulation Harness & Edge Environment Setup

#### AC-B1.1: Environment Setup Script
```gherkin
Scenario: Automated virtual environment configuration
  Given a clean Ubuntu Linux workstation with Python 3.10+ installed
  When the administrator executes "bash setup_env.sh"
  Then a virtual environment ".venv" shall be created
  And all required packages (ai-edge-litert, scikit-learn, imbalanced-learn, flask, psutil, pandas, numpy) shall be installed without conflicts
  And verifying imports shall exit with return code 0.
```

#### AC-B1.2: Workstation Simulation Resource Containment
```gherkin
Scenario: Restricting process execution to single logical CPU and memory budget
  Given an activated virtual environment
  When the user executes the simulation harness "bash run_simulated.sh --cmd <command>"
  Then the process shall be pinned to logical CPU 0 via taskset
  And CUDA GPU access shall be disabled via CUDA_VISIBLE_DEVICES=-1
  And TFLite intra-op and inter-op threads shall be restricted to 1
  And the memory ceiling of 2 GB shall be enforced via systemd-run or ulimit.
```

#### AC-B1.3: Hardware Resource Instrumentation
```gherkin
Scenario: Continuous sampling of system telemetry via psutil
  Given the detection service is actively classifying flows
  When the telemetry sampler runs at its configured interval (1 second)
  Then it shall record CPU utilization percentage, process RSS memory in megabytes, and flow throughput
  And write the timestamped sample to the "system_metrics" table in SQLite
  And the telemetry overhead shall consume less than 1% CPU utilization.
```

---

### Epic 2: Data Acquisition & Preprocessing Pipeline

#### AC-B2.1: Dataset Ingestion & Schema Validation
```gherkin
Scenario: Parsing raw benchmark CSV files
  Given valid CSV files for UNSW-NB15 or the 200k CICIDS2017 subset
  When the data loader ingests the files
  Then all column headers shall match the canonical schema
  And infinite values (inf, -inf) shall be replaced with column maximums
  And null values shall be imputed with column medians without dropping valid rows.
```

#### AC-B2.2: Data Leakage Prevention during Scaling
```gherkin
Scenario: Fitting scaler exclusively on training partition
  Given an 70/15/15 stratified train/val/test split
  When the StandardScaler is executed
  Then scaler.fit() shall be invoked solely on X_train
  And X_val, X_test, and streaming flow records shall only be transformed via scaler.transform()
  And the fitted scaler shall be persisted to "models/scaler.joblib".
```

#### AC-B2.3: PCA Dimensionality Reduction
```gherkin
Scenario: Compressing features with >= 95% variance retention
  Given scaled training feature matrix X_train_scaled
  When PCA is fitted with variance threshold 0.95
  Then the cumulative explained variance ratio shall be >= 0.9500
  And for CICIDS2017, the retained component count shall be exactly 22
  And the fitted PCA transformer shall be persisted to "models/pca.joblib".
```

#### AC-B2.4: Training Split Class Balancing (SMOTE + Tomek Links)
```gherkin
Scenario: Applying SMOTE and Tomek Links to training data only
  Given an imbalanced training partition (X_train_pca, y_train)
  When the resampling pipeline is executed
  Then SMOTE shall oversample minority attack instances
  And Tomek Links shall remove ambiguous boundary pairs
  And the validation and test sets (y_val, y_test) shall remain untouched with their original ground-truth class distributions.
```

#### AC-B2.5: Temporal Sequence Windowing (10-Flow Windows)
```gherkin
Scenario: Transforming tabular records into 3D sequential tensors
  Given an ordered sequence of processed flow feature vectors of length N
  When the window formatter executes with window size W=10 and stride S=10
  Then the resulting tensor shall have shape (N // 10, 10, num_features)
  And sequence labels shall represent the presence of an attack within the window (or window terminal state).
```

---

### Epic 3: Baseline & Hybrid Model Training Pipeline

#### AC-B3.1: Classical Baseline Training
```gherkin
Scenario: Training Decision Tree, Random Forest, and SVM baselines
  Given flattened sequence matrices of shape (N, 10 * num_features)
  When the baseline training script is executed
  Then models for Decision Tree, Random Forest (n=100), and Linear/RBF SVM shall be fitted
  And each model's accuracy, precision, recall, and F1 score shall be logged to the evaluation registry.
```

#### AC-B3.3: Hybrid CNN–LSTM Model Training
```gherkin
Scenario: Training the proposed hybrid deep learning architecture
  Given 3D training sequences of shape (Batch, 10, num_features)
  When model.fit() is executed on the hybrid architecture (Conv1D -> MaxPool -> Dropout -> LSTM -> Dense)
  Then training loss and validation loss shall be logged per epoch
  And early stopping shall halt training if validation loss does not improve for 5 consecutive epochs
  And the best weights shall be saved to "models/hybrid_best.keras".
```

---

### Epic 4: Model Compression & Edge Quantization

#### AC-B4.1: Post-Training Dynamic-Range Quantization
```gherkin
Scenario: Converting trained Keras model to 8-bit dynamic-range TFLite
  Given a validated Keras model file "models/hybrid_best.keras"
  When the TFLite converter is executed with optimizations=[tf.lite.Optimize.DEFAULT]
  Then the exported file "models/hybrid_model.tflite" shall be created
  And the binary file size shall be <= 150 KB (passing NFR2: <= 1.0 MB).
```

#### AC-B4.2: TFLite Signature & Numerical Validation
```gherkin
Scenario: Verifying TFLite model input/output signatures and numerical consistency
  Given the exported "models/hybrid_model.tflite"
  When the interpreter allocates tensors
  Then input tensor shape shall be (1, 10, num_features) with dtype float32
  And output tensor shape shall be (1, 2) with dtype float32
  And evaluated predictions on test sequences shall yield valid probability distributions summing to 1.0 +/- 1e-5.
```

---

### Epic 5: Edge Inference Worker & Local Persistence

#### AC-B5.1: SQLite Database Initialization & WAL Mode
```gherkin
Scenario: Initializing local SQLite database
  Given the database path "data/threat_detection.db"
  When the initialization script executes
  Then tables "detection_log", "alert", "system_metrics", and "model_registry" shall be created
  And PRAGMA journal_mode=WAL shall be enabled
  And indexed lookups on timestamp and severity shall succeed without syntax errors.
```

#### AC-B5.2: Real-Time Single-Core Inference Performance
```gherkin
Scenario: Verifying inference latency under simulated single-core ceiling
  Given the TFLite model loaded in "ai-edge-litert" under "taskset -c 0"
  When 1,000 warm 10-flow sequence inferences are executed
  Then the mean latency per sequence shall be <= 50.0 ms (passing NFR1)
  And the 95th-percentile (p95) latency shall be <= 50.0 ms
  And the peak Resident Set Size (RSS) shall remain <= 512 MB (passing NFR3).
```

#### AC-B5.3: Threat Alert Generation & Severity Assignment
```gherkin
Scenario: Raising classified threat alerts based on attack probability
  Given an incoming sequence classified as Malicious (attack probability P)
  When P >= 0.50
  Then an alert record shall be inserted into the "alert" table
  And if P is between 0.50 and 0.69, severity shall be "LOW"
  And if P is between 0.70 and 0.84, severity shall be "MEDIUM"
  And if P is between 0.85 and 0.94, severity shall be "HIGH"
  And if P is >= 0.95, severity shall be "CRITICAL".
```

#### AC-B5.4: Network Flow Stream Replay Simulation
```gherkin
Scenario: Streaming benchmark flows into the detection worker
  Given a preprocessed evaluation flow dataset
  When the stream replay engine is started with rate 50 flows/sec
  Then flow records shall be buffered and fed into the detector in windows of 10
  And the arrival rate shall maintain within +/- 10% of the target frequency
  And the simulation can be paused, resumed, or terminated cleanly via SIGINT.
```

---

### Epic 6: Threat Monitoring Dashboard & Operational UI

#### AC-B6.1: Web Dashboard Availability & Separation
```gherkin
Scenario: Accessing the decoupled monitoring dashboard
  Given the Flask web application is started on port 5000
  When a user opens "http://localhost:5000" in a web browser
  Then the page shall load with HTTP status 200 within 1.0 second
  And querying the dashboard shall not affect inference worker latency.
```

#### AC-B6.2: Real-Time Alert Triage & Acknowledgement
```gherkin
Scenario: Acknowledging an active security alert
  Given an unacknowledged alert with ID 42 displayed on the dashboard
  When the operator clicks the "Acknowledge" button
  Then a POST request shall be sent to "/api/alerts/42/acknowledge"
  And the database column "acknowledged" shall update from 0 to 1
  And the alert shall visually transition to acknowledged state in the UI.
```

#### AC-B6.3: System Telemetry Monitoring
```gherkin
Scenario: Viewing real-time CPU and memory graphs
  Given active flow processing in the background
  When the dashboard polls "/api/metrics/system"
  Then rolling CPU %, RAM RSS (MB), and throughput shall update on the UI charts
  And if RSS exceeds 450 MB, a visual amber warning indicator shall be triggered.
```

---

### Epic 7: Empirical Benchmarking & Academic Evaluation

#### AC-B7.1: Held-Out Test Evaluation & Metric Reproduction
```gherkin
Scenario: Evaluating test partition and reproducing dissertation metrics
  Given the saved UNSW-NB15 test partition of 3,865 sequences
  When the evaluation script "src/evaluate.py" is run against the hybrid model
  Then test accuracy shall be >= 0.9500 (target ~0.9984)
  And attack recall shall be >= 0.9500 (target ~0.9988)
  And False Positive Rate shall be <= 0.0100 (target ~0.0023)
  And Macro-averaged F1 shall be >= 0.9500 (target ~0.9982)
  And a confusion matrix PNG shall be saved to "reports/figures/".
```

#### AC-B7.2: Academic Simulation Audit Compliance
```gherkin
Scenario: Generating formal benchmark audit report
  Given completion of the 1,000-iteration benchmark under taskset -c 0
  When the benchmark report is compiled
  Then it shall explicitly log: Host Processor, CPU Architecture (x86_64), Core Pinning (CPU 0), Memory Ceiling (2GB / 512MB), Python Version, LiteRT Version, and Power Draw = "NOT MEASURED"
  And it shall contain a statement declaring: "Measurements represent simulated workstation performance and not physical Raspberry Pi ARM performance."
```
