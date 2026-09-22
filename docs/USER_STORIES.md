# User Stories: AI-Powered Network Threat Detection System

**Project:** AI-Powered Network Threat Detection System in Resource-Constrained Environments  
**Target Platform:** Simulated Raspberry Pi-Class Edge Node (Ubuntu Workstation)  
**Standard Format:** `As a <Role>, I want <Capability>, so that <Benefit / Rationale>`  

---

## Persona 1: Network Security Operations Analyst (SecOps)

### US-SEC-01: Real-Time Flow Classification
- **Story:** As a SecOps analyst, I want incoming network flows to be automatically classified as Benign or Malicious in 10-flow sequences, so that anomalous multi-step attack behaviors are identified as they traverse the network perimeter.
- **Value:** Prevents sophisticated multi-packet attack campaigns (e.g., port scans, brute-force bursts, volumetric DDoS) from bypassing perimeter security.
- **Backlog Mapping:** B5.2, B6.2

### US-SEC-02: Prioritized Threat Alerting
- **Story:** As a SecOps analyst, I want malicious detections to trigger prioritized alerts categorized by severity (LOW, MEDIUM, HIGH, CRITICAL), so that I can immediately focus on the most dangerous, high-confidence network intrusions.
- **Value:** Reduces alert fatigue and accelerates incident response during active cyberattacks.
- **Backlog Mapping:** B5.3, B6.2

### US-SEC-03: Alert Acknowledgement & Triage
- **Story:** As a SecOps analyst, I want to review unacknowledged alerts on the web dashboard and mark them as acknowledged with a single click, so that active incident queues accurately reflect remaining unhandled threats.
- **Value:** Streamlines operational workflows and provides accountability for incident resolution.
- **Backlog Mapping:** B6.2

### US-SEC-04: False Positive Rate Minimization
- **Story:** As a SecOps analyst, I want the threat detector to maintain a False Positive Rate below 1.0%, so that legitimate business and campus traffic is not continually flagged as malicious.
- **Value:** Preserves trust in automated security alerts and ensures network availability.
- **Backlog Mapping:** B2.4, B7.1

---

## Persona 2: Resource-Constrained System Administrator (Edge Admin)

### US-SYS-01: Single-Core Execution Pinning
- **Story:** As an edge system administrator, I want to pin the threat detection service to a single logical CPU core using `taskset -c 0`, so that the detector does not starve other critical gateway services of CPU cycles.
- **Value:** Guarantees that edge gateways and single-board computers remain stable and responsive under high network traffic loads.
- **Backlog Mapping:** B1.2, B5.2

### US-SYS-02: Memory Ceiling Compliance
- **Story:** As an edge system administrator, I want the detection service to sustain steady-state memory consumption below 512 MB RSS, so that the service runs safely within low-cost 1GB/2GB edge devices without triggering OOM kernel panics.
- **Value:** Enables deployment on affordable, low-spec hardware without requiring expensive hardware upgrades.
- **Backlog Mapping:** B1.2, B1.3, B5.2

### US-SYS-03: Real-Time Telemetry Dashboard
- **Story:** As an edge system administrator, I want a dedicated resource telemetry panel on the web dashboard displaying rolling CPU %, Memory RSS (MB), and processing throughput (flows/sec), so that I can immediately observe system health and capacity saturation.
- **Value:** Provides proactive visibility into performance bottlenecks before service degradation occurs.
- **Backlog Mapping:** B1.3, B6.3

### US-SYS-04: Zero-Configuration Embedded Persistence
- **Story:** As an edge system administrator, I want all detection events and metrics stored in a local, serverless SQLite database, so that no heavy database daemons (like PostgreSQL or MySQL) are required to run on the edge device.
- **Value:** Simplifies system administration, eliminates background database memory consumption, and allows portable backup of audit logs.
- **Backlog Mapping:** B5.1

### US-SYS-05: Decoupled Web and Inference Processes
- **Story:** As an edge system administrator, I want the web dashboard to run as a separate process from the edge inference worker, so that heavy browser web traffic cannot degrade detection latency or cause dropped network flows.
- **Value:** Isolates the mission-critical detection loop from the user interface presentation layer.
- **Backlog Mapping:** B5.2, B6.1, ADR-001

---

## Persona 3: Cybersecurity Researcher & Academic Examiner (Student)

### US-RES-01: Leakage-Free Dataset Partitioning
- **Story:** As an academic researcher, I want preprocessing transformations (StandardScaler, PCA) to fit strictly on the training partition and transform validation/test partitions independently, so that empirical evaluation is strictly insulated from data leakage.
- **Value:** Ensures academic integrity and guarantees that published evaluation metrics reflect genuine generalization capability.
- **Backlog Mapping:** B2.2, B2.3

### US-RES-02: SMOTE + Tomek Links Class Balancing
- **Story:** As an academic researcher, I want SMOTE oversampling and Tomek Links under-sampling applied exclusively to the training partition, so that minority attack classes are learnable by neural networks without distorting test split class distributions.
- **Value:** Resolves severe benchmark class imbalance while preserving valid academic testing protocols.
- **Backlog Mapping:** B2.4

### US-RES-03: Comparative Baseline Benchmarking
- **Story:** As an academic examiner, I want to evaluate Decision Tree, Random Forest, SVM, CNN-only, and LSTM-only baselines alongside the hybrid CNN–LSTM model on the identical test split, so that the value of the hybrid architecture is scientifically established.
- **Value:** Validates the research contribution against established classical and deep learning baselines as required by the University of Uyo dissertation.
- **Backlog Mapping:** B3.1, B3.2, B7.1

### US-RES-04: Transparent Workstation Simulation Reporting
- **Story:** As an academic examiner, I want all edge benchmark logs and reports to explicitly identify the host execution environment as an x86_64 workstation simulation with pinned CPU affinity, so that experimental claims accurately distinguish simulated edge performance from physical Raspberry Pi ARM measurements.
- **Value:** Maintains academic honesty, avoids false claims of physical ARM hardware testing, and documents methodology transparently.
- **Backlog Mapping:** B1.2, B7.2, ADR-001

### US-RES-05: Reproducible Evaluation Figures and Tables
- **Story:** As an academic researcher, I want automated generation of confusion matrices, training loss curves, and metric summary CSVs, so that all tables and figures in the dissertation (Chapters 3 and 4) can be verified and regenerated on demand.
- **Value:** Enables peer verification and seamless reproduction of dissertation results.
- **Backlog Mapping:** B7.1

---

## Persona 4: Edge Machine Learning Engineer

### US-MLE-01: Hybrid CNN–LSTM Neural Architecture
- **Story:** As an ML engineer, I want to construct a hybrid CNN–LSTM neural network taking 3D inputs `(Batch, 10, Features)`, where Conv1D extracts intra-flow spatial features and LSTM models inter-flow temporal dependencies, so that high detection accuracy is achieved across sequential traffic flows.
- **Value:** Combines feature extraction and sequence modeling in a single end-to-end differentiable neural pipeline.
- **Backlog Mapping:** B3.3

### US-MLE-02: Post-Training 8-Bit Dynamic Range Quantization
- **Story:** As an ML engineer, I want to quantize the trained hybrid Keras model into an 8-bit dynamic-range TensorFlow Lite FlatBuffer, so that model weights are compressed by ~75% and model file size is reduced to ~100 KB without measurable loss of detection accuracy.
- **Value:** Reduces memory footprint and enables fast CPU-only integer arithmetic on edge processors.
- **Backlog Mapping:** B4.1, B4.2

### US-MLE-03: Lightweight Interpreter Execution via LiteRT
- **Story:** As an ML engineer, I want to execute inference using the standalone `ai-edge-litert` interpreter rather than the monolithic TensorFlow package, so that memory RSS drops from ~687 MB to < 50 MB.
- **Value:** Satisfies NFR3 (< 512 MB memory budget) and avoids massive Python package dependencies on the deployment host.
- **Backlog Mapping:** B5.2

### US-MLE-04: Automated Pipeline Co-Versioning
- **Story:** As an ML engineer, I want fitted scalers, PCA components, label maps, and `.tflite` model files to be saved together with a cryptographic manifest, so that runtime inference never encounters mismatched feature ordering or incompatible tensor shapes.
- **Value:** Prevents catastrophic runtime dimension mismatch errors and ensures robust deployment governance.
- **Backlog Mapping:** B2.2, B4.2, NFR7
