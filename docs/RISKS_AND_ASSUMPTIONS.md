# Risks, Assumptions, and Boundary Log
## AI-Powered Network Threat Detection System

**Project:** AI-Powered Network Threat Detection System in Resource-Constrained Environments  
**Author:** Imeh Grace Mfon (21/SC/CO/1117)  
**Academic Context:** Department of Computer Science, University of Uyo, Nigeria  
**Cross-Reference:** Chapter 5.5 (Limitations) & Chapter 5.6 (Recommendations) of B.Sc. Project Report  

---

## 1. Project Risk Register

| Risk ID | Risk Description | Severity | Probability | Impact Area | Mitigation Strategy & Safeguard | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | **Simulation Fidelity Disparity:** Running on an Intel Core i7 x86_64 workstation yields faster clock speeds and wider vector instructions than an ARM Cortex-A72 on a Raspberry Pi. | High | High | Academic Integrity, Latency Claims | Pin to single core (`taskset -c 0`), cap memory at 2GB, restrict to single thread, and **transparently label all findings as "Workstation Simulation Results"**. Physical power draw is strictly recorded as **NOT MEASURED**. | Mitigated |
| **R2** | **Framework Memory Bloat:** Full `import tensorflow` consumes ~687 MB RSS upon initialization, violating the 512 MB sustained service memory ceiling (NFR3). | High | High | Memory Compliance | Execute edge inference exclusively via the standalone **`ai-edge-litert`** runtime, which consumes only **49.93 MB RSS**. | Mitigated |
| **R3** | **Synthetic Data Leakage:** Applying SMOTE, Tomek Links, or scalers across the entire dataset before train/test splitting leaks test statistics into training. | Critical | Medium | Academic Validity | Enforce strict pipeline boundaries: `StandardScaler` and `PCA` fit strictly on $X_{train}$; SMOTE + Tomek Links are applied strictly to $(X_{train}, y_{train})$. Validation and test splits remain completely untouched. | Mitigated |
| **R4** | **Class Imbalance Metric Distortion:** Achieving >99% accuracy by trivial majority-class prediction while missing rare attacks. | High | High | Detection Quality | Never evaluate on accuracy alone. Prioritize **Macro-averaged F1 score**, **Attack Recall (True Positive Rate)**, and **False Positive Rate (FPR)** alongside confusion matrices. | Mitigated |
| **R5** | **Edge Memory Leaks in Long-Running Workers:** Python garbage collection delays or accumulating event queues causing gradual memory exhaustion. | Medium | Medium | Service Stability | Enforce explicit garbage collection passes, bounded in-memory sliding windows (fixed 10 records), and direct-to-SQLite streaming. | Mitigated |
| **R6** | **SQLite Write Lock Contention:** High-frequency inference writes colliding with browser dashboard query reads. | Medium | High | System Concurrency | Enable SQLite **Write-Ahead Logging (WAL mode)**, set `PRAGMA synchronous = NORMAL;`, and configure a 5,000 ms busy timeout. | Mitigated |
| **R7** | **Pruning / QAT Tool Incompatibility:** TensorFlow Model Optimization Toolkit (tfmot) incompatibilities with modern Keras/TF 2.16+ model formats. | Medium | High | Model Optimization | Explicitly document weight pruning and Quantization-Aware Training (QAT) as **unexecuted optimization extensions** in the thesis; rely on verified post-training dynamic range quantization. | Managed |

---

## 2. Formal Assumptions Log

### A1: Platform & Execution Scope
- **Assumption:** The official software evaluation is performed on an **Ubuntu 22.04 LTS x86_64 workstation** under controlled resource simulation boundaries rather than a physical Raspberry Pi single-board computer.
- **Rationale:** Physical Raspberry Pi hardware was unavailable. The simulation accurately bounds single-core CPU scheduling, memory limits, and single-threaded TFLite execution.

### A2: Power Draw Out of Scope
- **Assumption:** Electrical power draw (in Watts) and battery depletion rates are **unmeasured** in this software implementation.
- **Rationale:** Workstation motherboard power rails and power supplies cannot accurately simulate the 5V / 3A DC input characteristics of a single-board ARM computer. All documentation marks power draw as `NOT MEASURED`.

### A3: Data Ingestion Mechanism
- **Assumption:** Network traffic is ingested either from standardized benchmark CSV archives (UNSW-NB15, 200k CICIDS2017 subset) or replayed through a software stream generator.
- **Rationale:** Capturing live raw network packets via promiscuous interfaces (`pcap`) requires root access, introduces heavy OS packet-dissection overhead, and lacks ground-truth attack labels necessary for scientific validation.

### A4: Stationary Feature Distribution
- **Assumption:** Benchmark flow features (duration, packet counts, protocol byte rates) extracted in UNSW-NB15 and CICIDS2017 represent realistic traffic patterns for validating spatial-temporal intrusion detection.
- **Rationale:** Both datasets are widely accepted international benchmarks used in academic NIDS research.

### A5: Independent Process Coupling
- **Assumption:** The web monitoring dashboard and the edge detection engine can be cleanly decoupled via an embedded SQLite database without requiring distributed message brokers (Kafka, Redis, RabbitMQ).
- **Rationale:** In a resource-constrained 512 MB environment, external message brokers waste excessive RAM. SQLite WAL mode provides ample throughput for single-node edge operation.

---

## 3. Academic Defense Notes & Examiner Guidance

When defending the implementation before an academic panel or project supervisor (Mr. U. J. Ntia):

1. **When asked: *"Why did you test on Ubuntu instead of a physical Raspberry Pi?"***
   - *Answer:* "Physical Raspberry Pi hardware was unavailable during this phase. To maintain research momentum and scientific rigor, we designed a five-pillar OS-level simulation: pinning execution to a single CPU core via `taskset -c 0`, enforcing a 2GB cgroup memory ceiling, disabling GPU acceleration, and running single-threaded `ai-edge-litert`. This proved that our quantized model (100 KB) and runtime (49.9 MB RSS) operate well within the Raspberry Pi's 512 MB budget. Physical ARM testing and power measurements are documented as the immediate next hardware phase."

2. **When asked: *"Why is your model latency so low (< 0.01 ms)?"***
   - *Answer:* "The measured latency of 0.0095 ms per 10-flow sequence reflects the efficiency of our 8-bit dynamic range quantization on a single x86 core. We transparently report this as workstation simulation performance. On an actual ARM Cortex-A72, clock frequencies and cache sizes are lower, so we anticipate physical latency between 5 ms and 25 ms—which remains comfortably below our 50 ms real-time requirement."

3. **When asked: *"Why did you use SMOTE and Tomek Links together?"***
   - *Answer:* "Network intrusion datasets suffer from extreme class imbalance. SMOTE alone synthesizes minority attack samples, but can inadvertently blur the boundary between attack and benign clusters. Tomek Links identifies and removes these borderline, overlapping pairs, yielding clean, well-defined decision boundaries for our neural network. Crucially, we applied this exclusively to the training split to prevent synthetic data leakage into our test evaluation."
