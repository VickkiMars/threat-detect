# Sprint 3 Backlog — AI-Powered Network Threat Detection System

**Sprint Identifier:** SPRINT-03-OPERATIONALIZATION  
**Sprint Goal:** Achieve full end-to-end operationalization, live continuous flow stream ingestion under single-core simulation, real-time threat triage and alert lifecycle verification in the dashboard, automated test suite expansion, cross-dataset portability evaluation (CICIDS2017), and compilation of the definitive dissertation acceptance defense package.  
**Timebox:** 1 committed iteration slice (approx. 5 working days)  
**Pulled From:** `docs/PRODUCT_BACKLOG.md` items: B5.2, B5.3, B5.4, B6.2, B6.3, B7.1, B7.2, DoD  

---

## 1. Committed Engineering Tasks

| Task ID | User Story | Engineering Task Description | Est. | Depends on | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **T3.1** | B5.4, B5.2 | Implement end-to-end continuous flow replay stream simulator and wire `stream_simulator.py` to feed held-out test flows directly to `inference_worker.py` under `taskset -c 0`. | M | B1.2, B5.2 | Completed |
| **T3.2** | B5.3, B6.2 | Verify real-time threat alert lifecycle: validate severity thresholding (LOW, MEDIUM, HIGH, CRITICAL), instant operator acknowledgment via dashboard REST API, and alert queue state synchronization. | S | T3.1 | Completed |
| **T3.3** | DoD | Expand automated test suite coverage: implement dedicated test modules for balancing (`test_balancing.py`), model architectures (`test_models.py`), quantization (`test_quantization.py`), and evaluation (`test_evaluation.py`). | M | T3.1 | Completed |
| **T3.4** | B2.1, B7.1 | Evaluate cross-dataset portability on CICIDS2017 benchmark: assess zero-shot / fine-tuned generalization on the 200,000 class-capped subset to substantiate Chapter 4/5 portability findings. | M | T3.1 | Completed |
| **T3.5** | DoD | Compile comprehensive Dissertation Acceptance & Defense Package (`reports/system_acceptance_report.md`): consolidate Tables 11, 12, 13, NFR1–NFR7 verification matrix, and defense demonstration runbook. | S | T3.2, T3.3 | Completed |

---

## 2. Sprint Definition of Done

This sprint increment is considered **Shippable and Complete** when:
1. `src/inference_worker.py` sustains streaming inference against `data/UNSW_NB15_testing-set.csv` or `sample_flows.csv` with zero dropped windows and mean latency $\le 50.0\text{ ms}$.
2. Real-time alerts populate the dashboard without page refreshes, and clicking "Acknowledge" updates the database immediately.
3. System telemetry (CPU %, Memory RSS MB, Throughput FPS) updates continuously on the live dashboard.
4. Active model is registered in `model_registry` as the 8-bit dynamic-range quantized FlatBuffer (`v2.0.0-hybrid-quantized`, 56.89 KB).
5. Automated test suite expands to cover class balancing, DL models, quantization, and evaluation with 100% passing tests.
6. The consolidated Dissertation Acceptance Report (`reports/system_acceptance_report.md`) is finalized for thesis submission and oral defense.

---

## 3. Burn-down and Daily Sequencing

```
Day 1: T3.1 (Stream Replay & Worker Wiring) -> T3.2 (Alert Lifecycle & Triage Verification)
Day 2: T3.3 (Test Suite Expansion: Balancing, Models, Quantization, Evaluation)
Day 3: T3.4 (CICIDS2017 Cross-Dataset Evaluation)
Day 4: T3.5 (System Acceptance Report & Dissertation Defense Documentation)
Day 5: Full System Integration Verification & Final Sprint Review
```
