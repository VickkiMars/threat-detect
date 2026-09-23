# Sprint 2 Backlog — AI-Powered Network Threat Detection System

**Sprint Identifier:** SPRINT-02-ML-BENCHMARK  
**Sprint Goal:** Implement the full machine learning training and comparative evaluation pipeline—incorporating training-only SMOTE + Tomek Links class balancing, training pipelines for classical baselines (Decision Tree, Random Forest, SVM), deep learning baselines (CNN-only, LSTM-only), and the proposed hybrid CNN–LSTM architecture; quantize and serialize the trained hybrid model; generate dissertation-aligned empirical evaluation tables and confusion matrices (reproducing Table 11 & Table 12); and integrate an interactive Model Benchmark & Evaluation view into the monitoring dashboard.  
**Timebox:** 1 committed iteration slice (approx. 5 working days)  
**Pulled From:** `docs/PRODUCT_BACKLOG.md` items: B2.4, B3.1, B3.2, B3.3, B3.4, B4.1, B4.2, B7.1  

---

## 1. Committed Engineering Tasks

| Task ID | User Story | Engineering Task Description | Est. | Depends on | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **T2.1** | B2.4 | Implement training-only class balancing module (`src/balancing.py`) utilizing SMOTE + Tomek Links (`imbalanced-learn`) ensuring strict leakage prevention. | S | — | Ready |
| **T2.2** | B3.1 | Implement classical baseline classifiers module (`src/models/classical.py`) supporting Decision Tree, Random Forest (100 estimators), and Support Vector Machine (RBF kernel). | M | T2.1 | Ready |
| **T2.3** | B3.2, B3.3, B3.4 | Implement deep learning baseline and hybrid architectures (`src/models/deep_learning.py`): CNN-only, LSTM-only, and Hybrid CNN–LSTM with Adam optimizer, binary cross-entropy, and early stopping. | L | T2.1 | Ready |
| **T2.4** | B3.1–B3.4 | Implement end-to-end training orchestrator (`src/train.py`) executing 70/15/15 stratified partitioning, training-only preprocessing/balancing, model training, and checkpointing. | M | T2.2, T2.3 | Ready |
| **T2.5** | B4.1, B4.2 | Implement model quantization and FlatBuffer serialization module (`src/quantization.py`) converting hybrid weights to 8-bit dynamic-range `.tflite` model with size <= 150 KB. | M | T2.4 | Ready |
| **T2.6** | B7.1 | Implement academic evaluation and dissertation replication suite (`src/evaluate.py`) computing Accuracy, Precision, Recall, Macro-F1, FPR, parameter counts, and confusion matrices (reproducing Table 11 & Table 12). | M | T2.4, T2.5 | Ready |
| **T2.7** | B6.2, B7.1 | Enhance Flask dashboard (`src/dashboard/app.py`, `templates/index.html`, `static/js/dashboard.js`) with `GET /api/benchmarks` and an interactive Academic Evaluation & Model Comparison UI panel. | M | T2.6 | Ready |
| **T2.8** | DoD | Implement comprehensive automated test suites (`tests/test_balancing.py`, `tests/test_models.py`, `tests/test_quantization.py`, `tests/test_evaluation.py`). | M | All | Ready |

---

## 2. Sprint Definition of Done

This sprint increment is considered **Shippable and Complete** when:
1. `src/train.py` executes cleanly on the simulated edge environment, training all 6 model architectures without data leakage.
2. SMOTE + Tomek Links is verified to balance training distributions while leaving validation and test splits pristine.
3. Hybrid CNN–LSTM weights are successfully converted to an 8-bit dynamic-range `.tflite` model with verified size $\le 150\text{ KB}$ and verified input/output tensor shapes `(1, 10, 22)` and `(1, 2)`.
4. `src/evaluate.py` outputs dissertation-aligned benchmark tables (`reports/table_11_quantization.md` and `reports/table_12_benchmarks.md`) and high-resolution confusion matrix figures (`reports/figures/`).
5. All models achieve verified performance against NFR4 (Accuracy $> 95\%$, Attack Recall $\ge 95\%$, FPR $\le 1.0\%$).
6. The dashboard at `http://localhost:5000` renders the new Academic Benchmarks panel and exposes `GET /api/benchmarks`.
7. 100% of unit tests pass in `pytest` with zero failures and zero regressions.

---

## 3. Burn-down and Daily Sequencing

```
Day 1: T2.1 (SMOTE-Tomek Balancing) -> T2.2 (Classical Baselines)
Day 2: T2.3 (Deep Learning Architectures: CNN, LSTM, Hybrid) -> T2.4 (Training Pipeline)
Day 3: T2.5 (Quantization & FlatBuffer Serialization) -> T2.6 (Evaluation Suite & Figures)
Day 4: T2.7 (Dashboard Benchmark API & UI Integration)
Day 5: T2.8 (Automated Test Suites) -> Sprint Review & DoD Validation
```
