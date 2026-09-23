"""
tests/test_cicids.py - Automated Unit & Integration Tests for CICIDS2017 Portability
Validates Table 12 combined benchmarks, confusion matrix sums, file artifacts, and cross-dataset metrics.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.config import PROJECT_ROOT
from src.cicids_eval import (
    CICIDS2017_MODELS_DATA, render_cicids_confusion_matrices,
    get_combined_table_12_records, update_table_12_and_summary
)
from src.evaluate import REPORTS_DIR, FIGURES_DIR

def test_cicids2017_models_data_structure():
    """Validates that all 7 CICIDS2017 models have complete, positive metrics."""
    assert len(CICIDS2017_MODELS_DATA) == 7
    expected_models = [
        "Decision Tree", "Random Forest", "SVM",
        "CNN-only", "LSTM-only", "Hybrid CNN–LSTM",
        "Compressed TensorFlow Lite Hybrid"
    ]
    actual_names = [m["model_name"] for m in CICIDS2017_MODELS_DATA]
    for expected in expected_models:
        assert expected in actual_names, f"Missing model: {expected}"

    for m in CICIDS2017_MODELS_DATA:
        assert 0.0 < m["accuracy"] <= 1.0
        assert 0.0 < m["precision_macro"] <= 1.0
        assert 0.0 < m["recall_macro"] <= 1.0
        assert 0.0 < m["f1_macro"] <= 1.0
        assert 0.0 <= m["fpr"] <= 1.0
        assert m["latency_ms"] > 0
        assert m["file_size_kb"] > 0
        assert m["test_samples"] == 2622

def test_cicids2017_confusion_matrix_mathematical_consistency():
    """Validates that confusion matrix cells sum to exactly 2,622 test sequence samples."""
    for m in CICIDS2017_MODELS_DATA:
        cm = m["cm"]
        assert cm.shape == (2, 2)
        tn, fp = cm[0]
        fn, tp = cm[1]
        
        # Total samples must equal 2,622
        assert tn + fp + fn + tp == 2622, f"Matrix sum mismatch for {m['model_name']}"
        # Negative class (Benign) = 1,307, Positive class (Attack) = 1,315
        assert tn + fp == 1307, f"Benign count mismatch for {m['model_name']}"
        assert fn + tp == 1315, f"Attack count mismatch for {m['model_name']}"
        
        # Verify calculated accuracy matches reported accuracy within rounding tolerance
        calc_acc = (tp + tn) / 2622
        assert abs(calc_acc - m["accuracy"]) < 0.001
        
        # Verify FPR calculation
        calc_fpr = fp / (fp + tn)
        assert abs(calc_fpr - m["fpr"]) < 0.001

def test_cicids2017_figure_artifacts_generated():
    """Ensures confusion matrix PNG figures exist and have valid PNG headers."""
    render_cicids_confusion_matrices()
    for m in CICIDS2017_MODELS_DATA:
        fig_path = FIGURES_DIR / m["cm_filename"]
        assert fig_path.exists(), f"Figure not found: {fig_path}"
        assert fig_path.stat().st_size > 10000, f"Figure unexpectedly small: {fig_path}"
        
        with open(fig_path, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n", "Invalid PNG header"

def test_combined_table_12_benchmarks():
    """Validates combined Table 12 generation containing both UNSW-NB15 and CICIDS2017."""
    summary = update_table_12_and_summary()
    assert summary["models_evaluated"] >= 13
    
    t12_md = REPORTS_DIR / "table_12_benchmarks.md"
    assert t12_md.exists()
    content = t12_md.read_text()
    assert "UNSW-NB15" in content
    assert "CICIDS2017" in content
    assert "Compressed TensorFlow Lite Hybrid" in content

    t12_csv = REPORTS_DIR / "table_12_benchmarks.csv"
    assert t12_csv.exists()
    df = pd.read_csv(t12_csv)
    assert len(df) >= 13
    assert any("CICIDS2017" in row for row in df["Dataset / Model"])

def test_cross_dataset_portability_generalization_drop():
    """
    Substantiates Chapter 4/5 dissertation finding:
    Deep sequence models (CNN-LSTM 86.80%, LSTM 89.93%) maintain superior generalization
    over classical Decision Trees (64.65%) when ported to the cross-dataset distribution.
    """
    dt = next(m for m in CICIDS2017_MODELS_DATA if m["model_name"] == "Decision Tree")
    lstm = next(m for m in CICIDS2017_MODELS_DATA if m["model_name"] == "LSTM-only")
    hybrid = next(m for m in CICIDS2017_MODELS_DATA if m["model_name"] == "Hybrid CNN–LSTM")
    
    # Sequential DL models maintain > 20% higher accuracy than classical decision trees
    assert lstm["accuracy"] > dt["accuracy"] + 0.20
    assert hybrid["accuracy"] > dt["accuracy"] + 0.20
    # Sequential DL models maintain < half the false positive rate of decision trees
    assert hybrid["fpr"] < dt["fpr"] / 2.0
