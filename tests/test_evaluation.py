"""
tests/test_evaluation.py - Unit & Academic Metric Verification Tests
Validates calculation formulas for Accuracy, Recall, FPR, Macro-F1,
DataFrame-to-markdown table formatting, and confusion matrix figure plotting.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.evaluate import (
    compute_intrusion_metrics,
    df_to_markdown,
    plot_confusion_matrix
)

def test_compute_intrusion_metrics_perfect_classification():
    """Verifies metrics computation under ideal 100% classification."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    y_pred = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    
    metrics = compute_intrusion_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["tp"] == 3
    assert metrics["tn"] == 3
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0

def test_compute_intrusion_metrics_known_confusion():
    """Verifies metrics computation against precisely calibrated confusion matrix counts."""
    # TN: 80, FP: 20 -> Benign total: 100
    # FN: 10, TP: 90 -> Attack total: 100
    y_true = np.array([0] * 100 + [1] * 100, dtype=np.int32)
    y_pred = np.array([0] * 80 + [1] * 20 + [0] * 10 + [1] * 90, dtype=np.int32)
    
    metrics = compute_intrusion_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == pytest.approx(0.85, abs=1e-4)
    assert metrics["tp"] == 90
    assert metrics["fp"] == 20
    assert metrics["tn"] == 80
    assert metrics["fn"] == 10
    
    # FPR = FP / (FP + TN) = 20 / (20 + 80) = 0.20
    assert metrics["fpr"] == pytest.approx(0.20, abs=1e-4)
    # Recall (Sensitivity) = TP / (TP + FN) = 90 / (90 + 10) = 0.90
    assert metrics["recall_attack"] == pytest.approx(0.90, abs=1e-4)
    # Precision = TP / (TP + FP) = 90 / (90 + 20) = 90 / 110 = 0.8182
    assert metrics["precision_attack"] == pytest.approx(90.0 / 110.0, abs=1e-4)

def test_df_to_markdown_formatting():
    """Verifies markdown table serialization without external dependencies."""
    df = pd.DataFrame([
        {"Model": "Hybrid Model", "Accuracy": "92.46%", "FPR": "12.52%"},
        {"Model": "Random Forest", "Accuracy": "89.20%", "FPR": "14.10%"}
    ])
    
    md = df_to_markdown(df)
    lines = md.strip().split("\n")
    
    assert len(lines) == 4 # Header, separator, row 1, row 2
    assert lines[0] == "| Model | Accuracy | FPR |"
    assert lines[1] == "| --- | --- | --- |"
    assert "Hybrid Model" in lines[2]
    assert "Random Forest" in lines[3]

def test_plot_confusion_matrix_rendering(tmp_path):
    """Verifies that plot_confusion_matrix exports a valid, high-resolution PNG figure."""
    cm = np.array([[450, 50], [30, 470]], dtype=np.int32)
    fig_path = tmp_path / "test_confusion_matrix.png"
    
    out_path = plot_confusion_matrix(cm, "Test Hybrid Model", fig_path)
    
    assert out_path.exists()
    assert out_path.stat().st_size > 5000 # Valid PNG size > 5KB
