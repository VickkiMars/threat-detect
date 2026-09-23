"""
tests/test_balancing.py - Unit & Academic Compliance Tests for SMOTE-Tomek Resampling
Verifies synthetic over-sampling, boundary cleaning, shape preservation,
and strict data leakage prevention (FR3).
"""

import pytest
import numpy as np
from src.balancing import balance_training_data, assert_leakage_free_resampling

def test_balance_training_data_imbalanced_input():
    """Verifies SMOTE + Tomek Links balances minority attack samples in training data."""
    np.random.seed(42)
    # Create imbalanced dataset: 100 benign (class 0), 15 attack (class 1)
    X_benign = np.random.randn(100, 22).astype(np.float32)
    X_attack = np.random.randn(15, 22).astype(np.float32) + 2.0
    
    X_train = np.vstack([X_benign, X_attack])
    y_train = np.array([0] * 100 + [1] * 15, dtype=np.int32)
    
    X_res, y_res, meta = balance_training_data(X_train, y_train, random_state=42)
    
    # Assertions
    assert meta["resampling_applied"] is True
    assert meta["samples_after"] > meta["samples_before"]
    assert X_res.shape[1] == 22
    assert len(X_res) == len(y_res)
    
    # Minority class count should have increased significantly
    classes, counts = np.unique(y_res, return_counts=True)
    count_dict = dict(zip(classes, counts))
    assert count_dict[1] > 15
    assert set(np.unique(y_res)) == {0, 1}

def test_balance_training_data_small_minority_fallback():
    """Verifies graceful fallback when minority class has fewer than 2 samples."""
    X_train = np.random.randn(20, 22).astype(np.float32)
    y_train = np.array([0] * 19 + [1] * 1, dtype=np.int32) # only 1 minority sample
    
    X_res, y_res, meta = balance_training_data(X_train, y_train)
    
    assert meta["resampling_applied"] is False
    assert "fewer than 2 samples" in meta["reason"]
    assert len(X_res) == 20
    assert len(y_res) == 20

def test_balance_training_data_feature_preservation():
    """Verifies that column dimension and dtype are preserved after resampling."""
    X_train = np.random.randn(80, 15).astype(np.float32)
    y_train = np.array([0] * 60 + [1] * 20, dtype=np.int32)
    
    X_res, y_res, meta = balance_training_data(X_train, y_train, random_state=42)
    assert X_res.shape[1] == 15
    assert X_res.dtype == np.float32 or np.issubdtype(X_res.dtype, np.floating)
    assert not np.isnan(X_res).any()

def test_assert_leakage_free_resampling_guard():
    """Verifies that non-training partition names trigger strict Data Leakage exceptions."""
    # Permitted partition names
    assert_leakage_free_resampling("train")
    assert_leakage_free_resampling("training")
    assert_leakage_free_resampling("TRAIN")
    assert_leakage_free_resampling("  training  ")
    
    # Prohibited partition names (data leakage violations)
    with pytest.raises(ValueError, match="DATA LEAKAGE VIOLATION"):
        assert_leakage_free_resampling("test")
        
    with pytest.raises(ValueError, match="DATA LEAKAGE VIOLATION"):
        assert_leakage_free_resampling("val")
        
    with pytest.raises(ValueError, match="DATA LEAKAGE VIOLATION"):
        assert_leakage_free_resampling("validation")
        
    with pytest.raises(ValueError, match="DATA LEAKAGE VIOLATION"):
        assert_leakage_free_resampling("live_stream")
