"""
tests/test_models.py - Unit Tests for Classical & Deep Learning Architectures
Validates forward passes, output probability shapes, parameter complexity counts,
and gradient update steps across all 6 comparative model architectures.
"""

import pytest
import numpy as np
from src.models.classical import (
    DecisionTreeModel,
    RandomForestModel,
    SVMModel,
    train_classical_baselines
)
from src.models.deep_learning import (
    CNNOnlyModel,
    LSTMOnlyModel,
    HybridCNNLSTMModel
)

@pytest.fixture
def sequence_batch():
    """Generates synthetic 3D sequence tensors of shape (N=8, Window=10, Features=22)."""
    np.random.seed(42)
    X = np.random.randn(8, 10, 22).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1], dtype=np.int32)
    return X, y

# ==============================================================================
# Classical Baseline Model Tests
# ==============================================================================

def test_decision_tree_baseline(sequence_batch):
    X, y = sequence_batch
    dt = DecisionTreeModel(max_depth=5, random_state=42)
    dt.fit(X, y)
    
    preds = dt.predict(X)
    probs = dt.predict_proba(X)
    
    assert preds.shape == (8,)
    assert probs.shape == (8, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert dt.count_parameters() > 0

def test_random_forest_baseline(sequence_batch):
    X, y = sequence_batch
    rf = RandomForestModel(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X, y)
    
    preds = rf.predict(X)
    probs = rf.predict_proba(X)
    
    assert preds.shape == (8,)
    assert probs.shape == (8, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert rf.count_parameters() > 10

def test_svm_baseline(sequence_batch):
    X, y = sequence_batch
    svm = SVMModel(C=1.0, random_state=42)
    svm.fit(X, y)
    
    preds = svm.predict(X)
    probs = svm.predict_proba(X)
    
    assert preds.shape == (8,)
    assert probs.shape == (8, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert svm.count_parameters() > 0

def test_train_classical_baselines_orchestrator(sequence_batch):
    X, y = sequence_batch
    models = train_classical_baselines(X, y, random_state=42, max_svm_samples=50)
    
    assert "Decision Tree" in models
    assert "Random Forest" in models
    assert "SVM" in models
    for m in models.values():
        assert m.is_fitted is True

# ==============================================================================
# Deep Learning Baseline & Proposed Hybrid Model Tests
# ==============================================================================

def test_cnn_only_model_architecture(sequence_batch):
    X, _ = sequence_batch
    model = CNNOnlyModel(window_size=10, n_features=22, n_classes=2)
    
    probs, cache = model.forward(X, training=False)
    preds = model.predict(X)
    
    assert probs.shape == (8, 2)
    assert preds.shape == (8,)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    assert model.count_parameters() > 5000
    assert "d1_act" in cache

def test_lstm_only_model_architecture(sequence_batch):
    X, _ = sequence_batch
    model = LSTMOnlyModel(window_size=10, n_features=22, hidden_dim=32, n_classes=2)
    
    probs, cache = model.forward(X, training=False)
    preds = model.predict(X)
    
    assert probs.shape == (8, 2)
    assert preds.shape == (8,)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    assert model.count_parameters() > 2000
    assert "h_final" in cache

def test_hybrid_cnn_lstm_model_proposed(sequence_batch):
    X, y = sequence_batch
    model = HybridCNNLSTMModel(
        window_size=10,
        n_features=22,
        cnn_filters=32,
        lstm_units=32,
        n_classes=2
    )
    
    # Forward pass
    probs, cache = model.forward(X, training=False)
    preds = model.predict(X)
    
    assert probs.shape == (8, 2)
    assert preds.shape == (8,)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    assert model.count_parameters() > 5000
    
    # Train step gradient update
    loss = model.train_step(X, y)
    assert isinstance(loss, float)
    assert loss > 0.0
