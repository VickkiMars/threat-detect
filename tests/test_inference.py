"""
tests/test_inference.py - Unit Tests for TFLite Edge Inference Engine
Verifies tensor shapes, probability normalization, file size, and latency contracts.
"""

import pytest
import numpy as np
from pathlib import Path
from src.config import TFLITE_MODEL_PATH, NFR_TARGETS, WINDOW_SIZE, PCA_COMPONENTS
from src.inference_engine import EdgeInferenceEngine

def test_inference_engine_loads_model():
    engine = EdgeInferenceEngine(TFLITE_MODEL_PATH)
    meta = engine.get_model_metadata()
    
    assert meta["input_shape"] == [1, WINDOW_SIZE, PCA_COMPONENTS]
    assert meta["output_shape"] == [1, 2]
    assert meta["file_size_kb"] <= NFR_TARGETS["NFR2_MAX_MODEL_SIZE_KB"]
    assert meta["num_threads"] == 1

def test_predict_window_produces_normalized_probabilities():
    engine = EdgeInferenceEngine(TFLITE_MODEL_PATH)
    sample_seq = np.random.randn(WINDOW_SIZE, PCA_COMPONENTS).astype(np.float32)
    
    pred_class, confidence, latency_ms = engine.predict_window(sample_seq)
    
    assert pred_class in (0, 1)
    assert 0.0 <= confidence <= 1.0
    assert latency_ms > 0.0
    # Strict compliance with NFR1 (<= 50.0 ms)
    assert latency_ms <= NFR_TARGETS["NFR1_MAX_LATENCY_MS"]

def test_predict_window_with_batch_dimension():
    engine = EdgeInferenceEngine(TFLITE_MODEL_PATH)
    sample_seq_3d = np.random.randn(1, WINDOW_SIZE, PCA_COMPONENTS).astype(np.float32)
    
    pred_class, confidence, latency_ms = engine.predict_window(sample_seq_3d)
    assert pred_class in (0, 1)
    assert 0.0 <= confidence <= 1.0
