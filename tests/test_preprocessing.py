"""
tests/test_preprocessing.py - Unit Tests for Data Preprocessing & Windowing
Verifies missing value imputation, scaling, PCA reduction, and sequence tensor shapes.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import (
    clean_flow_records, fit_preprocessors,
    transform_flow_records, shape_sequences
)
from src.config import WINDOW_SIZE, PCA_COMPONENTS

@pytest.fixture
def mock_raw_df():
    """Generates synthetic flow dataframe with anomalies."""
    np.random.seed(42)
    n = 35
    data = {
        "dur": [0.05 if i % 2 == 0 else np.nan for i in range(n)],
        "sbytes": [1024 if i % 3 != 0 else np.inf for i in range(n)],
        "dbytes": [2048 if i % 4 != 0 else -np.inf for i in range(n)],
        "spkts": [10] * n,
        "dpkts": [20] * n,
        "proto": ["tcp", "udp", "icmp"] * 11 + ["tcp", "udp"],
        "service": ["http", "dns"] * 17 + ["http"],
        "state": ["con", "fin"] * 17 + ["con"],
        "label": [0] * 30 + [1] * 5
    }
    # Add dummy columns to ensure >= 25 numeric features
    for col_idx in range(25):
        data[f"feat_{col_idx}"] = np.random.randn(n)
    return pd.DataFrame(data)

def test_clean_flow_records_removes_nans_and_infs(mock_raw_df):
    clean = clean_flow_records(mock_raw_df)
    assert not clean.isna().any().any()
    assert not np.isinf(clean.select_dtypes(include=[np.number]).values).any()

def test_fit_and_transform_pipeline(mock_raw_df):
    scaler, pca, cols = fit_preprocessors(mock_raw_df, n_components=PCA_COMPONENTS)
    assert len(cols) >= PCA_COMPONENTS
    assert pca.n_components == PCA_COMPONENTS
    
    transformed = transform_flow_records(mock_raw_df, scaler, pca, cols)
    assert transformed.shape == (len(mock_raw_df), PCA_COMPONENTS)
    assert transformed.dtype == np.float32

def test_shape_sequences_sliding_window():
    n_samples = 45
    n_features = 22
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    sequences = shape_sequences(features, window_size=10, step_size=10)
    # 45 samples with step 10 -> 4 full windows of 10
    assert sequences.shape == (4, 10, 22)
    assert sequences.dtype == np.float32

def test_shape_sequences_padding_small_input():
    features = np.random.randn(4, 22).astype(np.float32)
    sequences = shape_sequences(features, window_size=10)
    assert sequences.shape == (1, 10, 22)
    # Check that bottom 6 rows are padded with zeros
    assert np.all(sequences[0, 4:] == 0.0)
