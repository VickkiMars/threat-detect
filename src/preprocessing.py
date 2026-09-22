"""
src/preprocessing.py - Data Cleaning, Scaling, PCA Reduction & Sequence Windowing
Implements leakage-controlled transformations and 10-flow sequence formatting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from pathlib import Path
from src.config import WINDOW_SIZE, PCA_COMPONENTS, PCA_VARIANCE_TARGET, SCALER_PATH, PCA_PATH

# Canonical feature list representing benchmark flow attributes
NUMERIC_FEATURES = [
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "sload", "dload",
    "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt",
    "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"
]

CATEGORICAL_FEATURES = ["proto", "service", "state"]

def clean_flow_records(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces infinite numbers and imputes missing values."""
    df_clean = df.copy()
    
    # Strip string whitespaces
    for col in df_clean.select_dtypes(include=["object", "string"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        
    # Replace infinite values with NaN then impute
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Median imputation for numeric
    for col in numeric_cols:
        median_val = df_clean[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df_clean[col] = df_clean[col].fillna(median_val)
        
    return df_clean

def fit_preprocessors(
    df_train: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    variance_threshold: float = PCA_VARIANCE_TARGET,
    n_components: int = PCA_COMPONENTS
) -> Tuple[StandardScaler, PCA, List[str]]:
    """Fits StandardScaler and PCA strictly on training split."""
    clean_df = clean_flow_records(df_train)
    
    # Select available numeric columns
    cols = feature_cols or [c for c in NUMERIC_FEATURES if c in clean_df.columns]
    if len(cols) < n_components:
        # Fallback to all available numeric columns
        cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
        if "label" in cols:
            cols.remove("label")
        if "attack_cat" in cols:
            cols.remove("attack_cat")
            
    X_train = clean_df[cols].values.astype(np.float32)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Fit PCA with defined component count
    actual_components = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=actual_components, random_state=42)
    pca.fit(X_scaled)
    
    explained_var = float(np.sum(pca.explained_variance_ratio_))
    return scaler, pca, cols

def transform_flow_records(
    df: pd.DataFrame,
    scaler: StandardScaler,
    pca: PCA,
    feature_cols: List[str]
) -> np.ndarray:
    """Transforms new or test flow records using frozen scaler and PCA."""
    clean_df = clean_flow_records(df)
    
    # Handle missing columns if any
    for col in feature_cols:
        if col not in clean_df.columns:
            clean_df[col] = 0.0
            
    X = clean_df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return X_pca.astype(np.float32)

def shape_sequences(
    features: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step_size: Optional[int] = None
) -> np.ndarray:
    """Groups continuous flow vectors into 3D sequential tensors (N_windows, window_size, features)."""
    step = step_size or window_size
    n_samples, n_features = features.shape
    
    if n_samples < window_size:
        # Pad with zeros if fewer than window_size
        padding = np.zeros((window_size - n_samples, n_features), dtype=np.float32)
        padded = np.vstack([features, padding])
        return np.expand_dims(padded, axis=0)
        
    sequences = []
    for i in range(0, n_samples - window_size + 1, step):
        seq = features[i:i + window_size]
        sequences.append(seq)
        
    if not sequences:
        # Terminal fallback
        sequences.append(features[-window_size:])
        
    return np.array(sequences, dtype=np.float32)

def save_preprocessors(
    scaler: StandardScaler,
    pca: PCA,
    feature_cols: List[str],
    scaler_path: Optional[Path] = None,
    pca_path: Optional[Path] = None
) -> None:
    """Serializes fitted scaler, PCA, and column order."""
    s_path = scaler_path or SCALER_PATH
    p_path = pca_path or PCA_PATH
    s_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump({"scaler": scaler, "features": feature_cols}, str(s_path))
    joblib.dump(pca, str(p_path))

def load_preprocessors(
    scaler_path: Optional[Path] = None,
    pca_path: Optional[Path] = None
) -> Tuple[StandardScaler, PCA, List[str]]:
    """Loads frozen scaler, PCA, and feature column list."""
    s_path = scaler_path or SCALER_PATH
    p_path = pca_path or PCA_PATH
    
    scaler_data = joblib.load(str(s_path))
    scaler = scaler_data["scaler"]
    features = scaler_data["features"]
    pca = joblib.load(str(p_path))
    return scaler, pca, features
