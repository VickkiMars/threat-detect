"""
src/train.py - End-to-End Training Pipeline Orchestrator
Executes sequence-level stratified data partitioning, leakage-free preprocessing,
training-only SMOTE+Tomek class balancing, and trains all baseline and hybrid models.
"""

import argparse
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR, SAMPLE_FLOWS_PATH, MODELS_DIR, REFERENCE_MODEL_DIR,
    SCALER_PATH, PCA_PATH, WINDOW_SIZE, PCA_COMPONENTS, PCA_VARIANCE_TARGET
)
from src.preprocessing import (
    clean_flow_records, fit_preprocessors, transform_flow_records,
    shape_sequences, save_preprocessors, NUMERIC_FEATURES
)
from src.balancing import balance_training_data, assert_leakage_free_resampling
from src.models.classical import train_classical_baselines, ClassicalBaselineModel
from src.models.deep_learning import (
    CNNOnlyModel, LSTMOnlyModel, HybridCNNLSTMModel,
    train_neural_model, DeepLearningModel
)
from src.quantization import quantize_and_export_hybrid_model
from src.evaluate import run_evaluation_suite

CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

def build_temporal_sequences(
    X_pca: np.ndarray,
    y: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step_size: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs ordered 10-flow sequences from flow representations.
    Target label is assigned as the terminal record's label in the sequence.
    """
    N = len(X_pca)
    X_seq = []
    y_seq = []
    for i in range(0, N - window_size + 1, step_size):
        X_seq.append(X_pca[i:i + window_size])
        y_seq.append(int(y[i + window_size - 1]))
        
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int32)

def run_training_pipeline(
    dataset_path: Optional[Path] = None,
    test_dataset_path: Optional[Path] = None,
    save_checkpoints: bool = True,
    step_size: int = 10,
    max_train_records: Optional[int] = None,
    epochs: int = 15,
    quantize: bool = True,
    evaluate: bool = True
) -> Dict[str, Any]:
    """Executes the complete training workflow across all 6 architectures."""
    unsw_train = DATA_DIR / "UNSW_NB15_training-set.csv"
    unsw_test = DATA_DIR / "UNSW_NB15_testing-set.csv"
    
    if dataset_path:
        train_file = Path(dataset_path)
    elif unsw_train.exists():
        train_file = unsw_train
    else:
        train_file = SAMPLE_FLOWS_PATH
        
    if test_dataset_path:
        test_file = Path(test_dataset_path)
    elif unsw_test.exists() and train_file == unsw_train:
        test_file = unsw_test
    else:
        test_file = None

    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_file}.")
        
    print("=" * 75)
    print(" AI NETWORK THREAT DETECTION — SPRINT 2 MODEL TRAINING PIPELINE")
    print("=" * 75)
    print(f"Training dataset source : {train_file}")
    if test_file:
        print(f"Testing dataset source  : {test_file}")
    
    df_train = pd.read_csv(train_file)
    if max_train_records and len(df_train) > max_train_records:
        print(f"Capping training records to: {max_train_records:,}")
        df_train = df_train.iloc[:max_train_records]
    print(f"Total training records    : {len(df_train):,} | Distribution: {df_train['label'].value_counts().to_dict()}")

    # 1. Fit Preprocessors Strictly on Training Dataset (Leakage Prevention)
    scaler, pca, feature_cols = fit_preprocessors(
        df_train,
        n_components=PCA_COMPONENTS,
        variance_threshold=PCA_VARIANCE_TARGET
    )
    save_preprocessors(scaler, pca, feature_cols, SCALER_PATH, PCA_PATH)
    print(f"Fitted & saved preprocessors (PCA components: {pca.n_components_}, variance: {np.sum(pca.explained_variance_ratio_):.4f})")

    # 2. Transform Training Flows using Frozen Preprocessors
    X_pca_train = transform_flow_records(df_train, scaler, pca, feature_cols)
    y_raw_train = df_train["label"].values.astype(np.int32)
    X_train_full_seq, y_train_full_seq = build_temporal_sequences(X_pca_train, y_raw_train, window_size=WINDOW_SIZE, step_size=step_size)
    print(f"Constructed training sequences: Shape={X_train_full_seq.shape}")

    # 3. Form Evaluation Partitions
    if test_file and test_file.exists():
        df_test = pd.read_csv(test_file)
        if max_train_records and len(df_test) > max_train_records:
            df_test = df_test.iloc[:max_train_records]
        print(f"Total testing records     : {len(df_test):,} | Distribution: {df_test['label'].value_counts().to_dict()}")
        X_pca_test = transform_flow_records(df_test, scaler, pca, feature_cols)
        y_raw_test = df_test["label"].values.astype(np.int32)
        X_test_seq, y_test_seq = build_temporal_sequences(X_pca_test, y_raw_test, window_size=WINDOW_SIZE, step_size=step_size)
        
        # 85% train / 15% validation split from training sequences for early stopping
        X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
            X_train_full_seq, y_train_full_seq, test_size=0.15, stratify=y_train_full_seq, random_state=42
        )
    else:
        # 70% / 15% / 15% Stratified Partitioning (Chapter 3.8)
        X_train_seq, X_temp, y_train_seq, y_temp = train_test_split(
            X_train_full_seq, y_train_full_seq, test_size=0.30, stratify=y_train_full_seq, random_state=42
        )
        X_val_seq, X_test_seq, y_val_seq, y_test_seq = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )

    print(f"Partitions: Train={len(y_train_seq)}, Val={len(y_val_seq)}, Held-out Test={len(y_test_seq)}")

    # 4. Apply SMOTE + Tomek Links Strictly to Training Sequences
    assert_leakage_free_resampling("train")
    print("Applying SMOTE + Tomek Links resampling strictly to Training sequences...")
    N_tr, W, F = X_train_seq.shape
    X_train_flat = X_train_seq.reshape(N_tr, W * F)
    X_train_bal_flat, y_train_bal, balance_meta = balance_training_data(X_train_flat, y_train_seq, random_state=42)
    X_train_bal = X_train_bal_flat.reshape(-1, W, F)
    
    print(f"  Before resampling: {balance_meta['initial_distribution']}")
    print(f"  After resampling : {balance_meta['resampled_distribution']}")
    print(f"  Balanced training tensor: {X_train_bal.shape}")

    # 5. Train Classical Baselines (Decision Tree, Random Forest, SVM)
    print("\n--- [1/2] Training Classical Baselines ---")
    classical_models = train_classical_baselines(X_train_bal, y_train_bal, random_state=42)
    for name, model in classical_models.items():
        val_acc = np.mean(model.predict(X_val_seq) == y_val_seq)
        print(f"  * {name:15s} | Val Accuracy: {val_acc:.4f} | Complexity: {model.count_parameters():,}")

    # 6. Train Deep Learning Baselines and Hybrid CNN–LSTM
    print("\n--- [2/2] Training Deep Learning & Hybrid Models ---")
    cnn_model = CNNOnlyModel(window_size=WINDOW_SIZE, n_features=PCA_COMPONENTS)
    cnn_model = train_neural_model(cnn_model, X_train_bal, y_train_bal, X_val_seq, y_val_seq, epochs=epochs, patience=5)
    print(f"  * CNN-only        | Val Accuracy: {cnn_model.history['val_accuracy'][-1]:.4f} | Params: {cnn_model.count_parameters():,}")
    
    lstm_model = LSTMOnlyModel(window_size=WINDOW_SIZE, n_features=PCA_COMPONENTS)
    lstm_model = train_neural_model(lstm_model, X_train_bal, y_train_bal, X_val_seq, y_val_seq, epochs=epochs, patience=5)
    print(f"  * LSTM-only       | Val Accuracy: {lstm_model.history['val_accuracy'][-1]:.4f} | Params: {lstm_model.count_parameters():,}")
    
    hybrid_model = HybridCNNLSTMModel(window_size=WINDOW_SIZE, n_features=PCA_COMPONENTS)
    hybrid_model = train_neural_model(hybrid_model, X_train_bal, y_train_bal, X_val_seq, y_val_seq, epochs=epochs+5, patience=5)
    print(f"  * Hybrid CNN–LSTM | Val Accuracy: {hybrid_model.history['val_accuracy'][-1]:.4f} | Params: {hybrid_model.count_parameters():,}")

    # 7. Save Checkpoints
    all_models = {
        **classical_models,
        "CNN-only": cnn_model,
        "LSTM-only": lstm_model,
        "Hybrid CNN–LSTM": hybrid_model
    }
    
    if save_checkpoints:
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        for name, m in all_models.items():
            slug = name.lower().replace(" ", "_").replace("–", "_").replace("(", "").replace(")", "")
            ckpt_path = CHECKPOINTS_DIR / f"{slug}.joblib"
            m.save(ckpt_path)
            
        eval_payload = {
            "X_train_seq": X_train_bal,
            "y_train_seq": y_train_bal,
            "X_test_seq": X_test_seq,
            "y_test_seq": y_test_seq,
            "X_val_seq": X_val_seq,
            "y_val_seq": y_val_seq,
            "feature_cols": feature_cols
        }
        joblib.dump(eval_payload, str(CHECKPOINTS_DIR / "eval_data.joblib"))
        print(f"\nAll model checkpoints saved to: {CHECKPOINTS_DIR}")

    # 8. Post-Training Quantization
    if quantize:
        print("\n" + "=" * 75)
        print(" TENSORFLOW LITE FLATBUFFER QUANTIZATION & COMPRESSION")
        print("=" * 75)
        quant_meta = quantize_and_export_hybrid_model()

    # 9. Empirical Evaluation Suite
    if evaluate:
        print("\n" + "=" * 75)
        print(" RUNNING ACADEMIC EVALUATION & DISSERTATION REPLICATION SUITE")
        print("=" * 75)
        eval_results = run_evaluation_suite()

    return {
        "models": all_models,
        "splits": {
            "X_train_seq": X_train_bal, "y_train_seq": y_train_bal,
            "X_val_seq": X_val_seq, "y_val_seq": y_val_seq,
            "X_test_seq": X_test_seq, "y_test_seq": y_test_seq
        },
        "balance_meta": balance_meta
    }

def main():
    parser = argparse.ArgumentParser(description="Train baseline & hybrid models for Grace edge NIDS.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to training CSV")
    parser.add_argument("--test-dataset", type=str, default=None, help="Path to testing CSV")
    parser.add_argument("--step-size", type=int, default=10, help="Window sliding step size")
    parser.add_argument("--max-records", type=int, default=None, help="Max raw flow records to process (default: all)")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs for neural models")
    parser.add_argument("--no-quantize", action="store_true", help="Skip TFLite quantization")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip evaluation suite")
    args = parser.parse_args()

    run_training_pipeline(
        dataset_path=args.dataset,
        test_dataset_path=args.test_dataset,
        step_size=args.step_size,
        max_train_records=args.max_records,
        epochs=args.epochs,
        quantize=not args.no_quantize,
        evaluate=not args.no_evaluate
    )

if __name__ == "__main__":
    main()
