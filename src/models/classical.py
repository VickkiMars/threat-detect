"""
src/models/classical.py - Classical Machine Learning Intrusion Detection Baselines
Implements Decision Tree, Random Forest, and SVM baselines evaluated on flattened sequence features.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ClassicalBaselineModel:
    """Base wrapper for classical scikit-learn classifiers on sequence data."""
    def __init__(self, name: str, estimator: Any):
        self.name = name
        self.estimator = estimator
        self.is_fitted = False
        
    def _flatten_input(self, X: np.ndarray) -> np.ndarray:
        """Flattens 3D sequences (N, Window, Features) into 2D tabular (N, Window * Features)."""
        if X.ndim == 3:
            N, W, F = X.shape
            return X.reshape(N, W * F)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalBaselineModel":
        X_flat = self._flatten_input(X)
        self.estimator.fit(X_flat, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(f"{self.name} is not fitted yet.")
        X_flat = self._flatten_input(X)
        return self.estimator.predict(X_flat)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(f"{self.name} is not fitted yet.")
        X_flat = self._flatten_input(X)
        proba = self.estimator.predict_proba(X_flat)
        # Ensure 2-column output [P(benign), P(attack)]
        if proba.shape[1] == 1:
            # Single class edge case
            p1 = proba[:, 0]
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])
        return proba

    def count_parameters(self) -> int:
        """Estimates model parameter / rule complexity."""
        if isinstance(self.estimator, DecisionTreeClassifier):
            return int(self.estimator.tree_.node_count)
        elif isinstance(self.estimator, RandomForestClassifier):
            return int(sum(t.tree_.node_count for t in self.estimator.estimators_))
        elif isinstance(self.estimator, SVC):
            return int(self.estimator.support_vectors_.size)
        return 0

    def save(self, file_path: Path) -> Path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(file_path))
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "ClassicalBaselineModel":
        return joblib.load(str(file_path))

class DecisionTreeModel(ClassicalBaselineModel):
    """Decision Tree Classifier Baseline (Chapter 3.9 & 4.5)."""
    def __init__(self, max_depth: int = 15, random_state: int = 42):
        estimator = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=5,
            random_state=random_state
        )
        super().__init__("Decision Tree", estimator)

class RandomForestModel(ClassicalBaselineModel):
    """Random Forest Classifier Baseline (100 estimators, Chapter 3.9 & 4.5)."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 15, random_state: int = 42):
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        super().__init__("Random Forest", estimator)

class SVMModel(ClassicalBaselineModel):
    """Support Vector Machine (RBF Kernel, Chapter 3.9 & 4.5)."""
    def __init__(self, C: float = 1.0, random_state: int = 42):
        estimator = SVC(
            C=C,
            kernel="rbf",
            probability=True,
            random_state=random_state
        )
        super().__init__("SVM (RBF)", estimator)

def train_classical_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    max_svm_samples: int = 3000
) -> Dict[str, ClassicalBaselineModel]:
    """Trains and returns all three classical machine learning baselines."""
    dt = DecisionTreeModel(random_state=random_state)
    dt.fit(X_train, y_train)

    rf = RandomForestModel(random_state=random_state)
    rf.fit(X_train, y_train)

    # SVM with RBF kernel scales quadratically O(N^2); subsample if training set is large
    svm = SVMModel(random_state=random_state)
    if len(X_train) > max_svm_samples:
        rng = np.random.RandomState(random_state)
        # Stratified subsample
        idx_0 = np.where(y_train == 0)[0]
        idx_1 = np.where(y_train == 1)[0]
        half = max_svm_samples // 2
        s0 = rng.choice(idx_0, size=min(half, len(idx_0)), replace=False)
        s1 = rng.choice(idx_1, size=min(half, len(idx_1)), replace=False)
        svm_idx = np.concatenate([s0, s1])
        rng.shuffle(svm_idx)
        svm.fit(X_train[svm_idx], y_train[svm_idx])
    else:
        svm.fit(X_train, y_train)
        
    return {
        "Decision Tree": dt,
        "Random Forest": rf,
        "SVM": svm
    }
