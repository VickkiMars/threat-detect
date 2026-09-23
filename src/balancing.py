"""
src/balancing.py - SMOTE + Tomek Links Resampling Module
Applies synthetic over-sampling (SMOTE) and boundary under-sampling (Tomek Links)
strictly to training partitions to mitigate attack class imbalance without data leakage.
"""

import numpy as np
from typing import Tuple, Dict, Any
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

def balance_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Applies SMOTE + Tomek Links resampling to the training partition.
    
    Args:
        X_train: Feature matrix of shape (N_samples, N_features).
        y_train: Binary target vector of shape (N_samples,).
        random_state: Seed for reproducibility.
        
    Returns:
        Tuple of (X_resampled, y_resampled, metadata_dict)
    """
    classes, counts = np.unique(y_train, return_counts=True)
    initial_distribution = dict(zip([int(c) for c in classes], [int(cnt) for cnt in counts]))
    
    min_count = min(counts)
    
    # Adjust k_neighbors for SMOTE if minority class is small
    k_neighbors = min(5, max(1, min_count - 1))
    
    if min_count < 2:
        # Cannot oversample with fewer than 2 samples of minority class
        return X_train, y_train, {
            "initial_distribution": initial_distribution,
            "resampled_distribution": initial_distribution,
            "resampling_applied": False,
            "reason": "Minority class has fewer than 2 samples."
        }
        
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    tomek = TomekLinks()
    
    resampler = SMOTETomek(
        smote=smote,
        tomek=tomek,
        random_state=random_state
    )
    
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    
    res_classes, res_counts = np.unique(y_resampled, return_counts=True)
    resampled_distribution = dict(zip([int(c) for c in res_classes], [int(cnt) for cnt in res_counts]))
    
    metadata = {
        "initial_distribution": initial_distribution,
        "resampled_distribution": resampled_distribution,
        "resampling_applied": True,
        "samples_before": int(len(y_train)),
        "samples_after": int(len(y_resampled)),
        "k_neighbors_used": k_neighbors
    }
    
    return X_resampled, y_resampled, metadata

def assert_leakage_free_resampling(partition_name: str) -> None:
    """Enforces academic requirement that SMOTE/Tomek is strictly forbidden on non-train partitions."""
    norm = partition_name.strip().lower()
    if norm not in ("train", "training"):
        raise ValueError(
            f"DATA LEAKAGE VIOLATION: Class balancing requested on '{partition_name}' partition. "
            "Per FR3 and dissertation requirements, SMOTE and Tomek Links must ONLY be applied to the training split."
        )
