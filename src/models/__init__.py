"""
src/models/ - Model Architecture Packages
Contains classical baselines and deep learning / hybrid neural network definitions.
"""
from src.models.classical import (
    ClassicalBaselineModel,
    DecisionTreeModel,
    RandomForestModel,
    SVMModel,
    train_classical_baselines
)
from src.models.deep_learning import (
    CNNOnlyModel,
    LSTMOnlyModel,
    HybridCNNLSTMModel,
    train_neural_model
)

__all__ = [
    "ClassicalBaselineModel",
    "DecisionTreeModel",
    "RandomForestModel",
    "SVMModel",
    "train_classical_baselines",
    "CNNOnlyModel",
    "LSTMOnlyModel",
    "HybridCNNLSTMModel",
    "train_neural_model"
]
