"""
src/models/deep_learning.py - Deep Learning Baselines and Hybrid CNN–LSTM Architecture

Implemented with TensorFlow / Keras, as specified in dissertation Table 4.1
("Development environment and tools": Deep learning = TensorFlow/Keras).

This module replaces the earlier hand-written NumPy forward/backward implementation.
It provides the CNN-only, LSTM-only and Hybrid CNN–LSTM detectors using Keras
functional models, sparse-categorical cross-entropy, the Adam optimiser and
early stopping, while preserving the public API consumed by:
    src/train.py, src/evaluate.py, src/cicids_eval.py, src/quantization.py,
    src/models/__init__.py and tests/test_models.py

Edge-deployment note: the fixed-length (10-timestep) recurrent layers are declared
with ``unroll=True`` so that the Keras graph lowers to built-in TensorFlow Lite
operators during conversion, as required by Table 4.5 / FR5.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducible initialisation for the dissertation experiment protocol (seed 42).
SEED = 42
keras.utils.set_random_seed(SEED)


class DeepLearningModel:
    """Base class for edge-oriented sequence deep learning models (TensorFlow/Keras backend)."""

    def __init__(
        self,
        name: str,
        window_size: int = 10,
        n_features: int = 22,
        n_classes: int = 2,
    ):
        self.name = name
        self.window_size = window_size
        self.n_features = n_features
        self.n_classes = n_classes
        self.network: Optional[keras.Model] = None
        self._probe: Optional[keras.Model] = None
        self._probe_layers: Dict[str, str] = {}
        self.is_fitted = False
        self.best_val_loss = float("inf")
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def count_parameters(self) -> int:
        """Total number of scalar parameters (trainable + non-trainable) in the network."""
        if self.network is None:
            return 0
        return int(sum(int(np.prod(w.shape)) for w in self.network.get_weights()))

    @property
    def weights(self) -> List[np.ndarray]:
        """Backward-compatible accessor for the trained Keras weight arrays."""
        if self.network is None:
            return []
        return self.network.get_weights()

    # ------------------------------------------------------------------
    # Probe sub-model: exposes intermediate activations for tests/analysis
    # ------------------------------------------------------------------
    def _build_probe(self, probe_layers: Dict[str, str]) -> None:
        """Records the cache_key -> Keras layer-name mapping used by ``forward``."""
        self._probe_layers = dict(probe_layers)

    def _rebuild_probe(self) -> None:
        """Constructs a multi-output Keras model exposing the probed layer activations."""
        if self.network is None or not self._probe_layers:
            self._probe = None
            return
        outputs = {
            cache_key: self.network.get_layer(layer_name).output
            for cache_key, layer_name in self._probe_layers.items()
        }
        self._probe = keras.Model(self.network.inputs, outputs, name=f"{self.name}_probe")

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------
    def compile(self, learning_rate: float = 0.001) -> None:
        """Compiles the Keras graph with Adam and sparse categorical cross-entropy."""
        if self.network is None:
            return
        self.network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def forward(
        self, X: np.ndarray, training: bool = False
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Runs a forward pass and returns ``(probabilities, activation_cache)``.

        The cache mirrors the previous NumPy implementation's contract so that
        existing callers and unit tests continue to work unchanged.
        """
        X = np.asarray(X, dtype=np.float32)
        probs = np.asarray(self.network(X, training=training), dtype=np.float32)

        cache: Dict[str, np.ndarray] = {}
        if self._probe is not None:
            # The probe is a multi-output functional model built from
            # ``self.network.inputs`` (a list), so it expects a list-structured
            # feed. Passing a bare tensor triggers a Keras structure warning.
            probed = self._probe([X], training=training)
            if isinstance(probed, dict):
                cache = {k: np.asarray(v) for k, v in probed.items()}
            elif isinstance(probed, (list, tuple)):
                cache = {
                    k: np.asarray(v)
                    for k, v in zip(self._probe_layers.keys(), probed)
                }
            else:
                cache = {next(iter(self._probe_layers)): np.asarray(probed)}
        return probs, cache

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class probabilities of shape (N, n_classes)."""
        X = np.asarray(X, dtype=np.float32)
        return np.asarray(self.network.predict(X, verbose=0), dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns the argmax class label for each input window."""
        return np.argmax(self.predict_proba(X), axis=1)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Performs a single Keras gradient update on one mini-batch and returns the loss."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        loss = self.network.train_on_batch(X, y)
        if isinstance(loss, (list, tuple)):
            loss = loss[0]
        return float(loss)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, file_path: Path) -> Path:
        """Serialises the wrapper (architecture + weights + history) via joblib."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(file_path))
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "DeepLearningModel":
        """Restores a previously saved model wrapper."""
        return joblib.load(str(file_path))

    def __getstate__(self) -> Dict[str, Any]:
        """Serialises the Keras graph as JSON + weight arrays so joblib round-trips cleanly."""
        state = self.__dict__.copy()
        if self.network is not None:
            state["_network_json"] = self.network.to_json()
            state["_network_weights"] = [np.asarray(w) for w in self.network.get_weights()]
        else:
            state["_network_json"] = None
            state["_network_weights"] = None
        # Keras objects are not reliably picklable; the JSON + weights pair replaces them.
        state["network"] = None
        state["_probe"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Rehydrates the Keras graph from the serialised JSON + weights."""
        net_json = state.pop("_network_json", None)
        net_weights = state.pop("_network_weights", None)
        self.__dict__.update(state)

        self.network = None
        self._probe = None
        if net_json is not None:
            self.network = keras.models.model_from_json(net_json)
            if net_weights is not None:
                self.network.set_weights(net_weights)
            self.compile()
            self._rebuild_probe()


class CNNOnlyModel(DeepLearningModel):
    """
    CNN-only Baseline (Chapter 3.9 & 4.5):
    Input (N, 10, 22) -> Conv1D(64 filters, kernel=3, ReLU) -> MaxPooling1D(2)
    -> Flatten (5 * 64 = 320) -> Dense(64, ReLU) -> Dense(2, Softmax)
    """

    def __init__(self, window_size: int = 10, n_features: int = 22, n_classes: int = 2):
        super().__init__("CNN-only Baseline", window_size, n_features, n_classes)

        inputs = keras.Input(shape=(window_size, n_features), name="input_flow_sequence")
        x = layers.Conv1D(
            filters=64, kernel_size=3, padding="same", activation="relu", name="conv1d_1"
        )(inputs)
        x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)
        x = layers.Flatten(name="flatten")(x)
        d1 = layers.Dense(64, activation="relu", name="d1_act")(x)
        probs = layers.Dense(n_classes, activation="softmax", name="probabilities")(d1)

        self.network = keras.Model(inputs, probs, name="cnn_only")
        self._build_probe({"probabilities": "probabilities", "d1_act": "d1_act"})
        self.compile()
        self._rebuild_probe()


class LSTMOnlyModel(DeepLearningModel):
    """
    LSTM-only Baseline (Chapter 3.9 & 4.5):
    Input (N, 10, 22) -> LSTM(hidden_dim, tanh/sigmoid gates) -> Dense(64, ReLU)
    -> Dense(2, Softmax)

    ``unroll=True`` statically unrolls the fixed 10-step recurrence so the graph
    converts to built-in TensorFlow Lite operators (see Table 4.5 / FR5).
    """

    def __init__(
        self,
        window_size: int = 10,
        n_features: int = 22,
        hidden_dim: int = 64,
        n_classes: int = 2,
    ):
        super().__init__("LSTM-only Baseline", window_size, n_features, n_classes)

        inputs = keras.Input(shape=(window_size, n_features), name="input_flow_sequence")
        h = layers.LSTM(
            hidden_dim, return_sequences=False, unroll=True, name="h_final"
        )(inputs)
        d1 = layers.Dense(64, activation="relu", name="d1_act")(h)
        probs = layers.Dense(n_classes, activation="softmax", name="probabilities")(d1)

        self.network = keras.Model(inputs, probs, name="lstm_only")
        self._build_probe({"probabilities": "probabilities", "h_final": "h_final"})
        self.compile()
        self._rebuild_probe()


class HybridCNNLSTMModel(DeepLearningModel):
    """
    Proposed Hybrid CNN–LSTM detector (dissertation Table 3.3):
    Conv1D(64, k=3, ReLU) -> BatchNormalisation -> MaxPooling1D(2)
    -> Conv1D(32, k=3, ReLU) -> Dropout(0.30)
    -> LSTM(64, return_sequences=True) -> LSTM(32)
    -> Dropout(0.30) -> Dense(64, ReLU) -> Dense(n_classes, Softmax)

    Both recurrent layers are statically unrolled over the fixed 10-flow window so
    that the Keras graph lowers to built-in TensorFlow Lite operators, avoiding
    TensorList/Flex dependencies at the edge (FR5, Table 4.5).
    """

    def __init__(
        self,
        window_size: int = 10,
        n_features: int = 22,
        cnn_filters: int = 64,
        lstm_units: int = 64,
        n_classes: int = 2,
    ):
        super().__init__("Hybrid CNN–LSTM", window_size, n_features, n_classes)

        inputs = keras.Input(shape=(window_size, n_features), name="input_flow_sequence")
        x = layers.Conv1D(
            filters=cnn_filters, kernel_size=3, padding="same",
            activation="relu", name="conv1d_1",
        )(inputs)
        x = layers.BatchNormalization(name="batch_norm")(x)
        x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)
        x = layers.Conv1D(
            filters=max(cnn_filters // 2, 1), kernel_size=3, padding="same",
            activation="relu", name="conv1d_2",
        )(x)
        x = layers.Dropout(0.30, name="dropout_1")(x)
        x = layers.LSTM(
            lstm_units, return_sequences=True, unroll=True, name="lstm_1"
        )(x)
        h = layers.LSTM(
            max(lstm_units // 2, 1), return_sequences=False, unroll=True, name="h_final"
        )(x)
        h = layers.Dropout(0.30, name="dropout_2")(h)
        d1 = layers.Dense(64, activation="relu", name="d1_act")(h)
        probs = layers.Dense(n_classes, activation="softmax", name="probabilities")(d1)

        self.network = keras.Model(inputs, probs, name="hybrid_cnn_lstm")
        self._build_probe({
            "probabilities": "probabilities",
            "d1_act": "d1_act",
            "h_final": "h_final",
        })
        self.compile()
        self._rebuild_probe()


def train_neural_model(
    model: DeepLearningModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 64,
    patience: int = 5,
) -> DeepLearningModel:
    """
    Trains a Keras deep learning model with Adam optimisation, sparse categorical
    cross-entropy and early stopping on the validation loss.

    Mirrors the previous NumPy training loop's contract: the fitted model is returned
    with ``history`` populated by ``train_loss``, ``val_loss`` and ``val_accuracy``
    lists, and ``best_val_loss`` set to the best observed validation loss.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int32)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    hist = model.network.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    model.history["train_loss"] = [float(v) for v in hist.history.get("loss", [])]
    model.history["val_loss"] = [float(v) for v in hist.history.get("val_loss", [])]
    model.history["val_accuracy"] = [
        float(v) for v in hist.history.get("val_accuracy", [])
    ]
    model.is_fitted = True
    model.best_val_loss = (
        float(min(model.history["val_loss"]))
        if model.history["val_loss"]
        else float("inf")
    )
    return model
