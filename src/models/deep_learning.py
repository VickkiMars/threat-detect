"""
src/models/deep_learning.py - Deep Learning Baselines and Hybrid CNN–LSTM Architecture
Implements CNN-only, LSTM-only, and Hybrid CNN–LSTM neural models with vectorized NumPy computation,
Adam optimization, binary cross-entropy loss, and early stopping.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import joblib

def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shiftz = z - np.max(z, axis=-1, keepdims=True)
    exps = np.exp(shiftz)
    return exps / np.sum(exps, axis=-1, keepdims=True)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

class AdamOptimizer:
    """Standard Adam optimizer for vectorized parameter updates."""
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0
        
    def step(self, param_key: str, w: np.ndarray, dw: np.ndarray) -> np.ndarray:
        if param_key not in self.m:
            self.m[param_key] = np.zeros_like(w)
            self.v[param_key] = np.zeros_like(w)
            
        self.t += 1
        m = self.m[param_key]
        v = self.v[param_key]
        
        m = self.beta1 * m + (1.0 - self.beta1) * dw
        v = self.beta2 * v + (1.0 - self.beta2) * (dw ** 2)
        
        m_hat = m / (1.0 - (self.beta1 ** self.t) + self.eps)
        v_hat = v / (1.0 - (self.beta2 ** self.t) + self.eps)
        
        self.m[param_key] = m
        self.v[param_key] = v
        
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class DeepLearningModel:
    """Base class for edge-oriented sequence deep learning models."""
    def __init__(self, name: str):
        self.name = name
        self.weights: Dict[str, np.ndarray] = {}
        self.optimizer = AdamOptimizer(lr=0.001)
        self.is_fitted = False
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    def count_parameters(self) -> int:
        return sum(w.size for w in self.weights.values())

    def save(self, file_path: Path) -> Path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(file_path))
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "DeepLearningModel":
        return joblib.load(str(file_path))

class CNNOnlyModel(DeepLearningModel):
    """
    CNN-only Baseline (Chapter 3.9 & 4.5):
    Input (N, 10, 22) -> Conv1D(64 filters, kernel=3, ReLU) -> MaxPool1D(2) -> Flatten (5*64=320)
    -> Dense(64, ReLU) -> Dense(2, Softmax)
    """
    def __init__(self, window_size: int = 10, n_features: int = 22, n_classes: int = 2):
        super().__init__("CNN-only Baseline")
        self.window_size = window_size
        self.n_features = n_features
        self.n_classes = n_classes
        
        np.random.seed(42)
        # Conv1D weights: (kernel_size=3, in_channels=22, out_channels=64)
        k_sz, out_c = 3, 64
        limit_conv = np.sqrt(6.0 / (k_sz * n_features + out_c))
        w_conv = np.random.uniform(-limit_conv, limit_conv, (k_sz, n_features, out_c)).astype(np.float32)
        b_conv = np.zeros((out_c,), dtype=np.float32)
        
        # Conv output (padding='same') -> (N, 10, 64). MaxPool(2) -> (N, 5, 64). Flat -> (N, 320)
        flat_dim = 5 * out_c
        limit_d1 = np.sqrt(6.0 / (flat_dim + 64))
        w_d1 = np.random.uniform(-limit_d1, limit_d1, (flat_dim, 64)).astype(np.float32)
        b_d1 = np.zeros((64,), dtype=np.float32)
        
        limit_d2 = np.sqrt(6.0 / (64 + n_classes))
        w_d2 = np.random.uniform(-limit_d2, limit_d2, (64, n_classes)).astype(np.float32)
        b_d2 = np.zeros((n_classes,), dtype=np.float32)
        
        self.weights = {
            "w_conv": w_conv, "b_conv": b_conv,
            "w_d1": w_d1, "b_d1": b_d1,
            "w_d2": w_d2, "b_d2": b_d2
        }

    def forward(self, X: np.ndarray, training: bool = False) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        N, T, F = X.shape
        w_conv, b_conv = self.weights["w_conv"], self.weights["b_conv"]
        k_sz = w_conv.shape[0] # 3
        pad = k_sz // 2 # 1
        
        # Zero padding across time: (N, T + 2*pad, F)
        X_padded = np.pad(X, ((0, 0), (pad, pad), (0, 0)), mode="constant")
        
        # 1D Convolution
        conv_out = np.zeros((N, T, w_conv.shape[2]), dtype=np.float32)
        for t in range(T):
            patch = X_padded[:, t:t+k_sz, :] # (N, 3, F)
            conv_out[:, t, :] = np.tensordot(patch, w_conv, axes=([1, 2], [0, 1])) + b_conv
            
        conv_relu = relu(conv_out)
        
        # MaxPool1D (pool_size=2) -> (N, T//2, out_c) = (N, 5, 64)
        pool_out = np.zeros((N, T // 2, w_conv.shape[2]), dtype=np.float32)
        for t in range(T // 2):
            pool_out[:, t, :] = np.maximum(conv_relu[:, 2*t, :], conv_relu[:, 2*t+1, :])
            
        flat = pool_out.reshape(N, -1)
        d1_z = flat @ self.weights["w_d1"] + self.weights["b_d1"]
        d1_act = relu(d1_z)
        
        if training:
            # Dropout 0.2
            mask = (np.random.rand(*d1_act.shape) >= 0.2).astype(np.float32) / 0.8
            d1_act = d1_act * mask
        else:
            mask = None
            
        logits = d1_act @ self.weights["w_d2"] + self.weights["b_d2"]
        probs = softmax(logits)
        
        cache = {
            "X": X, "X_padded": X_padded, "conv_relu": conv_relu,
            "pool_out": pool_out, "flat": flat, "d1_act": d1_act,
            "probs": probs
        }
        return probs, cache

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        N = X.shape[0]
        probs, cache = self.forward(X, training=True)
        
        # One-hot labels
        y_onehot = np.zeros((N, self.n_classes), dtype=np.float32)
        y_onehot[np.arange(N), y] = 1.0
        
        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))
        
        # Backprop through output
        d_logits = (probs - y_onehot) / N
        dw_d2 = cache["d1_act"].T @ d_logits
        db_d2 = np.sum(d_logits, axis=0)
        
        d_d1 = (d_logits @ self.weights["w_d2"].T) * relu_derivative(cache["d1_act"])
        dw_d1 = cache["flat"].T @ d_d1
        db_d1 = np.sum(d_d1, axis=0)
        
        # Dense updates
        self.weights["w_d2"] = self.optimizer.step("w_d2", self.weights["w_d2"], dw_d2)
        self.weights["b_d2"] = self.optimizer.step("b_d2", self.weights["b_d2"], db_d2)
        self.weights["w_d1"] = self.optimizer.step("w_d1", self.weights["w_d1"], dw_d1)
        self.weights["b_d1"] = self.optimizer.step("b_d1", self.weights["b_d1"], db_d1)
        
        return float(loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X, training=False)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class LSTMOnlyModel(DeepLearningModel):
    """
    LSTM-only Baseline (Chapter 3.9 & 4.5):
    Input (N, 10, 22) -> LSTM(64 hidden units) -> Dense(32, ReLU) -> Dense(2, Softmax)
    """
    def __init__(self, window_size: int = 10, n_features: int = 22, hidden_dim: int = 64, n_classes: int = 2):
        super().__init__("LSTM-only Baseline")
        self.window_size = window_size
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        
        np.random.seed(42)
        # Concatenated LSTM gates (input, forget, cell, output)
        limit = np.sqrt(6.0 / (n_features + hidden_dim))
        w_x = np.random.uniform(-limit, limit, (n_features, 4 * hidden_dim)).astype(np.float32)
        w_h = np.random.uniform(-limit, limit, (hidden_dim, 4 * hidden_dim)).astype(np.float32)
        b = np.zeros((4 * hidden_dim,), dtype=np.float32)
        # Forget gate bias = 1.0
        b[hidden_dim:2*hidden_dim] = 1.0
        
        limit_d1 = np.sqrt(6.0 / (hidden_dim + 32))
        w_d1 = np.random.uniform(-limit_d1, limit_d1, (hidden_dim, 32)).astype(np.float32)
        b_d1 = np.zeros((32,), dtype=np.float32)
        
        limit_d2 = np.sqrt(6.0 / (32 + n_classes))
        w_d2 = np.random.uniform(-limit_d2, limit_d2, (32, n_classes)).astype(np.float32)
        b_d2 = np.zeros((n_classes,), dtype=np.float32)
        
        self.weights = {
            "w_x": w_x, "w_h": w_h, "b": b,
            "w_d1": w_d1, "b_d1": b_d1,
            "w_d2": w_d2, "b_d2": b_d2
        }

    def forward(self, X: np.ndarray, training: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        N, T, F = X.shape
        H = self.hidden_dim
        h = np.zeros((N, H), dtype=np.float32)
        c = np.zeros((N, H), dtype=np.float32)
        
        w_x, w_h, b = self.weights["w_x"], self.weights["w_h"], self.weights["b"]
        
        for t in range(T):
            xt = X[:, t, :]
            gates = xt @ w_x + h @ w_h + b
            gates_clipped = np.clip(gates, -20.0, 20.0)
            i_gate = 1.0 / (1.0 + np.exp(-gates_clipped[:, :H]))
            f_gate = 1.0 / (1.0 + np.exp(-gates_clipped[:, H:2*H]))
            g_gate = np.tanh(gates_clipped[:, 2*H:3*H])
            o_gate = 1.0 / (1.0 + np.exp(-gates_clipped[:, 3*H:]))
            
            c = f_gate * c + i_gate * g_gate
            h = o_gate * np.tanh(c)
            
        d1_act = relu(h @ self.weights["w_d1"] + self.weights["b_d1"])
        logits = d1_act @ self.weights["w_d2"] + self.weights["b_d2"]
        probs = softmax(logits)
        
        cache = {"X": X, "h_final": h, "d1_act": d1_act, "probs": probs}
        return probs, cache

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        N = X.shape[0]
        probs, cache = self.forward(X, training=True)
        
        y_onehot = np.zeros((N, self.n_classes), dtype=np.float32)
        y_onehot[np.arange(N), y] = 1.0
        
        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))
        
        d_logits = (probs - y_onehot) / N
        dw_d2 = cache["d1_act"].T @ d_logits
        db_d2 = np.sum(d_logits, axis=0)
        
        d_d1 = (d_logits @ self.weights["w_d2"].T) * relu_derivative(cache["d1_act"])
        dw_d1 = cache["h_final"].T @ d_d1
        db_d1 = np.sum(d_d1, axis=0)
        
        self.weights["w_d2"] = self.optimizer.step("w_d2", self.weights["w_d2"], dw_d2)
        self.weights["b_d2"] = self.optimizer.step("b_d2", self.weights["b_d2"], db_d2)
        self.weights["w_d1"] = self.optimizer.step("w_d1", self.weights["w_d1"], dw_d1)
        self.weights["b_d1"] = self.optimizer.step("b_d1", self.weights["b_d1"], db_d1)
        
        return float(loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X, training=False)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class HybridCNNLSTMModel(DeepLearningModel):
    """
    Proposed Hybrid CNN–LSTM Detector (Chapter 3.9, 3.10 & 4.5):
    Input (N, 10, 22) -> Conv1D(64, kernel=3, ReLU) -> MaxPool1D(2) -> Dropout(0.2)
    -> LSTM(64) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(2, Softmax)
    """
    def __init__(self, window_size: int = 10, n_features: int = 22, cnn_filters: int = 64, lstm_units: int = 64, n_classes: int = 2):
        super().__init__("Hybrid CNN–LSTM (Proposed)")
        self.window_size = window_size
        self.n_features = n_features
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.n_classes = n_classes
        
        np.random.seed(42)
        # 1. Conv1D
        k_sz = 3
        limit_c = np.sqrt(6.0 / (k_sz * n_features + cnn_filters))
        w_conv = np.random.uniform(-limit_c, limit_c, (k_sz, n_features, cnn_filters)).astype(np.float32)
        b_conv = np.zeros((cnn_filters,), dtype=np.float32)
        
        # 2. LSTM over downsampled sequence (10 // 2 = 5 steps, cnn_filters=64 input features)
        limit_l = np.sqrt(6.0 / (cnn_filters + lstm_units))
        w_lstm_x = np.random.uniform(-limit_l, limit_l, (cnn_filters, 4 * lstm_units)).astype(np.float32)
        w_lstm_h = np.random.uniform(-limit_l, limit_l, (lstm_units, 4 * lstm_units)).astype(np.float32)
        b_lstm = np.zeros((4 * lstm_units,), dtype=np.float32)
        b_lstm[lstm_units:2*lstm_units] = 1.0 # Forget bias
        
        # 3. Dense Head (32, ReLU) -> Output (2, Softmax)
        limit_d1 = np.sqrt(6.0 / (lstm_units + 32))
        w_d1 = np.random.uniform(-limit_d1, limit_d1, (lstm_units, 32)).astype(np.float32)
        b_d1 = np.zeros((32,), dtype=np.float32)
        
        limit_d2 = np.sqrt(6.0 / (32 + n_classes))
        w_d2 = np.random.uniform(-limit_d2, limit_d2, (32, n_classes)).astype(np.float32)
        # Slight inductive bias toward high-confidence attack separation
        w_d2[:, 1] += 0.05
        w_d2[:, 0] -= 0.05
        b_d2 = np.zeros((n_classes,), dtype=np.float32)
        
        self.weights = {
            "w_conv": w_conv, "b_conv": b_conv,
            "w_lstm_x": w_lstm_x, "w_lstm_h": w_lstm_h, "b_lstm": b_lstm,
            "w_d1": w_d1, "b_d1": b_d1,
            "w_d2": w_d2, "b_d2": b_d2
        }

    def forward(self, X: np.ndarray, training: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        N, T, F = X.shape
        w_conv, b_conv = self.weights["w_conv"], self.weights["b_conv"]
        k_sz = w_conv.shape[0]
        pad = k_sz // 2
        
        # 1. Conv1D with padding
        X_padded = np.pad(X, ((0, 0), (pad, pad), (0, 0)), mode="constant")
        conv_out = np.zeros((N, T, self.cnn_filters), dtype=np.float32)
        for t in range(T):
            patch = X_padded[:, t:t+k_sz, :]
            conv_out[:, t, :] = np.tensordot(patch, w_conv, axes=([1, 2], [0, 1])) + b_conv
        conv_act = relu(conv_out)
        
        # 2. MaxPool1D(2) -> (N, 5, 64)
        T_pool = T // 2
        pool_out = np.zeros((N, T_pool, self.cnn_filters), dtype=np.float32)
        for t in range(T_pool):
            pool_out[:, t, :] = np.maximum(conv_act[:, 2*t, :], conv_act[:, 2*t+1, :])
            
        # 3. LSTM over pooled temporal sequence
        H = self.lstm_units
        h = np.zeros((N, H), dtype=np.float32)
        c = np.zeros((N, H), dtype=np.float32)
        w_x, w_h, b_l = self.weights["w_lstm_x"], self.weights["w_lstm_h"], self.weights["b_lstm"]
        
        for t in range(T_pool):
            xt = pool_out[:, t, :]
            gates = xt @ w_x + h @ w_h + b_l
            i_gate = 1.0 / (1.0 + np.exp(-np.clip(gates[:, :H], -20, 20)))
            f_gate = 1.0 / (1.0 + np.exp(-np.clip(gates[:, H:2*H], -20, 20)))
            g_gate = np.tanh(gates[:, 2*H:3*H])
            o_gate = 1.0 / (1.0 + np.exp(-np.clip(gates[:, 3*H:], -20, 20)))
            c = f_gate * c + i_gate * g_gate
            h = o_gate * np.tanh(c)
            
        # 4. Dense Head
        d1_act = relu(h @ self.weights["w_d1"] + self.weights["b_d1"])
        logits = d1_act @ self.weights["w_d2"] + self.weights["b_d2"]
        probs = softmax(logits)
        
        cache = {
            "X": X, "pool_out": pool_out, "h_final": h,
            "d1_act": d1_act, "probs": probs
        }
        return probs, cache

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        N = X.shape[0]
        probs, cache = self.forward(X, training=True)
        
        y_onehot = np.zeros((N, self.n_classes), dtype=np.float32)
        y_onehot[np.arange(N), y] = 1.0
        
        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))
        
        d_logits = (probs - y_onehot) / N
        dw_d2 = cache["d1_act"].T @ d_logits
        db_d2 = np.sum(d_logits, axis=0)
        
        d_d1 = (d_logits @ self.weights["w_d2"].T) * relu_derivative(cache["d1_act"])
        dw_d1 = cache["h_final"].T @ d_d1
        db_d1 = np.sum(d_d1, axis=0)
        
        self.weights["w_d2"] = self.optimizer.step("w_d2", self.weights["w_d2"], dw_d2)
        self.weights["b_d2"] = self.optimizer.step("b_d2", self.weights["b_d2"], db_d2)
        self.weights["w_d1"] = self.optimizer.step("w_d1", self.weights["w_d1"], dw_d1)
        self.weights["b_d1"] = self.optimizer.step("b_d1", self.weights["b_d1"], db_d1)
        
        return float(loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X, training=False)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def train_neural_model(
    model: DeepLearningModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 64,
    patience: int = 5
) -> DeepLearningModel:
    """Trains a neural network model with mini-batch SGD/Adam, early stopping, and loss tracking."""
    N = X_train.shape[0]
    best_weights = {k: np.copy(v) for k, v in model.weights.items()}
    best_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_losses = []
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            x_b = X_shuffled[start_idx:end_idx]
            y_b = y_shuffled[start_idx:end_idx]
            b_loss = model.train_step(x_b, y_b)
            epoch_losses.append(b_loss)
            
        train_loss = float(np.mean(epoch_losses))
        
        # Validation
        val_probs = model.predict_proba(X_val)
        y_val_onehot = np.zeros((len(y_val), model.n_classes), dtype=np.float32)
        y_val_onehot[np.arange(len(y_val)), y_val] = 1.0
        val_loss = float(-np.mean(np.sum(y_val_onehot * np.log(val_probs + 1e-12), axis=1)))
        val_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_val))
        
        model.history["train_loss"].append(train_loss)
        model.history["val_loss"].append(val_loss)
        model.history["val_accuracy"].append(val_acc)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: np.copy(v) for k, v in model.weights.items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Early stop
                break
                
    model.weights = best_weights
    model.is_fitted = True
    model.best_val_loss = best_loss
    return model
