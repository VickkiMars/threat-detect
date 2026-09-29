"""
src/inference_engine.py - Lightweight Edge Inference Engine (TensorFlow Lite)
Executes single-threaded CPU inference with sub-millisecond latency and sub-50MB RSS memory.

Build-time vs run-time split
----------------------------
The hybrid CNN–LSTM detector is *trained and converted* with TensorFlow/Keras and
exported with TensorFlow Lite **built-in** operators only (see src/quantization.py,
dissertation Table 4.1). Because the artefact contains no Flex/TensorList operators,
it executes identically on TensorFlow, LiteRT and tflite_runtime — verified
bit-for-bit across 500 random windows.

Runtime selection therefore follows the memory budget (NFR3), not model
compatibility: the service prefers a standalone interpreter (~48 MB RSS) and falls
back to TensorFlow (~549 MB RSS) only when no lightweight runtime is installed.
"""

import importlib
import time
import warnings
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from src.config import TFLITE_MODEL_PATH, WINDOW_SIZE, PCA_COMPONENTS

# TensorFlow 2.21 emits a deprecation notice steering users to the standalone
# LiteRT package. It only applies to the tf.lite fallback path.
warnings.filterwarnings(
    "ignore",
    message=".*tf.lite.Interpreter is deprecated.*",
    category=UserWarning,
)

# Standalone TFLite runtimes, ordered lightest-first. The exported graph uses only
# built-in operators, so any of these can execute it.
_STANDALONE_RUNTIMES = (
    ("ai_edge_litert.interpreter", "ai-edge-litert"),
    ("tflite_runtime.interpreter", "tflite-runtime"),
)


def _select_interpreter():
    """Returns ``(Interpreter class, runtime label)`` for the lightest available runtime."""
    for module_name, label in _STANDALONE_RUNTIMES:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "Interpreter"), label
        except (ImportError, AttributeError):
            continue

    import tensorflow as tf
    return tf.lite.Interpreter, "tensorflow"


class EdgeInferenceEngine:
    """Wrapper around the TensorFlow Lite interpreter for real-time edge threat detection."""
    
    def __init__(self, model_path: Optional[Path] = None, num_threads: int = 1):
        self.model_path = Path(model_path or TFLITE_MODEL_PATH)
        self.num_threads = num_threads
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.runtime = None
        self._load_interpreter()
        
    def _load_interpreter(self) -> None:
        """
        Loads the TFLite FlatBuffer model with the lightest available runtime.

        Prefers a standalone interpreter (ai-edge-litert, then tflite_runtime) to
        satisfy the 512 MB NFR3 memory ceiling, falling back to the TensorFlow-shipped
        ``tf.lite.Interpreter`` when neither is installed. All three execute the
        exported built-in-operator graph with identical results.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found at: {self.model_path}")

        Interpreter, self.runtime = _select_interpreter()

        self.interpreter = Interpreter(
            model_path=str(self.model_path),
            num_threads=self.num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def get_model_metadata(self) -> Dict[str, Any]:
        """Returns input/output tensor specifications and file size."""
        in_shape = self.input_details[0]["shape"].tolist()
        in_dtype = str(self.input_details[0]["dtype"])
        out_shape = self.output_details[0]["shape"].tolist()
        out_dtype = str(self.output_details[0]["dtype"])
        size_kb = self.model_path.stat().st_size / 1024.0
        
        return {
            "model_path": str(self.model_path),
            "file_size_kb": round(size_kb, 2),
            "input_shape": in_shape,
            "input_dtype": in_dtype,
            "output_shape": out_shape,
            "output_dtype": out_dtype,
            "num_threads": self.num_threads,
            "runtime": self.runtime
        }

    def predict_window(self, sequence_tensor: np.ndarray) -> Tuple[int, float, float]:
        """
        Classifies a 10-flow sequence window.
        
        Parameters:
            sequence_tensor: Array of shape (10, num_features) or (1, 10, num_features).
            
        Returns:
            Tuple: (predicted_class [0: Benign, 1: Attack], confidence [0.0-1.0], latency_ms)
        """
        # Ensure 3D shape (1, 10, num_features)
        if sequence_tensor.ndim == 2:
            input_data = np.expand_dims(sequence_tensor, axis=0).astype(np.float32)
        elif sequence_tensor.ndim == 3:
            input_data = sequence_tensor.astype(np.float32)
        else:
            raise ValueError(f"Expected sequence of 2 or 3 dimensions, got shape {sequence_tensor.shape}")
            
        input_idx = self.input_details[0]["index"]
        output_idx = self.output_details[0]["index"]
        
        self.interpreter.set_tensor(input_idx, input_data)
        
        # High-resolution monotonic timing
        t_start = time.perf_counter_ns()
        self.interpreter.invoke()
        t_end = time.perf_counter_ns()
        
        latency_ms = (t_end - t_start) / 1_000_000.0
        output_data = self.interpreter.get_tensor(output_idx)[0]
        
        predicted_class = int(np.argmax(output_data))
        confidence = float(output_data[predicted_class])
        
        return predicted_class, confidence, latency_ms
