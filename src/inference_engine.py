"""
src/inference_engine.py - Lightweight Edge Inference Engine using ai-edge-litert
Executes single-threaded CPU inference with sub-millisecond latency and sub-50MB RSS memory.
"""

import time
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from src.config import TFLITE_MODEL_PATH, WINDOW_SIZE, PCA_COMPONENTS

class EdgeInferenceEngine:
    """Wrapper around ai-edge-litert / TFLite interpreter for real-time edge threat detection."""
    
    def __init__(self, model_path: Optional[Path] = None, num_threads: int = 1):
        self.model_path = Path(model_path or TFLITE_MODEL_PATH)
        self.num_threads = num_threads
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_interpreter()
        
    def _load_interpreter(self) -> None:
        """Loads TFLite model using ai-edge-litert or compatible lightweight runtime."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found at: {self.model_path}")
            
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
                
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
            "num_threads": self.num_threads
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
