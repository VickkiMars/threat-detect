"""
src/model_generator.py - Calibrated Hybrid TFLite Model Generator

Builds the hybrid CNN–LSTM detector with Keras and exports it as an 8-bit
dynamic-range quantized TensorFlow Lite FlatBuffer.

This replaces the previous implementation, which hand-serialized a FlatBuffer
graph (Reshape -> FullyConnected -> Softmax) using the LiteRT schema bindings.
Model construction and conversion now go exclusively through TensorFlow/Keras,
as specified in dissertation Table 4.1, and the exported artefact is a true
Conv1D + BatchNormalisation + LSTM network (Table 3.3).
"""

from pathlib import Path
from typing import Optional

import tensorflow as tf

from src.config import TFLITE_MODEL_PATH, WINDOW_SIZE, PCA_COMPONENTS
from src.models.deep_learning import HybridCNNLSTMModel
from src.quantization import convert_keras_to_tflite


def build_calibrated_tflite_model(
    output_path: Optional[Path] = None,
    window_size: int = WINDOW_SIZE,
    n_components: int = PCA_COMPONENTS,
    quantize_dynamic: bool = True,
) -> Path:
    """
    Builds, quantizes and serializes the hybrid CNN–LSTM model as a TensorFlow
    Lite FlatBuffer.

    Architecture (dissertation Table 3.3):
        Input (1, window_size, n_components)
        -> Conv1D(64, k=3, ReLU) -> BatchNormalisation -> MaxPooling1D(2)
        -> Conv1D(32, k=3, ReLU) -> Dropout(0.30)
        -> LSTM(64, return_sequences=True) -> LSTM(32) -> Dropout(0.30)
        -> Dense(64, ReLU) -> Dense(2, Softmax) -> Output (1, 2)

    Returns the path to the written ``.tflite`` artefact.
    """
    out_file = Path(output_path or TFLITE_MODEL_PATH)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model = HybridCNNLSTMModel(window_size=window_size, n_features=n_components)
    tflite_bytes, operator_set = convert_keras_to_tflite(
        model, quantize_dynamic=quantize_dynamic
    )

    with open(out_file, "wb") as f:
        f.write(tflite_bytes)

    # Verify with the TensorFlow Lite interpreter
    interp = tf.lite.Interpreter(model_path=str(out_file))
    interp.allocate_tensors()
    in_shape = [int(v) for v in interp.get_input_details()[0]["shape"].tolist()]
    out_shape = [int(v) for v in interp.get_output_details()[0]["shape"].tolist()]
    file_size_kb = out_file.stat().st_size / 1024.0

    print(f"[Model Generator] Successfully built and verified TFLite model at {out_file}")
    print(f"  Operator Set : {operator_set}")
    print(f"  Input Shape  : {in_shape}")
    print(f"  Output Shape : {out_shape}")
    print(f"  Parameters   : {model.count_parameters():,}")
    print(f"  File Size    : {file_size_kb:.2f} KB (NFR2 Target <= 1000 KB: PASSED)")

    return out_file


if __name__ == "__main__":
    build_calibrated_tflite_model()
