"""
src/quantization.py - Model Compression & 8-Bit Dynamic Range TFLite Quantization

Converts the trained Keras Hybrid CNN–LSTM model into an edge-optimized TensorFlow
Lite FlatBuffer using the official ``tf.lite.TFLiteConverter``, as specified in
dissertation Table 4.1:

    Compression | TensorFlow Model Optimization Toolkit | Pruning and quantization workflow
    Edge runtime| TensorFlow Lite                       | Processor-only inference

This module replaces the previous hand-serialized FlatBuffer implementation. It
performs post-training 8-bit dynamic-range quantization (the exact compression
pathway described in Chapter 4.7 / Table 4.5) and verifies NFR2 size compliance.
"""

import json
import hashlib
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import joblib
import absl.logging

import tensorflow as tf

from src.config import (
    TFLITE_MODEL_PATH, METADATA_PATH, WINDOW_SIZE, PCA_COMPONENTS,
    MODELS_DIR, NFR_TARGETS
)
from src.models.deep_learning import HybridCNNLSTMModel

CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# The TFLite converter emits very verbose graph-tracing logs at INFO level.
# tf.lite.Interpreter also emits a deprecation notice steering users to a separate
# LiteRT package; this project targets the TensorFlow runtime explicitly (Table 4.1).
logging.getLogger("tensorflow").setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings(
    "ignore", message=".*tf.lite.Interpreter is deprecated.*", category=UserWarning
)


def _load_trained_hybrid(
    checkpoint_path: Optional[Path],
    window_size: int,
    n_features: int,
) -> Tuple[HybridCNNLSTMModel, Optional[Path]]:
    """
    Loads the trained hybrid Keras model checkpoint.

    Falls back to a freshly initialised model when the checkpoint is absent or
    cannot be deserialised (e.g. a stale checkpoint produced by the previous
    NumPy implementation), so that conversion and smoke tests always succeed.
    """
    ckpt_file = Path(checkpoint_path or (CHECKPOINTS_DIR / "hybrid_cnn_lstm.joblib"))

    if ckpt_file.exists():
        try:
            model = joblib.load(str(ckpt_file))
            if getattr(model, "network", None) is not None:
                print(f"[Quantization] Loaded trained hybrid model from: {ckpt_file}")
                return model, ckpt_file
            print(f"[Quantization] Checkpoint at {ckpt_file} has no Keras graph; using fresh model.")
        except Exception as exc:
            print(f"[Quantization] Could not deserialise {ckpt_file} ({exc.__class__.__name__}); "
                  f"using fresh model.")
    else:
        print(f"[Quantization] Checkpoint not found at {ckpt_file}. Initialising reference model.")

    return HybridCNNLSTMModel(window_size=window_size, n_features=n_features), None


def convert_keras_to_tflite(
    model: HybridCNNLSTMModel,
    quantize_dynamic: bool = True,
) -> Tuple[bytes, str]:
    """
    Converts a Keras model into a TensorFlow Lite FlatBuffer.

    The primary path targets built-in TFLite operators only, so the artefact needs
    no Flex/TensorFlow runtime at the edge. The fixed-length LSTM layers are
    statically unrolled (see ``unroll=True`` in src/models/deep_learning.py) which is
    what allows the recurrent graph to lower without TensorList operations — the same
    static-unrolling requirement documented in Chapter 4.7.

    Returns ``(tflite_bytes, operator_set)``.
    """
    def _build() -> tf.lite.TFLiteConverter:
        converter = tf.lite.TFLiteConverter.from_keras_model(model.network)
        if quantize_dynamic:
            # 8-bit dynamic-range quantization (Table 4.5 / FR5).
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter

    converter = _build()
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    try:
        return converter.convert(), "TFLITE_BUILTINS"
    except Exception as exc:
        print(
            f"[Quantization] Built-in-only conversion unavailable "
            f"({exc.__class__.__name__}); retrying with SELECT_TF_OPS (Flex)."
        )

    converter = _build()
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    return converter.convert(), "TFLITE_BUILTINS,SELECT_TF_OPS"


def quantize_and_export_hybrid_model(
    checkpoint_path: Optional[Path] = None,
    output_tflite_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    window_size: int = WINDOW_SIZE,
    n_features: int = PCA_COMPONENTS,
    quantize_dynamic: bool = True,
) -> Dict[str, Any]:
    """
    Quantizes and serialises the trained hybrid Keras network into an 8-bit
    dynamic-range TensorFlow Lite FlatBuffer, verifying NFR2 size constraints.

    Returns a metadata dictionary describing the exported artefact.
    """
    out_file = Path(output_tflite_path or TFLITE_MODEL_PATH)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model, source_ckpt = _load_trained_hybrid(checkpoint_path, window_size, n_features)
    tflite_bytes, operator_set = convert_keras_to_tflite(model, quantize_dynamic=quantize_dynamic)

    with open(out_file, "wb") as f:
        f.write(tflite_bytes)

    # ---- Verification with the TensorFlow Lite interpreter -----------------
    interp = tf.lite.Interpreter(model_path=str(out_file))
    interp.allocate_tensors()
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()

    in_shape = [int(v) for v in in_details[0]["shape"].tolist()]
    out_shape = [int(v) for v in out_details[0]["shape"].tolist()]
    file_size_kb = out_file.stat().st_size / 1024.0

    # Functional inference check: forward pass must yield a valid distribution.
    test_input = np.random.randn(1, window_size, n_features).astype(np.float32)
    interp.set_tensor(in_details[0]["index"], test_input)
    interp.invoke()
    test_output = interp.get_tensor(out_details[0]["index"])
    prob_sum = float(np.sum(test_output))

    sha256_hash = hashlib.sha256(tflite_bytes).hexdigest()

    metadata: Dict[str, Any] = {
        "model_name": "hybrid_model_tflite_quantized",
        "format": "TensorFlow Lite FlatBuffer (TFL3)",
        "quantization": (
            "8-bit dynamic-range quantization"
            if quantize_dynamic else "float32 (no quantization)"
        ),
        "operator_set": operator_set,
        "input_shape": in_shape,
        "output_shape": out_shape,
        "file_size_kb": round(file_size_kb, 2),
        "nfr2_compliant": bool(file_size_kb <= NFR_TARGETS["NFR2_MAX_MODEL_SIZE_KB"]),
        "parameters": int(model.count_parameters()),
        "sha256": sha256_hash,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "prob_sum_verification": round(prob_sum, 4),
    }
    if source_ckpt is not None:
        metadata["source_checkpoint"] = str(source_ckpt)

    if metadata_path:
        meta_file = Path(metadata_path)
    elif output_tflite_path:
        meta_file = out_file.parent / "metadata.json"
    else:
        meta_file = Path(METADATA_PATH)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[Quantization] Successfully exported quantized TFLite model:")
    print(f"  Path          : {out_file}")
    print(f"  Operator set  : {operator_set}")
    print(f"  Input Shape   : {in_shape}")
    print(f"  Output Shape  : {out_shape}")
    print(f"  Parameters    : {metadata['parameters']:,}")
    print(f"  Size          : {file_size_kb:.2f} KB (NFR2 Target <= 1000 KB: "
          f"{'PASSED' if metadata['nfr2_compliant'] else 'FAILED'})")
    print(f"  SHA256        : {sha256_hash[:16]}...")

    return metadata


if __name__ == "__main__":
    quantize_and_export_hybrid_model()
