"""
tests/test_quantization.py - Unit & Edge Optimization Tests for TFLite FlatBuffer Quantization
Verifies 8-bit dynamic-range quantization, FlatBuffer magic header, tensor signatures,
NFR2 size constraints (<= 150 KB target), and numerical inference sanity.
"""

import pytest
import numpy as np
from pathlib import Path
from ai_edge_litert.interpreter import Interpreter

from src.quantization import quantize_and_export_hybrid_model
from src.config import TFLITE_MODEL_PATH, NFR_TARGETS

@pytest.fixture(scope="module")
def exported_tflite_model(tmp_path_factory):
    """Exports a fresh quantized hybrid TFLite FlatBuffer model to a temp path for testing."""
    tmp_dir = tmp_path_factory.mktemp("quant_test")
    out_path = tmp_dir / "test_hybrid_model.tflite"
    
    meta = quantize_and_export_hybrid_model(
        output_tflite_path=out_path,
        window_size=10,
        n_features=22
    )
    return out_path, meta

def test_quantize_and_export_hybrid_model_creation(exported_tflite_model):
    out_path, meta = exported_tflite_model
    assert out_path.exists()
    assert meta["format"] == "TensorFlow Lite FlatBuffer (TFL3)"
    assert meta["input_shape"] == [1, 10, 22]
    assert meta["output_shape"] == [1, 2]
    assert meta["nfr2_compliant"] is True

def test_tflite_flatbuffer_magic_header(exported_tflite_model):
    """Verifies that the serialized FlatBuffer includes standard TensorFlow Lite magic identifier."""
    out_path, _ = exported_tflite_model
    with open(out_path, "rb") as f:
        data = f.read(16)
    # In FlatBuffers, offset 4..8 stores the 4-byte file identifier "TFL3"
    magic_id = data[4:8]
    assert magic_id == b"TFL3", f"Expected magic 'TFL3', got {magic_id}"

def test_tflite_file_size_nfr2_contract(exported_tflite_model):
    """Verifies compliance against NFR2 (<= 1000 KB ceiling, <= 150 KB dissertation target)."""
    out_path, meta = exported_tflite_model
    size_kb = out_path.stat().st_size / 1024.0
    
    # Must be strictly under NFR2 ceiling
    assert size_kb <= NFR_TARGETS["NFR2_MAX_MODEL_SIZE_KB"]
    # Must be under our dissertation optimization target (150 KB)
    assert size_kb <= 150.0, f"Quantized model size {size_kb:.2f} KB exceeds 150 KB target."
    assert size_kb == pytest.approx(meta["file_size_kb"], abs=0.5)

def test_tflite_interpreter_tensor_signatures(exported_tflite_model):
    """Verifies tensor shapes and types using the standalone ai-edge-litert runtime."""
    out_path, _ = exported_tflite_model
    interp = Interpreter(model_path=str(out_path))
    interp.allocate_tensors()
    
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()
    
    assert len(in_details) == 1
    assert len(out_details) == 1
    
    np.testing.assert_array_equal(in_details[0]["shape"], [1, 10, 22])
    np.testing.assert_array_equal(out_details[0]["shape"], [1, 2])
    assert in_details[0]["dtype"] == np.float32
    assert out_details[0]["dtype"] == np.float32

def test_tflite_inference_numerical_sanity(exported_tflite_model):
    """Verifies that forward pass outputs are valid probability distributions in [0, 1] summing to 1.0."""
    out_path, _ = exported_tflite_model
    interp = Interpreter(model_path=str(out_path))
    interp.allocate_tensors()
    
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()
    
    np.random.seed(42)
    sample_input = np.random.randn(1, 10, 22).astype(np.float32)
    
    interp.set_tensor(in_details[0]["index"], sample_input)
    interp.invoke()
    output = interp.get_tensor(out_details[0]["index"])
    
    assert output.shape == (1, 2)
    assert not np.isnan(output).any()
    assert (output >= 0.0).all() and (output <= 1.0).all()
    assert np.isclose(np.sum(output), 1.0, atol=1e-4)

def test_tflite_deterministic_inference(exported_tflite_model):
    """Verifies that repeated inference on identical inputs produces bit-exact deterministic outputs."""
    out_path, _ = exported_tflite_model
    interp = Interpreter(model_path=str(out_path))
    interp.allocate_tensors()
    
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()
    
    sample_input = np.ones((1, 10, 22), dtype=np.float32) * 0.5
    
    interp.set_tensor(in_details[0]["index"], sample_input)
    interp.invoke()
    out1 = interp.get_tensor(out_details[0]["index"]).copy()
    
    interp.set_tensor(in_details[0]["index"], sample_input)
    interp.invoke()
    out2 = interp.get_tensor(out_details[0]["index"]).copy()
    
    np.testing.assert_array_equal(out1, out2)
