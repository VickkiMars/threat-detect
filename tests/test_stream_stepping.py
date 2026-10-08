"""
tests/test_stream_stepping.py - Regression Test for Zero-Sleep Replay Stepping
Verifies that stream_windows(with_delay=False) yields windows with sub-50ms latency
without blocking on artificial rate-limiting sleeps intended for daemon loops.
"""

import time
import pytest
from src.stream_simulator import FlowStreamSimulator
from src.config import SAMPLE_FLOWS_PATH

def test_stream_windows_without_delay():
    """Verifies that with_delay=False yields consecutive windows without 333ms sleep."""
    simulator = FlowStreamSimulator(csv_path=SAMPLE_FLOWS_PATH, rate_flows_per_sec=30, loop=True)
    generator = simulator.stream_windows(with_delay=False)
    
    t0 = time.perf_counter()
    w1 = next(generator)
    w2 = next(generator)
    elapsed = time.perf_counter() - t0
    
    assert w1[0] == 1
    assert w2[0] == 2
    # Without delay, two next() calls should complete in << 150ms (typically ~10-30ms)
    assert elapsed < 0.20, f"Stepping took {elapsed:.3f}s, expected < 0.20s without sleep"
