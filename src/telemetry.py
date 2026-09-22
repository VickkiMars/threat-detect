"""
src/telemetry.py - Hardware Profiling & Resource Metrics Instrumentation
Monitors process CPU %, Resident Set Size (RSS) memory, and flow throughput via psutil.
"""

import os
import time
import psutil
from typing import Dict, Any, Optional
from src.config import NFR_TARGETS
from src.database import record_metrics

class TelemetrySampler:
    """Samples and logs CPU, memory, and throughput metrics for the edge detection process."""
    
    def __init__(self, pid: Optional[int] = None):
        self.process = psutil.Process(pid or os.getpid())
        # Prime the CPU measurement
        self.process.cpu_percent(interval=None)
        self.last_sample_time = time.time()
        self.last_flow_count = 0
        
    def sample(self, current_flow_count: int = 0) -> Dict[str, Any]:
        """Collects current hardware metrics and logs them to SQLite."""
        now = time.time()
        elapsed = max(now - self.last_sample_time, 0.001)
        
        # CPU and Memory measurements
        cpu_pct = self.process.cpu_percent(interval=None)
        mem_info = self.process.memory_info()
        rss_mb = mem_info.rss / (1024.0 * 1024.0)
        
        # Throughput calculation
        flows_processed = max(0, current_flow_count - self.last_flow_count)
        throughput_fps = flows_processed / elapsed
        
        self.last_sample_time = now
        self.last_flow_count = current_flow_count
        
        # Determine host compliance against NFR3 (512 MB target)
        if rss_mb > NFR_TARGETS["NFR3_MAX_MEMORY_RSS_MB"]:
            host_status = f"CRITICAL_MEM_EXCEEDED (>{NFR_TARGETS['NFR3_MAX_MEMORY_RSS_MB']}MB)"
        elif rss_mb > 450.0:
            host_status = "WARNING_HIGH_MEM (>450MB)"
        else:
            host_status = "NORMAL"
            
        record_metrics(
            cpu_percent=round(cpu_pct, 2),
            memory_rss_mb=round(rss_mb, 2),
            throughput_fps=round(throughput_fps, 2),
            host_status=host_status
        )
        
        return {
            "cpu_percent": round(cpu_pct, 2),
            "memory_rss_mb": round(rss_mb, 2),
            "throughput_fps": round(throughput_fps, 2),
            "host_status": host_status
        }
