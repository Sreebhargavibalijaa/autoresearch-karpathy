"""
Hardware Specification Module
"""
from dataclasses import dataclass

@dataclass
class HardwareSpecs:
    gpu_name: str
    tflops: float
    memory_gb: float
    bandwidth_gb: float
    gpu_count: int = 1

    @property
    def compute_units(self) -> float:
        base_tflops = 312
        base_memory = 80
        base_bandwidth = 1555
        compute_factor = (self.tflops / base_tflops) * 0.4
        memory_factor = (self.memory_gb / base_memory) * 0.3
        bandwidth_factor = (self.bandwidth_gb / base_bandwidth) * 0.3
        return compute_factor + memory_factor + bandwidth_factor
