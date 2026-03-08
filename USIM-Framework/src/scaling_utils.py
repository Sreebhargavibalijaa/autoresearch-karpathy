"""
Scaling Strategy and GPU Mapping Module
"""
def estimate_gpu_requirements(n_agents, memory_per_agent_mb=500, gpu_vram_mb=24000):
    agents_per_gpu = gpu_vram_mb // memory_per_agent_mb
    n_gpus = (n_agents + agents_per_gpu - 1) // agents_per_gpu
    return n_gpus

def scaling_strategy(n_agents, memory_per_agent_mb=500, gpu_vram_mb=24000):
    if n_agents < 2:
        return f"Use single GPU. Swarm intelligence is limited to {n_agents} agents."
    elif n_agents <= 20:
        n_gpus = estimate_gpu_requirements(n_agents, memory_per_agent_mb, gpu_vram_mb)
        return f"Use {n_gpus} GPU(s) with memory optimization. Swarm intelligence shared among {n_agents} agents."
    else:
        n_gpus = estimate_gpu_requirements(n_agents, memory_per_agent_mb, gpu_vram_mb)
        return f"Distributed setup required. Use {n_gpus} GPU(s). Swarm intelligence shared among {n_agents} agents."
