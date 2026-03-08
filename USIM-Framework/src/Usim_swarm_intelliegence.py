"""
Universal Swarm Intelligence Metric (USIM) Framework
Run this on your A30, RTX6000, and A100 GPUs with real data input
"""

import torch
import numpy as np
import json
import time
import hashlib
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM JSON ENCODER
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GPUSpecs:
    """GPU specifications"""
    name: str
    memory_gb: float
    tflops: float
    bandwidth_gb: float
    
    @property
    def compute_units(self) -> float:
        """Convert to A100-equivalent compute units"""
        # A100 baseline
        base_tflops = 312.0
        base_memory = 80.0
        base_bandwidth = 1555.0
        
        compute_factor = (self.tflops / base_tflops) * 0.4
        memory_factor = (self.memory_gb / base_memory) * 0.3
        bandwidth_factor = (self.bandwidth_gb / base_bandwidth) * 0.3
        
        return float(compute_factor + memory_factor + bandwidth_factor)

@dataclass
class ExperimentData:
    """Raw experiment data input by user"""
    gpu_name: str
    n_agents: int
    duration_hours: float
    total_insights: int
    unique_insights: int
    shared_insights: int
    agent_focus_areas: List[List[str]]
    agent_compute_time: List[float]
    agent_comm_time: List[float]
    agent_memory_mb: List[float]
    insight_timestamps: List[float]
    insight_sharing: List[List[int]]

@dataclass
class USIMMetrics:
    """Calculated USIM metrics"""
    gpu_name: str
    n_agents: int
    
    # Core components
    knowledge_density: float
    swarm_coherence: float
    exploration_efficiency: float
    emergence_factor: float
    communication_overhead: float
    
    # Derived
    usim_score: float
    gpu_equivalents: float
    scalability_score: float
    karpathy_threshold: int
    
    # Intelligence level
    intelligence_level: str
    level_description: str

# ============================================================================
# GPU DETECTION
# ============================================================================

class GPUDetector:
    """Automatically detect connected GPUs"""
    
    GPU_DATABASE = {
        'A30': GPUSpecs('NVIDIA A30', 24.0, 165.0, 933.0),
        'RTX 6000': GPUSpecs('NVIDIA RTX 6000', 24.0, 130.0, 672.0),
        'A100': GPUSpecs('NVIDIA A100', 80.0, 312.0, 1555.0),
        'A100 40GB': GPUSpecs('NVIDIA A100 40GB', 40.0, 312.0, 1555.0),
        'A100 80GB': GPUSpecs('NVIDIA A100 80GB', 80.0, 312.0, 1555.0),
        'H100': GPUSpecs('NVIDIA H100', 80.0, 1979.0, 3350.0),
        'V100': GPUSpecs('NVIDIA V100', 32.0, 125.0, 900.0),
        'T4': GPUSpecs('NVIDIA T4', 16.0, 65.0, 320.0),
        'A10': GPUSpecs('NVIDIA A10', 24.0, 125.0, 600.0),
        'RTX 4090': GPUSpecs('NVIDIA RTX 4090', 24.0, 330.0, 1008.0),
    }
    
    @staticmethod
    def detect_gpus() -> List[Dict]:
        """Detect all available GPUs"""
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                name = props.name
                
                # Find matching GPU in database
                gpu_spec = None
                for key, spec in GPUDetector.GPU_DATABASE.items():
                    if key.lower() in name.lower():
                        gpu_spec = spec
                        break
                
                if gpu_spec is None:
                    # Create generic spec
                    gpu_spec = GPUSpecs(
                        name=name,
                        memory_gb=props.total_memory / 1e9,
                        tflops=props.multi_processor_count * props.max_threads_per_multiprocessor / 1e9,
                        bandwidth_gb=props.memory_bandwidth / 1e9 if hasattr(props, 'memory_bandwidth') else 500.0
                    )
                
                gpus.append({
                    'index': i,
                    'name': name,
                    'specs': gpu_spec,
                    'memory_gb': props.total_memory / 1e9,
                    'compute_capability': f"{props.major}.{props.minor}"
                })
        return gpus
    
    @staticmethod
    def get_gpu_spec(gpu_name: str) -> GPUSpecs:
        """Get specs for a specific GPU by name"""
        for key, spec in GPUDetector.GPU_DATABASE.items():
            if key.lower() in gpu_name.lower():
                return spec
        
        # Return generic if not found
        return GPUSpecs(gpu_name, 24.0, 150.0, 800.0)

# ============================================================================
# USIM CALCULATOR
# ============================================================================

class USIMCalculator:
    """Calculate USIM metrics from experimental data"""
    
    def __init__(self):
        self.max_knowledge_density = 1000.0
        self.max_emergence = 10.0
    
    def calculate(self, data: ExperimentData) -> USIMMetrics:
        """Calculate all USIM metrics from experiment data"""
        
        # Get GPU specs
        gpu_spec = GPUDetector.get_gpu_spec(data.gpu_name)
        
        # 1. Knowledge Density (Kd)
        compute_units = gpu_spec.compute_units * data.duration_hours
        knowledge_density = data.unique_insights / max(compute_units, 1e-8)
        knowledge_density = min(knowledge_density / self.max_knowledge_density, 1.0)
        
        # 2. Swarm Coherence (Sc)
        swarm_coherence = self._calc_coherence(data)
        
        # 3. Exploration Efficiency (Ee)
        exploration_efficiency = self._calc_exploration(data, compute_units)
        
        # 4. Emergence Factor (Ef)
        emergence_factor = self._calc_emergence(data)
        
        # 5. Communication Overhead (C)
        communication_overhead = self._calc_communication(data)
        
        # 6. USIM Score
        if data.n_agents > 0 and communication_overhead > 0:
            numerator = (knowledge_density * swarm_coherence * 
                        exploration_efficiency * emergence_factor)
            denominator = communication_overhead * np.sqrt(data.n_agents)
            usim_score = numerator / denominator if denominator > 0 else 0.0
        else:
            usim_score = 0.0
        
        # 7. GPU Equivalents
        gpu_equivalents = self._calc_gpu_requirements(
            data.n_agents,
            np.mean(data.agent_memory_mb) / 1024.0,
            knowledge_density,
            swarm_coherence,
            gpu_spec
        )
        
        # 8. Scalability Score
        scalability_score = self._calc_scalability(
            knowledge_density,
            communication_overhead,
            data.n_agents
        )
        
        # 9. Karpathy Threshold
        karpathy_threshold = self._find_karpathy_threshold(
            data.n_agents,
            knowledge_density,
            communication_overhead
        )
        
        # 10. Intelligence Level
        level, desc = self._get_intelligence_level(usim_score)
        
        return USIMMetrics(
            gpu_name=data.gpu_name,
            n_agents=data.n_agents,
            knowledge_density=float(knowledge_density),
            swarm_coherence=float(swarm_coherence),
            exploration_efficiency=float(exploration_efficiency),
            emergence_factor=float(emergence_factor),
            communication_overhead=float(communication_overhead),
            usim_score=float(usim_score),
            gpu_equivalents=float(gpu_equivalents),
            scalability_score=float(scalability_score),
            karpathy_threshold=int(karpathy_threshold),
            intelligence_level=level,
            level_description=desc
        )
    
    def _calc_coherence(self, data: ExperimentData) -> float:
        """Calculate swarm coherence"""
        if data.n_agents < 2:
            return 1.0
        
        # Focus overlap
        total_overlap = 0.0
        total_pairs = 0
        
        for i in range(data.n_agents):
            for j in range(i+1, data.n_agents):
                focus1 = set(data.agent_focus_areas[i])
                focus2 = set(data.agent_focus_areas[j])
                if focus1 and focus2:
                    overlap = len(focus1 & focus2) / max(len(focus1 | focus2), 1)
                    total_overlap += overlap
                total_pairs += 1
        
        focus_coherence = total_overlap / max(total_pairs, 1)
        
        # Sharing efficiency
        if data.total_insights > 0 and data.n_agents > 1:
            sharing_efficiency = data.shared_insights / (data.total_insights * (data.n_agents - 1))
        else:
            sharing_efficiency = 0.0
        
        return float(0.5 * focus_coherence + 0.5 * min(sharing_efficiency, 1.0))
    
    def _calc_exploration(self, data: ExperimentData, compute_units: float) -> float:
        """Calculate exploration efficiency"""
        if data.n_agents == 0 or compute_units == 0:
            return 0.0
        
        insights_per_agent = data.unique_insights / data.n_agents
        efficiency = insights_per_agent / compute_units
        
        return float(min(efficiency / 10.0, 1.0))
    
    def _calc_emergence(self, data: ExperimentData) -> float:
        """Calculate emergence factor"""
        if data.n_agents < 2 or data.duration_hours == 0:
            return 0.0
        
        # Individual rates (simplified)
        individual_rate = np.mean([c / max(t, 0.001) for c, t in 
                                  zip(data.agent_compute_time, data.agent_compute_time)])
        
        # Swarm rate
        swarm_rate = data.unique_insights / data.duration_hours
        
        # Emergence
        if individual_rate > 0:
            emergence = (swarm_rate - individual_rate * data.n_agents) / (individual_rate * data.n_agents)
        else:
            emergence = 0.0
        
        return float(min(max(emergence / self.max_emergence, 0.0), 1.0))
    
    def _calc_communication(self, data: ExperimentData) -> float:
        """Calculate communication overhead"""
        total_comm = sum(data.agent_comm_time)
        total_compute = sum(data.agent_compute_time)
        total_work = total_comm + total_compute
        
        if total_work == 0:
            return 0.0
        
        return float(min(total_comm / total_work, 1.0))
    
    def _calc_gpu_requirements(self, n_agents: int, memory_per_agent: float,
                              knowledge_density: float, swarm_coherence: float,
                              gpu_spec: GPUSpecs) -> float:
        """Calculate A100-equivalent GPUs needed"""
        if n_agents == 0 or swarm_coherence == 0:
            return 0.0
        
        # A100 baseline
        G = 80.0
        baseline_memory = 1.5
        
        memory_factor = memory_per_agent / baseline_memory
        density_factor = 1.0 + knowledge_density
        
        required = (n_agents * memory_per_agent * density_factor) / (G * swarm_coherence)
        required /= max(gpu_spec.compute_units, 0.1)
        
        return float(max(required, 0.1))
    
    def _calc_scalability(self, knowledge_density: float,
                         communication_cost: float, n_agents: int) -> float:
        """Calculate scalability score"""
        if n_agents == 0:
            return 0.0
        
        alpha = communication_cost * 10.0
        
        theoretical = np.log(n_agents + 1) * np.sqrt(1.0 - communication_cost) * np.exp(-alpha/n_agents)
        theoretical_max = np.log(100) * 1.0 * 1.0
        
        scalability = theoretical / theoretical_max if theoretical_max > 0 else 0.0
        return float(min(scalability, 1.0))
    
    def _find_karpathy_threshold(self, n_agents: int,
                                knowledge_density: float,
                                communication_cost: float) -> int:
        """Find where adding agents reduces intelligence"""
        thresholds = []
        for n in range(2, 50):
            if communication_cost > 0:
                i1 = np.log(n) * np.sqrt(1.0 - communication_cost) * np.exp(-communication_cost*10.0/n)
                i2 = np.log(n-1) * np.sqrt(1.0 - communication_cost) * np.exp(-communication_cost*10.0/(n-1))
                if i1 - i2 < 0:
                    thresholds.append(n)
        
        return thresholds[0] if thresholds else int(n_agents * 2)
    
    def _get_intelligence_level(self, usim: float) -> Tuple[str, str]:
        """Get intelligence level and description"""
        if usim < 0.3:
            return "BASIC", "Individual agents working independently"
        elif usim < 0.6:
            return "INTERMEDIATE", "Some collaboration, moderate synergy"
        elif usim < 0.8:
            return "ADVANCED", "Strong collaboration, significant emergence"
        else:
            return "EMERGENT", "True swarm intelligence, superlinear scaling"

# ============================================================================
# DATA INPUT INTERFACE
# ============================================================================

class ExperimentInput:
    """Interactive data input for experiments"""
    
    @staticmethod
    def get_gpu_selection(gpus: List[Dict]) -> Dict:
        """Let user select which GPU to analyze"""
        print("\n" + "="*60)
        print("AVAILABLE GPUs DETECTED:")
        print("="*60)
        
        for i, gpu in enumerate(gpus):
            print(f"\n[{i}] {gpu['name']}")
            print(f"    Memory: {gpu['memory_gb']:.1f} GB")
            print(f"    Compute Units: {gpu['specs'].compute_units:.2f} A100-equiv")
        
        while True:
            try:
                choice = int(input(f"\nSelect GPU (0-{len(gpus)-1}): "))
                if 0 <= choice < len(gpus):
                    return gpus[choice]
            except:
                pass
            print("Invalid selection, try again")
    
    @staticmethod
    def input_experiment_data(gpu_name: str) -> ExperimentData:
        """Input experimental data for a GPU"""
        print(f"\n" + "="*60)
        print(f"ENTER EXPERIMENT DATA FOR {gpu_name}")
        print("="*60)
        
        # Basic info
        n_agents = int(input("\nNumber of agents in swarm: "))
        duration = float(input("Experiment duration (hours): "))
        
        # Insights
        total_insights = int(input("Total insights generated: "))
        unique_insights = int(input("Unique insights (after deduplication): "))
        shared_insights = int(input("Total sharing events between agents: "))
        
        # Agent details
        print("\n--- AGENT DETAILS ---")
        agent_focus = []
        agent_compute = []
        agent_comm = []
        agent_memory = []
        
        for i in range(n_agents):
            print(f"\nAgent {i+1}:")
            focuses = input("  Focus areas (comma-separated, e.g., optimizer,architecture): ").split(',')
            agent_focus.append([f.strip() for f in focuses])
            
            compute = float(input("  Compute time (hours): "))
            agent_compute.append(compute)
            
            comm = float(input("  Communication time (hours): "))
            agent_comm.append(comm)
            
            memory = float(input("  Memory used (MB): "))
            agent_memory.append(memory)
        
        # Optional: Insight timestamps (simplified)
        print("\n--- INSIGHT TIMING (simplified) ---")
        use_timestamps = input("Do you want to input individual insight timestamps? (y/n): ").lower() == 'y'
        
        insight_times = []
        insight_sharing = []
        
        if use_timestamps and total_insights > 0:
            print(f"Enter timestamps for {total_insights} insights (in hours from start):")
            for i in range(total_insights):
                t = float(input(f"  Insight {i+1} timestamp: "))
                insight_times.append(t)
                
                # Sharing info
                shared_with = input(f"  Agents this was shared with (comma-separated IDs 0-{n_agents-1}): ")
                if shared_with.strip():
                    insight_sharing.append([int(x.strip()) for x in shared_with.split(',')])
                else:
                    insight_sharing.append([])
        else:
            # Generate synthetic timestamps
            insight_times = [i * duration / max(total_insights, 1) for i in range(total_insights)]
            insight_sharing = [[] for _ in range(total_insights)]
        
        return ExperimentData(
            gpu_name=gpu_name,
            n_agents=n_agents,
            duration_hours=duration,
            total_insights=total_insights,
            unique_insights=unique_insights,
            shared_insights=shared_insights,
            agent_focus_areas=agent_focus,
            agent_compute_time=agent_compute,
            agent_comm_time=agent_comm,
            agent_memory_mb=agent_memory,
            insight_timestamps=insight_times,
            insight_sharing=insight_sharing
        )

# ============================================================================
# RESULTS MANAGER
# ============================================================================

class ResultsManager:
    """Manage and compare results across GPUs"""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, gpu_name: str, n_agents: int, metrics: USIMMetrics):
        """Add a result for a GPU"""
        if gpu_name not in self.results:
            self.results[gpu_name] = []
        self.results[gpu_name].append({
            'n_agents': n_agents,
            'metrics': metrics
        })
    
    def save(self, filename: str = None):
        """Save all results to file"""
        if filename is None:
            filename = f"usim_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to serializable format
        data = {}
        for gpu_name, experiments in self.results.items():
            data[gpu_name] = []
            for exp in experiments:
                data[gpu_name].append({
                    'n_agents': exp['n_agents'],
                    'metrics': asdict(exp['metrics'])
                })
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'results': data
            }, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n✅ Results saved to {filename}")
        return filename
    
    def load(self, filename: str):
        """Load results from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    
    def compare_gpus(self) -> Dict:
        """Compare USIM across different GPUs"""
        comparison = {}
        
        for gpu_name, experiments in self.results.items():
            # Sort by number of agents
            experiments.sort(key=lambda x: x['n_agents'])
            
            comparison[gpu_name] = {
                'n_agents': [e['n_agents'] for e in experiments],
                'usim_scores': [e['metrics'].usim_score for e in experiments],
                'max_usim': max([e['metrics'].usim_score for e in experiments]),
                'optimal_agents': max(experiments, key=lambda x: x['metrics'].usim_score)['n_agents'],
                'karpathy_threshold': experiments[-1]['metrics'].karpathy_threshold
            }
        
        return comparison
    
    def generate_report(self) -> str:
        """Generate comparison report"""
        comparison = self.compare_gpus()
        
        report = []
        report.append("\n" + "="*80)
        report.append("CROSS-GPU USIM COMPARISON REPORT")
        report.append("="*80)
        
        for gpu_name, data in comparison.items():
            report.append(f"\n📊 {gpu_name}:")
            report.append(f"  Optimal agents: {data['optimal_agents']} (USIM={data['max_usim']:.4f})")
            report.append(f"  Karpathy threshold: {data['karpathy_threshold']} agents")
            report.append(f"  Scaling efficiency: {(data['max_usim'] / max(data['n_agents'])):.4f}")
        
        # Find best GPU for swarm intelligence
        best_gpu = max(comparison.items(), key=lambda x: x[1]['max_usim'])
        report.append(f"\n🏆 Best GPU for swarm intelligence: {best_gpu[0]}")
        report.append(f"   Max USIM: {best_gpu[1]['max_usim']:.4f}")
        
        # GPU equivalence
        report.append("\n" + "="*80)
        report.append("GPU INTELLIGENCE EQUIVALENCE (per A100 unit):")
        report.append("="*80)
        
        baseline = comparison.get('NVIDIA A100', {}).get('max_usim', 1.0)
        for gpu_name, data in comparison.items():
            if gpu_name != 'NVIDIA A100':
                ratio = data['max_usim'] / baseline if baseline > 0 else 0
                report.append(f"  {gpu_name}: {ratio:.2f}x A100 intelligence")
        
        return '\n'.join(report)
    
    def plot_comparison(self):
        """Plot USIM across GPUs"""
        comparison = self.compare_gpus()
        
        plt.figure(figsize=(12, 6))
        
        for gpu_name, data in comparison.items():
            plt.plot(data['n_agents'], data['usim_scores'], 'o-', linewidth=2, label=gpu_name)
        
        plt.xlabel('Number of Agents')
        plt.ylabel('USIM Score')
        plt.title('USIM Scaling Across Different GPUs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('usim_gpu_comparison.png', dpi=150)
        plt.show()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class USIMApp:
    """Main application"""
    
    def __init__(self):
        self.detector = GPUDetector()
        self.calculator = USIMCalculator()
        self.results_manager = ResultsManager()
        self.gpus = self.detector.detect_gpus()
    
    def run(self):
        """Main execution loop"""
        print("\n" + "="*80)
        print("UNIVERSAL SWARM INTELLIGENCE METRIC (USIM) FRAMEWORK")
        print("="*80)
        print("\nA hardware-agnostic metric for AI swarm intelligence")
        print("\nDetected GPUs:")
        for i, gpu in enumerate(self.gpus):
            print(f"  [{i}] {gpu['name']}")
        
        while True:
            print("\n" + "-"*40)
            print("MAIN MENU")
            print("-"*40)
            print("1. Input new experiment data")
            print("2. View results for all GPUs")
            print("3. Compare across GPUs")
            print("4. Save results")
            print("5. Load previous results")
            print("6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                self.input_experiment()
            elif choice == '2':
                self.view_results()
            elif choice == '3':
                self.compare()
            elif choice == '4':
                self.save()
            elif choice == '5':
                self.load()
            elif choice == '6':
                break
            else:
                print("Invalid option")
    
    def input_experiment(self):
        """Input new experiment data"""
        if not self.gpus:
            print("No GPUs detected!")
            return
        
        # Select GPU
        gpu = ExperimentInput.get_gpu_selection(self.gpus)
        
        # Input data
        data = ExperimentInput.input_experiment_data(gpu['name'])
        
        # Calculate metrics
        metrics = self.calculator.calculate(data)
        
        # Store result
        self.results_manager.add_result(gpu['name'], data.n_agents, metrics)
        
        # Display result
        self.display_metrics(metrics)
    
    def display_metrics(self, metrics: USIMMetrics):
        """Display USIM metrics"""
        print("\n" + "="*80)
        print(f"USIM RESULTS FOR {metrics.gpu_name} ({metrics.n_agents} agents)")
        print("="*80)
        
        print(f"\n📊 CORE METRICS:")
        print(f"  USIM Score: {metrics.usim_score:.4f}")
        print(f"  Knowledge Density (Kd): {metrics.knowledge_density:.4f}")
        print(f"  Swarm Coherence (Sc): {metrics.swarm_coherence:.4f}")
        print(f"  Exploration Efficiency (Ee): {metrics.exploration_efficiency:.4f}")
        print(f"  Emergence Factor (Ef): {metrics.emergence_factor:.4f}")
        print(f"  Communication Overhead (C): {metrics.communication_overhead:.4f}")
        
        print(f"\n🐝 SWARM CHARACTERISTICS:")
        print(f"  Intelligence Level: {metrics.intelligence_level}")
        print(f"  Description: {metrics.level_description}")
        print(f"  Scalability Score: {metrics.scalability_score:.4f}")
        print(f"  Karpathy Threshold: {metrics.karpathy_threshold} agents")
        
        print(f"\n🎮 GPU REQUIREMENTS:")
        print(f"  Current GPU Equivalents: {metrics.gpu_equivalents:.2f} A100-equivalent")
    
    def view_results(self):
        """View all results"""
        if not self.results_manager.results:
            print("No results yet")
            return
        
        for gpu_name, experiments in self.results_manager.results.items():
            print(f"\n{'-'*40}")
            print(f"GPU: {gpu_name}")
            print(f"{'-'*40}")
            
            for exp in experiments:
                m = exp['metrics']
                print(f"  {exp['n_agents']} agents: USIM={m.usim_score:.4f} ({m.intelligence_level})")
    
    def compare(self):
        """Compare across GPUs"""
        if len(self.results_manager.results) < 2:
            print("Need at least 2 GPUs with results to compare")
            return
        
        report = self.results_manager.generate_report()
        print(report)
        
        # Plot
        plot = input("\nGenerate comparison plot? (y/n): ").lower() == 'y'
        if plot:
            self.results_manager.plot_comparison()
    
    def save(self):
        """Save results"""
        if not self.results_manager.results:
            print("No results to save")
            return
        
        filename = input("Enter filename (or press Enter for auto-generated): ").strip()
        if not filename:
            self.results_manager.save()
        else:
            self.results_manager.save(filename)
    
    def load(self):
        """Load previous results"""
        filename = input("Enter filename to load: ").strip()
        try:
            data = self.results_manager.load(filename)
            print(f"Loaded results from {filename}")
            # Reconstruct results (simplified)
        except Exception as e:
            print(f"Error loading file: {e}")

# ============================================================================
# QUICK TEST FUNCTION
# ============================================================================

def quick_test():
    """Quick test with your actual A30 data"""
    
    # Your actual A30 data from the runs
    test_data = [
        # A30 with 1 agent
        ExperimentData(
            gpu_name="NVIDIA A30",
            n_agents=1,
            duration_hours=1.0,
            total_insights=22,
            unique_insights=20,
            shared_insights=0,
            agent_focus_areas=[['optimizer', 'architecture', 'training']],
            agent_compute_time=[0.95],
            agent_comm_time=[0.05],
            agent_memory_mb=[1532.0],
            insight_timestamps=[i*3 for i in range(22)],
            insight_sharing=[[] for _ in range(22)]
        ),
        # A30 with 2 agents
        ExperimentData(
            gpu_name="NVIDIA A30",
            n_agents=2,
            duration_hours=1.0,
            total_insights=39,
            unique_insights=36,
            shared_insights=5,
            agent_focus_areas=[['optimizer', 'architecture'], ['architecture', 'training']],
            agent_compute_time=[0.9, 0.85],
            agent_comm_time=[0.1, 0.15],
            agent_memory_mb=[1514.0, 1514.0],
            insight_timestamps=[i*1.5 for i in range(39)],
            insight_sharing=[[1] if i % 5 == 0 else [] for i in range(39)]
        ),
        # A30 with 4 agents
        ExperimentData(
            gpu_name="NVIDIA A30",
            n_agents=4,
            duration_hours=1.0,
            total_insights=89,
            unique_insights=80,
            shared_insights=20,
            agent_focus_areas=[
                ['optimizer'], ['architecture'], ['training'], ['data']
            ],
            agent_compute_time=[0.85, 0.8, 0.9, 0.85],
            agent_comm_time=[0.15, 0.2, 0.1, 0.15],
            agent_memory_mb=[1559.0, 1559.0, 1559.0, 1559.0],
            insight_timestamps=[i*0.7 for i in range(89)],
            insight_sharing=[[j % 4] if i % 3 == 0 else [] for i in range(89) for j in range(1)]
        )
    ]
    
    calculator = USIMCalculator()
    results = []
    
    print("\n" + "="*80)
    print("QUICK TEST WITH YOUR A30 DATA")
    print("="*80)
    
    for data in test_data:
        metrics = calculator.calculate(data)
        results.append(metrics)
        print(f"\n{data.n_agents} agents: USIM={metrics.usim_score:.4f}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("UNIVERSAL SWARM INTELLIGENCE METRIC (USIM)")
    print("="*80)
    
    # Check for quick test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        quick_test()
    else:
        # Run interactive app
        app = USIMApp()
        app.run()