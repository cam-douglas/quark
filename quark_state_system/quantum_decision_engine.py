#!/usr/bin/env python3
"""
Quantum Computing Decision Engine for Quark Brain Simulation
Intelligently decides when to use AWS Braket quantum computing vs classical computing
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import json
from pathlib import Path

class ComputationType(Enum):
    """Types of computation available"""
    CLASSICAL = "classical"
    QUANTUM_SIMULATOR = "quantum_simulator"
    QUANTUM_HARDWARE = "quantum_hardware"
    HYBRID = "hybrid"

class TaskComplexity(Enum):
    """Task complexity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class ComputationTask:
    """Represents a computation task that needs routing"""
    task_type: str
    problem_size: int
    expected_runtime_classical: float  # seconds
    requires_quantum_advantage: bool
    error_tolerance: float
    optimization_problem: bool = False
    search_space_size: Optional[int] = None
    entanglement_needed: bool = False
    superposition_benefit: bool = False
    
class QuantumDecisionEngine:
    """Intelligent decision engine for quantum vs classical computing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quantum advantage thresholds
        self.quantum_thresholds = {
            # Problem sizes where quantum might help
            "optimization_min_size": 100,
            "search_space_min_size": 1000,
            "matrix_size_for_quantum": 64,
            
            # Runtime thresholds (seconds)
            "classical_timeout": 300,  # 5 minutes
            "quantum_simulator_max": 3600,  # 1 hour
            
            # Cost considerations
            "quantum_hardware_cost_threshold": 10.0,  # USD
            "quantum_simulator_free_limit": 60,  # minutes per day
        }
        
        # Track usage statistics
        self.usage_stats = {
            "classical_tasks": 0,
            "quantum_simulator_tasks": 0,
            "quantum_hardware_tasks": 0,
            "total_quantum_cost": 0.0,
            "daily_simulator_usage": 0.0
        }
        
        # Load existing stats if available
        self._load_usage_stats()
    
    def _load_usage_stats(self):
        """Load usage statistics from file"""
        stats_file = Path.home() / ".quark" / "quantum_usage_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    self.usage_stats.update(json.load(f))
            except Exception as e:
                self.logger.warning(f"Could not load usage stats: {e}")
    
    def _save_usage_stats(self):
        """Save usage statistics to file"""
        stats_file = Path.home() / ".quark" / "quantum_usage_stats.json"
        stats_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.usage_stats, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save usage stats: {e}")
    
    def analyze_task_for_quantum_advantage(self, task: ComputationTask) -> Dict[str, Any]:
        """Analyze if a task would benefit from quantum computing"""
        
        analysis = {
            "quantum_advantage_score": 0.0,
            "recommended_type": ComputationType.CLASSICAL,
            "reasoning": [],
            "estimated_speedup": 1.0,
            "cost_estimate": 0.0,
            "complexity": TaskComplexity.LOW
        }
        
        # Determine task complexity
        analysis["complexity"] = self._assess_task_complexity(task)
        
        # Score factors for quantum advantage
        score = 0.0
        reasoning = []
        
        # 1. Problem type analysis
        if task.optimization_problem:
            score += 0.3
            reasoning.append("Optimization problem may benefit from quantum annealing")
        
        if task.entanglement_needed:
            score += 0.4
            reasoning.append("Task explicitly requires quantum entanglement")
        
        if task.superposition_benefit:
            score += 0.3
            reasoning.append("Task can leverage quantum superposition")
        
        # 2. Problem size analysis
        if task.problem_size >= self.quantum_thresholds["optimization_min_size"]:
            score += 0.2
            reasoning.append(f"Problem size ({task.problem_size}) suitable for quantum")
        
        if task.search_space_size and task.search_space_size >= self.quantum_thresholds["search_space_min_size"]:
            score += 0.3
            reasoning.append(f"Large search space ({task.search_space_size}) may benefit from Grover's algorithm")
        
        # 3. Classical runtime analysis
        if task.expected_runtime_classical > self.quantum_thresholds["classical_timeout"]:
            score += 0.4
            reasoning.append(f"Classical runtime ({task.expected_runtime_classical}s) exceeds threshold")
        
        # 4. Specific quantum computing use cases
        quantum_favorable_tasks = [
            "quantum_neural_network",
            "optimization_variational",
            "quantum_ml_training",
            "entanglement_simulation",
            "quantum_chemistry",
            "cryptographic_analysis"
        ]
        
        if task.task_type in quantum_favorable_tasks:
            score += 0.5
            reasoning.append(f"Task type '{task.task_type}' is quantum-favorable")
        
        # 5. Brain simulation specific considerations
        brain_quantum_tasks = [
            "consciousness_entanglement_modeling",
            "neural_superposition_states",
            "quantum_memory_consolidation",
            "brain_network_optimization"
        ]
        
        if task.task_type in brain_quantum_tasks:
            score += 0.6
            reasoning.append(f"Brain simulation task '{task.task_type}' may have quantum advantages")
        
        analysis["quantum_advantage_score"] = min(score, 1.0)
        analysis["reasoning"] = reasoning
        
        # Make recommendation based on score
        if score >= 0.7:
            analysis["recommended_type"] = ComputationType.QUANTUM_HARDWARE
            analysis["estimated_speedup"] = self._estimate_quantum_speedup(task)
            analysis["cost_estimate"] = self._estimate_quantum_cost(task, use_hardware=True)
        elif score >= 0.4:
            analysis["recommended_type"] = ComputationType.QUANTUM_SIMULATOR
            analysis["estimated_speedup"] = self._estimate_quantum_speedup(task, simulator=True)
            analysis["cost_estimate"] = self._estimate_quantum_cost(task, use_hardware=False)
        elif score >= 0.2:
            analysis["recommended_type"] = ComputationType.HYBRID
            reasoning.append("Hybrid approach recommended: classical preprocessing + quantum subroutines")
        else:
            reasoning.append("Classical computing recommended: no clear quantum advantage")
        
        return analysis
    
    def _assess_task_complexity(self, task: ComputationTask) -> TaskComplexity:
        """Assess the complexity level of a task"""
        if task.problem_size < 10:
            return TaskComplexity.LOW
        elif task.problem_size < 100:
            return TaskComplexity.MEDIUM
        elif task.problem_size < 1000:
            return TaskComplexity.HIGH
        else:
            return TaskComplexity.EXTREME
    
    def _estimate_quantum_speedup(self, task: ComputationTask, simulator: bool = False) -> float:
        """Estimate potential quantum speedup"""
        if simulator:
            # Simulators have overhead, limited speedup
            if task.problem_size < 20:
                return 0.5  # Slower due to simulation overhead
            else:
                return 1.2  # Slight advantage for larger problems
        
        # Hardware quantum speedup estimates
        if task.optimization_problem:
            return min(np.sqrt(task.problem_size), 10.0)  # Quadratic speedup, capped
        elif task.search_space_size:
            return min(np.sqrt(task.search_space_size), 100.0)  # Grover's algorithm
        else:
            return 2.0  # Conservative estimate
    
    def _estimate_quantum_cost(self, task: ComputationTask, use_hardware: bool = True) -> float:
        """Estimate cost in USD for quantum computation"""
        if not use_hardware:
            return 0.0  # Simulators are free (within limits)
        
        # Rough AWS Braket hardware pricing estimates
        base_cost = 0.30  # Base task cost
        per_shot_cost = 0.00035  # Per quantum circuit execution
        
        estimated_shots = max(100, task.problem_size * 10)
        return base_cost + (estimated_shots * per_shot_cost)
    
    def make_computation_decision(self, task: ComputationTask, force_classical: bool = False) -> Dict[str, Any]:
        """Make the final decision on computation type"""
        
        if force_classical:
            self.usage_stats["classical_tasks"] += 1
            self._save_usage_stats()
            return {
                "computation_type": ComputationType.CLASSICAL,
                "reasoning": "Forced to use classical computing",
                "estimated_cost": 0.0,
                "estimated_runtime": task.expected_runtime_classical
            }
        
        analysis = self.analyze_task_for_quantum_advantage(task)
        
        # Additional constraints and overrides
        decision = analysis["recommended_type"]
        
        # Cost constraints
        if analysis["cost_estimate"] > self.quantum_thresholds["quantum_hardware_cost_threshold"]:
            if decision == ComputationType.QUANTUM_HARDWARE:
                decision = ComputationType.QUANTUM_SIMULATOR
                analysis["reasoning"].append("Switched to simulator due to cost constraints")
        
        # Usage limits
        if (decision == ComputationType.QUANTUM_SIMULATOR and 
            self.usage_stats["daily_simulator_usage"] > self.quantum_thresholds["quantum_simulator_free_limit"]):
            decision = ComputationType.CLASSICAL
            analysis["reasoning"].append("Switched to classical due to daily simulator usage limit")
        
        # Update usage statistics
        if decision == ComputationType.CLASSICAL:
            self.usage_stats["classical_tasks"] += 1
        elif decision == ComputationType.QUANTUM_SIMULATOR:
            self.usage_stats["quantum_simulator_tasks"] += 1
            self.usage_stats["daily_simulator_usage"] += task.expected_runtime_classical / 60
        elif decision == ComputationType.QUANTUM_HARDWARE:
            self.usage_stats["quantum_hardware_tasks"] += 1
            self.usage_stats["total_quantum_cost"] += analysis["cost_estimate"]
        
        self._save_usage_stats()
        
        return {
            "computation_type": decision,
            "reasoning": analysis["reasoning"],
            "quantum_advantage_score": analysis["quantum_advantage_score"],
            "estimated_cost": analysis["cost_estimate"],
            "estimated_speedup": analysis["estimated_speedup"],
            "estimated_runtime": task.expected_runtime_classical / analysis["estimated_speedup"],
            "complexity": analysis["complexity"]
        }
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get usage statistics and recommendations"""
        total_tasks = (self.usage_stats["classical_tasks"] + 
                      self.usage_stats["quantum_simulator_tasks"] + 
                      self.usage_stats["quantum_hardware_tasks"])
        
        return {
            "total_tasks": total_tasks,
            "classical_percentage": (self.usage_stats["classical_tasks"] / max(1, total_tasks)) * 100,
            "quantum_percentage": ((self.usage_stats["quantum_simulator_tasks"] + 
                                  self.usage_stats["quantum_hardware_tasks"]) / max(1, total_tasks)) * 100,
            "total_quantum_cost": self.usage_stats["total_quantum_cost"],
            "daily_simulator_usage": self.usage_stats["daily_simulator_usage"],
            "recommendations": self._generate_usage_recommendations()
        }
    
    def _generate_usage_recommendations(self) -> List[str]:
        """Generate recommendations based on usage patterns"""
        recommendations = []
        
        if self.usage_stats["total_quantum_cost"] > 50:
            recommendations.append("⚠️ High quantum hardware costs - consider optimizing quantum tasks")
        
        if self.usage_stats["daily_simulator_usage"] > 45:
            recommendations.append("⚠️ Approaching daily simulator usage limit")
        
        quantum_percentage = ((self.usage_stats["quantum_simulator_tasks"] + 
                             self.usage_stats["quantum_hardware_tasks"]) / 
                            max(1, sum(self.usage_stats.values())))
        
        if quantum_percentage < 0.1:
            recommendations.append("💡 Consider exploring quantum computing for optimization tasks")
        elif quantum_percentage > 0.5:
            recommendations.append("🚀 High quantum usage - ensure tasks truly benefit from quantum computing")
        
        return recommendations

# Convenience functions for common brain simulation tasks
def create_brain_simulation_task(task_name: str, **kwargs) -> ComputationTask:
    """Create a computation task for brain simulation"""
    
    brain_task_configs = {
        "neural_network_training": {
            "requires_quantum_advantage": False,
            "optimization_problem": True,
            "error_tolerance": 0.01
        },
        "consciousness_modeling": {
            "requires_quantum_advantage": True,
            "entanglement_needed": True,
            "superposition_benefit": True,
            "error_tolerance": 0.001
        },
        "brain_connectivity_optimization": {
            "requires_quantum_advantage": False,
            "optimization_problem": True,
            "error_tolerance": 0.05
        },
        "quantum_memory_simulation": {
            "requires_quantum_advantage": True,
            "entanglement_needed": True,
            "superposition_benefit": True,
            "error_tolerance": 0.001
        },
        "neural_dynamics_modeling": {
            "requires_quantum_advantage": False,
            "optimization_problem": False,
            "error_tolerance": 0.01
        }
    }
    
    config = brain_task_configs.get(task_name, {})
    config.update(kwargs)
    
    return ComputationTask(
        task_type=task_name,
        problem_size=config.get("problem_size", 100),
        expected_runtime_classical=config.get("expected_runtime_classical", 60),
        requires_quantum_advantage=config.get("requires_quantum_advantage", False),
        error_tolerance=config.get("error_tolerance", 0.01),
        optimization_problem=config.get("optimization_problem", False),
        search_space_size=config.get("search_space_size"),
        entanglement_needed=config.get("entanglement_needed", False),
        superposition_benefit=config.get("superposition_benefit", False)
    )

def main():
    """Demo the quantum decision engine"""
    print("⚛️ Quantum Computing Decision Engine Demo")
    print("=" * 45)
    
    engine = QuantumDecisionEngine()
    
    # Test different types of brain simulation tasks
    test_tasks = [
        create_brain_simulation_task("neural_network_training", problem_size=50),
        create_brain_simulation_task("consciousness_modeling", problem_size=200),
        create_brain_simulation_task("brain_connectivity_optimization", problem_size=500),
        create_brain_simulation_task("quantum_memory_simulation", problem_size=100),
        create_brain_simulation_task("neural_dynamics_modeling", problem_size=1000)
    ]
    
    print("🧠 Testing Brain Simulation Tasks:")
    print()
    
    for task in test_tasks:
        decision = engine.make_computation_decision(task)
        
        print(f"📋 Task: {task.task_type}")
        print(f"   Computation Type: {decision['computation_type'].value}")
        print(f"   Quantum Score: {decision['quantum_advantage_score']:.2f}")
        print(f"   Estimated Speedup: {decision['estimated_speedup']:.1f}x")
        print(f"   Estimated Cost: ${decision['estimated_cost']:.2f}")
        print(f"   Complexity: {decision['complexity'].value}")
        print(f"   Reasoning: {decision['reasoning'][0] if decision['reasoning'] else 'Default classical'}")
        print()
    
    # Show usage report
    report = engine.get_usage_report()
    print("📊 Usage Report:")
    print(f"   Total Tasks: {report['total_tasks']}")
    print(f"   Classical: {report['classical_percentage']:.1f}%")
    print(f"   Quantum: {report['quantum_percentage']:.1f}%")
    print(f"   Total Quantum Cost: ${report['total_quantum_cost']:.2f}")
    
    if report['recommendations']:
        print("💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   {rec}")

if __name__ == "__main__":
    main()
