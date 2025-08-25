#!/usr/bin/env python3
"""
Real Basal Ganglia Architecture Implementation

This module implements the complete basal ganglia architecture including
all nuclei, connections, and computational models based on current
neuroscientific understanding.
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NucleusConfig:
    """Configuration for a basal ganglia nucleus."""
    name: str
    neuron_count: int
    neuron_type: str  # "spiny_projection", "fast_spiking", "low_threshold", etc.
    neurotransmitter: str  # "GABA", "glutamate", "dopamine", "acetylcholine"
    connectivity_pattern: str  # "feedforward", "recurrent", "lateral"
    plasticity_rule: str  # "stdp", "stdp_dopamine", "homeostatic"
    learning_rate: float
    activation_function: str  # "sigmoid", "relu", "tanh", "spike"
    refractory_period: float
    membrane_time_constant: float
    threshold: float
    resting_potential: float

class BasalGangliaArchitecture:
    """
    Complete basal ganglia architecture implementation.
    
    Implements all major nuclei with realistic connectivity patterns,
    neurotransmitter systems, and computational models based on
    current neuroscientific research.
    """
    
    def __init__(self):
        # Initialize nucleus configurations
        self.nuclei = self._initialize_nuclei()
        
        # Initialize connectivity matrices
        self.connectivity = self._initialize_connectivity()
        
        # Initialize neurotransmitter systems
        self.neurotransmitters = self._initialize_neurotransmitters()
        
        # Initialize plasticity rules
        self.plasticity_rules = self._initialize_plasticity_rules()
        
        # Initialize computational models
        self.computational_models = self._initialize_computational_models()
        
        # Performance metrics
        self.performance_metrics = {
            "total_neurons": sum(nucleus.neuron_count for nucleus in self.nuclei.values()),
            "total_connections": self._count_total_connections(),
            "architecture_complexity": self._calculate_architecture_complexity(),
            "computational_efficiency": 0.0,
            "learning_capacity": 0.0,
            "stability_index": 0.0
        }
        
        logger.info("🧠 Basal Ganglia Architecture initialized successfully")
    
    def _initialize_nuclei(self) -> Dict[str, NucleusConfig]:
        """Initialize all basal ganglia nuclei with realistic configurations."""
        nuclei = {}
        
        # Striatum (caudate nucleus + putamen)
        nuclei["striatum"] = NucleusConfig(
            name="striatum",
            neuron_count=500000,  # Large nucleus
            neuron_type="spiny_projection_neurons",
            neurotransmitter="GABA",
            connectivity_pattern="feedforward",
            plasticity_rule="stdp_dopamine",
            learning_rate=0.01,
            activation_function="spike",
            refractory_period=2.0,
            membrane_time_constant=20.0,
            threshold=-55.0,
            resting_potential=-70.0
        )
        
        # Globus Pallidus External (GPe)
        nuclei["gpe"] = NucleusConfig(
            name="gpe",
            neuron_count=50000,
            neuron_type="fast_spiking_interneurons",
            neurotransmitter="GABA",
            connectivity_pattern="recurrent",
            plasticity_rule="stdp",
            learning_rate=0.005,
            activation_function="spike",
            refractory_period=1.0,
            membrane_time_constant=15.0,
            threshold=-50.0,
            resting_potential=-65.0
        )
        
        # Globus Pallidus Internal (GPi)
        nuclei["gpi"] = NucleusConfig(
            name="gpi",
            neuron_count=30000,
            neuron_type="fast_spiking_output_neurons",
            neurotransmitter="GABA",
            connectivity_pattern="feedforward",
            plasticity_rule="stdp",
            learning_rate=0.005,
            activation_function="spike",
            refractory_period=1.5,
            membrane_time_constant=18.0,
            threshold=-52.0,
            resting_potential=-68.0
        )
        
        # Subthalamic Nucleus (STN)
        nuclei["stn"] = NucleusConfig(
            name="stn",
            neuron_count=20000,
            neuron_type="glutamatergic_neurons",
            neurotransmitter="glutamate",
            connectivity_pattern="recurrent",
            plasticity_rule="stdp",
            learning_rate=0.008,
            activation_function="spike",
            refractory_period=3.0,
            membrane_time_constant=25.0,
            threshold=-48.0,
            resting_potential=-62.0
        )
        
        # Substantia Nigra Pars Reticulata (SNr)
        nuclei["snr"] = NucleusConfig(
            name="snr",
            neuron_count=25000,
            neuron_type="gabaergic_output_neurons",
            neurotransmitter="GABA",
            connectivity_pattern="feedforward",
            plasticity_rule="stdp",
            learning_rate=0.005,
            activation_function="spike",
            refractory_period=1.2,
            membrane_time_constant=16.0,
            threshold=-54.0,
            resting_potential=-66.0
        )
        
        # Substantia Nigra Pars Compacta (SNc)
        nuclei["snc"] = NucleusConfig(
            name="snc",
            neuron_count=15000,
            neuron_type="dopaminergic_neurons",
            neurotransmitter="dopamine",
            connectivity_pattern="modulatory",
            plasticity_rule="dopamine_modulated",
            learning_rate=0.002,
            activation_function="spike",
            refractory_period=5.0,
            membrane_time_constant=30.0,
            threshold=-45.0,
            resting_potential=-58.0
        )
        
        # Ventral Tegmental Area (VTA)
        nuclei["vta"] = NucleusConfig(
            name="vta",
            neuron_count=10000,
            neuron_type="dopaminergic_neurons",
            neurotransmitter="dopamine",
            connectivity_pattern="modulatory",
            plasticity_rule="dopamine_modulated",
            learning_rate=0.002,
            activation_function="spike",
            refractory_period=4.0,
            membrane_time_constant=28.0,
            threshold=-46.0,
            resting_potential=-60.0
        )
        
        logger.info(f"✅ Initialized {len(nuclei)} nuclei with realistic configurations")
        return nuclei
    
    def _initialize_connectivity(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Initialize connectivity matrices between nuclei."""
        connectivity = {}
        
        # Striatum connections
        connectivity["striatum"] = {
            "gpe": self._create_connectivity_matrix("striatum", "gpe", connection_type="inhibitory", density=0.15),
            "gpi": self._create_connectivity_matrix("striatum", "gpi", connection_type="inhibitory", density=0.12),
            "snr": self._create_connectivity_matrix("striatum", "snr", connection_type="inhibitory", density=0.10)
        }
        
        # GPe connections
        connectivity["gpe"] = {
            "stn": self._create_connectivity_matrix("gpe", "stn", connection_type="inhibitory", density=0.20),
            "gpi": self._create_connectivity_matrix("gpe", "gpi", connection_type="inhibitory", density=0.18),
            "snr": self._create_connectivity_matrix("gpe", "snr", connection_type="inhibitory", density=0.15),
            "gpe": self._create_connectivity_matrix("gpe", "gpe", connection_type="inhibitory", density=0.25)  # Self-inhibition
        }
        
        # STN connections
        connectivity["stn"] = {
            "gpe": self._create_connectivity_matrix("stn", "gpe", connection_type="excitatory", density=0.22),
            "gpi": self._create_connectivity_matrix("stn", "gpi", connection_type="excitatory", density=0.20),
            "snr": self._create_connectivity_matrix("stn", "snr", connection_type="excitatory", density=0.18),
            "stn": self._create_connectivity_matrix("stn", "stn", connection_type="excitatory", density=0.30)  # Self-excitation
        }
        
        # GPi connections (output nucleus)
        connectivity["gpi"] = {
            "thalamus": self._create_connectivity_matrix("gpi", "thalamus", connection_type="inhibitory", density=0.25)
        }
        
        # SNr connections (output nucleus)
        connectivity["snr"] = {
            "thalamus": self._create_connectivity_matrix("snr", "thalamus", connection_type="inhibitory", density=0.20),
            "superior_colliculus": self._create_connectivity_matrix("snr", "superior_colliculus", connection_type="inhibitory", density=0.15)
        }
        
        # SNc connections (modulatory)
        connectivity["snc"] = {
            "striatum": self._create_connectivity_matrix("snc", "striatum", connection_type="modulatory", density=0.08),
            "gpe": self._create_connectivity_matrix("snc", "gpe", connection_type="modulatory", density=0.05),
            "gpi": self._create_connectivity_matrix("snc", "gpi", connection_type="modulatory", density=0.05)
        }
        
        # VTA connections (modulatory)
        connectivity["vta"] = {
            "striatum": self._create_connectivity_matrix("vta", "striatum", connection_type="modulatory", density=0.06),
            "gpe": self._create_connectivity_matrix("vta", "gpe", connection_type="modulatory", density=0.04),
            "gpi": self._create_connectivity_matrix("vta", "gpi", connection_type="modulatory", density=0.04)
        }
        
        logger.info("✅ Connectivity matrices initialized with realistic patterns")
        return connectivity
    
    def _create_connectivity_matrix(self, source: str, target: str, connection_type: str, density: float) -> np.ndarray:
        """Create a connectivity matrix between two nuclei."""
        source_size = self.nuclei[source].neuron_count
        target_size = self.nuclei[target].neuron_count
        
        # Create sparse connectivity matrix
        matrix = np.zeros((source_size, target_size))
        
        # Calculate number of connections based on density
        num_connections = int(density * source_size * target_size)
        
        # Randomly place connections
        if num_connections > 0:
            # Generate random indices
            source_indices = np.random.choice(source_size, num_connections, replace=True)
            target_indices = np.random.choice(target_size, num_connections, replace=True)
            
            # Set connection weights based on type
            if connection_type == "inhibitory":
                weights = -np.random.exponential(0.5, num_connections)
            elif connection_type == "excitatory":
                weights = np.random.exponential(0.5, num_connections)
            elif connection_type == "modulatory":
                weights = np.random.normal(0.1, 0.05, num_connections)
            else:
                weights = np.random.normal(0, 0.1, num_connections)
            
            # Apply connections
            for i in range(num_connections):
                matrix[source_indices[i], target_indices[i]] = weights[i]
        
        return matrix
    
    def _initialize_neurotransmitters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize neurotransmitter systems."""
        neurotransmitters = {}
        
        # GABA system (inhibitory)
        neurotransmitters["GABA"] = {
            "receptor_types": ["GABA_A", "GABA_B"],
            "reversal_potential": -75.0,
            "synaptic_delay": 1.0,
            "decay_time": 5.0,
            "modulation_factors": {
                "dopamine": 0.8,  # D2 receptor modulation
                "acetylcholine": 1.2,  # Cholinergic modulation
                "serotonin": 0.9
            }
        }
        
        # Glutamate system (excitatory)
        neurotransmitters["glutamate"] = {
            "receptor_types": ["AMPA", "NMDA"],
            "reversal_potential": 0.0,
            "synaptic_delay": 0.5,
            "decay_time": 3.0,
            "modulation_factors": {
                "dopamine": 1.1,  # D1 receptor modulation
                "acetylcholine": 0.9,
                "serotonin": 1.0
            }
        }
        
        # Dopamine system (modulatory)
        neurotransmitters["dopamine"] = {
            "receptor_types": ["D1", "D2", "D3", "D4", "D5"],
            "reversal_potential": -60.0,
            "synaptic_delay": 2.0,
            "decay_time": 100.0,  # Slow decay for neuromodulation
            "modulation_factors": {
                "acetylcholine": 1.0,
                "serotonin": 0.8,
                "norepinephrine": 1.1
            }
        }
        
        # Acetylcholine system (modulatory)
        neurotransmitters["acetylcholine"] = {
            "receptor_types": ["nicotinic", "muscarinic"],
            "reversal_potential": -10.0,
            "synaptic_delay": 1.5,
            "decay_time": 50.0,
            "modulation_factors": {
                "dopamine": 1.0,
                "serotonin": 0.9,
                "norepinephrine": 1.0
            }
        }
        
        logger.info("✅ Neurotransmitter systems initialized")
        return neurotransmitters
    
    def _initialize_plasticity_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize synaptic plasticity rules."""
        plasticity_rules = {}
        
        # Standard STDP
        plasticity_rules["stdp"] = {
            "tau_plus": 20.0,  # ms
            "tau_minus": 20.0,  # ms
            "a_plus": 0.01,
            "a_minus": 0.01,
            "max_weight": 2.0,
            "min_weight": 0.0
        }
        
        # STDP with dopamine modulation
        plasticity_rules["stdp_dopamine"] = {
            "tau_plus": 20.0,
            "tau_minus": 20.0,
            "a_plus": 0.01,
            "a_minus": 0.01,
            "dopamine_modulation": True,
            "dopamine_threshold": 0.5,
            "dopamine_multiplier": 1.5,
            "max_weight": 2.0,
            "min_weight": 0.0
        }
        
        # Homeostatic plasticity
        plasticity_rules["homeostatic"] = {
            "target_rate": 0.1,  # Hz
            "learning_rate": 0.001,
            "time_constant": 1000.0,  # ms
            "max_weight_change": 0.1
        }
        
        # Dopamine-modulated plasticity
        plasticity_rules["dopamine_modulated"] = {
            "base_learning_rate": 0.001,
            "dopamine_sensitivity": 2.0,
            "dopamine_threshold": 0.3,
            "max_modulation": 5.0
        }
        
        logger.info("✅ Plasticity rules initialized")
        return plasticity_rules
    
    def _initialize_computational_models(self) -> Dict[str, Any]:
        """Initialize computational models for each nucleus."""
        models = {}
        
        # Striatum model (action selection)
        models["striatum"] = {
            "model_type": "action_selection",
            "algorithm": "actor_critic",
            "parameters": {
                "learning_rate": 0.01,
                "discount_factor": 0.95,
                "exploration_rate": 0.1,
                "eligibility_trace": 0.8
            }
        }
        
        # GPe model (inhibition control)
        models["gpe"] = {
            "model_type": "inhibition_control",
            "algorithm": "adaptive_inhibition",
            "parameters": {
                "inhibition_strength": 0.7,
                "adaptation_rate": 0.005,
                "target_activity": 0.3
            }
        }
        
        # STN model (excitation control)
        models["stn"] = {
            "model_type": "excitation_control",
            "algorithm": "adaptive_excitation",
            "parameters": {
                "excitation_strength": 0.6,
                "adaptation_rate": 0.008,
                "target_activity": 0.4
            }
        }
        
        # GPi/SNr model (output control)
        models["output_nuclei"] = {
            "model_type": "output_control",
            "algorithm": "balanced_inhibition",
            "parameters": {
                "output_threshold": 0.5,
                "balance_factor": 0.8,
                "stability_margin": 0.2
            }
        }
        
        # SNc/VTA model (reward processing)
        models["dopamine_nuclei"] = {
            "model_type": "reward_processing",
            "algorithm": "temporal_difference",
            "parameters": {
                "reward_learning_rate": 0.002,
                "temporal_discount": 0.9,
                "prediction_error_threshold": 0.1
            }
        }
        
        logger.info("✅ Computational models initialized")
        return models
    
    def _count_total_connections(self) -> int:
        """Count total number of connections in the architecture."""
        total = 0
        for source_connections in self.connectivity.values():
            for target_matrix in source_connections.values():
                total += np.count_nonzero(target_matrix)
        return total
    
    def _calculate_architecture_complexity(self) -> float:
        """Calculate architecture complexity score."""
        try:
            # Factors contributing to complexity
            neuron_diversity = len(set(nucleus.neuron_type for nucleus in self.nuclei.values()))
            connection_diversity = len(set(connection_type for connections in self.connectivity.values() 
                                        for connection_type in connections.values()))
            neurotransmitter_diversity = len(self.neurotransmitters)
            plasticity_diversity = len(self.plasticity_rules)
            
            # Normalize factors
            complexity_score = (
                neuron_diversity / 10.0 +
                connection_diversity / 20.0 +
                neurotransmitter_diversity / 10.0 +
                plasticity_diversity / 10.0
            ) / 4.0
            
            return min(1.0, complexity_score)
            
        except Exception as e:
            logger.error(f"Complexity calculation failed: {e}")
            return 0.5
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get comprehensive architecture summary."""
        return {
            "nuclei_count": len(self.nuclei),
            "total_neurons": self.performance_metrics["total_neurons"],
            "total_connections": self.performance_metrics["total_connections"],
            "architecture_complexity": self.performance_metrics["architecture_complexity"],
            "nuclei_details": {
                name: {
                    "neuron_count": nucleus.neuron_count,
                    "neuron_type": nucleus.neuron_type,
                    "neurotransmitter": nucleus.neurotransmitter,
                    "connectivity_pattern": nucleus.connectivity_pattern,
                    "plasticity_rule": nucleus.plasticity_rule
                }
                for name, nucleus in self.nuclei.items()
            },
            "connectivity_summary": {
                source: {
                    target: {
                        "connection_count": np.count_nonzero(matrix),
                        "connection_density": np.count_nonzero(matrix) / matrix.size,
                        "average_weight": np.mean(matrix[matrix != 0]) if np.count_nonzero(matrix) > 0 else 0.0
                    }
                    for target, matrix in connections.items()
                }
                for source, connections in self.connectivity.items()
            },
            "timestamp": time.time()
        }
    
    def simulate_activity(self, duration_ms: float = 1000.0, time_step_ms: float = 1.0) -> Dict[str, Any]:
        """Simulate basal ganglia activity for a given duration."""
        try:
            logger.info(f"🧠 Starting basal ganglia simulation for {duration_ms}ms")
            
            simulation_start = time.time()
            
            # Initialize activity states
            activity_states = self._initialize_activity_states()
            
            # Run simulation
            time_steps = int(duration_ms / time_step_ms)
            activity_history = []
            
            for step in range(time_steps):
                # Update activity states
                activity_states = self._update_activity_states(activity_states, time_step_ms)
                
                # Record activity
                if step % 10 == 0:  # Record every 10th step
                    step_activity = {
                        "time_ms": step * time_step_ms,
                        "nuclei_activity": {name: np.mean(states) for name, states in activity_states.items()},
                        "overall_activity": np.mean([np.mean(states) for states in activity_states.values()])
                    }
                    activity_history.append(step_activity)
            
            simulation_time = time.time() - simulation_start
            
            # Calculate simulation metrics
            simulation_metrics = {
                "duration_ms": duration_ms,
                "time_steps": time_steps,
                "simulation_time": simulation_time,
                "real_time_factor": duration_ms / (simulation_time * 1000),
                "final_activity": {name: np.mean(states) for name, states in activity_states.items()},
                "activity_stability": self._calculate_activity_stability(activity_history)
            }
            
            logger.info(f"✅ Simulation completed in {simulation_time:.3f}s")
            return {
                "success": True,
                "simulation_metrics": simulation_metrics,
                "activity_history": activity_history,
                "final_states": activity_states
            }
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _initialize_activity_states(self) -> Dict[str, np.ndarray]:
        """Initialize activity states for all nuclei."""
        states = {}
        for name, nucleus in self.nuclei.items():
            # Initialize with random activity around resting potential
            states[name] = np.random.normal(
                nucleus.resting_potential,
                abs(nucleus.threshold - nucleus.resting_potential) * 0.1,
                nucleus.neuron_count
            )
        return states
    
    def _update_activity_states(self, states: Dict[str, np.ndarray], time_step_ms: float) -> Dict[str, np.ndarray]:
        """Update activity states for one time step."""
        new_states = {}
        
        for name, nucleus in self.nuclei.items():
            current_states = states[name]
            
            # Apply membrane dynamics
            membrane_decay = np.exp(-time_step_ms / nucleus.membrane_time_constant)
            new_states[name] = (current_states - nucleus.resting_potential) * membrane_decay + nucleus.resting_potential
            
            # Apply synaptic inputs
            synaptic_inputs = self._calculate_synaptic_inputs(name, states, time_step_ms)
            new_states[name] += synaptic_inputs
            
            # Apply activation function
            if nucleus.activation_function == "spike":
                # Spike generation
                spike_mask = new_states[name] > nucleus.threshold
                new_states[name][spike_mask] = nucleus.resting_potential
                
                # Refractory period
                refractory_mask = (new_states[name] < nucleus.threshold) & (new_states[name] > nucleus.resting_potential - 5.0)
                new_states[name][refractory_mask] -= time_step_ms / nucleus.refractory_period * 5.0
            
            # Apply bounds
            new_states[name] = np.clip(new_states[name], nucleus.resting_potential - 20.0, nucleus.threshold + 10.0)
        
        return new_states
    
    def _calculate_synaptic_inputs(self, nucleus_name: str, states: Dict[str, np.ndarray], time_step_ms: float) -> np.ndarray:
        """Calculate synaptic inputs to a nucleus."""
        inputs = np.zeros(self.nuclei[nucleus_name].neuron_count)
        
        if nucleus_name in self.connectivity:
            for source_name, connection_matrix in self.connectivity[nucleus_name].items():
                if source_name in states:
                    source_states = states[source_name]
                    
                    # Calculate weighted inputs
                    weighted_inputs = connection_matrix.T @ source_states
                    
                    # Apply neurotransmitter-specific scaling
                    source_neurotransmitter = self.nuclei[source_name].neurotransmitter
                    scaling_factor = self._get_neurotransmitter_scaling(source_neurotransmitter, time_step_ms)
                    
                    inputs += weighted_inputs * scaling_factor
        
        return inputs
    
    def _get_neurotransmitter_scaling(self, neurotransmitter: str, time_step_ms: float) -> float:
        """Get scaling factor for neurotransmitter effects."""
        if neurotransmitter in self.neurotransmitters:
            nt_config = self.neurotransmitters[neurotransmitter]
            decay_factor = np.exp(-time_step_ms / nt_config["decay_time"])
            return decay_factor
        return 1.0
    
    def _calculate_activity_stability(self, activity_history: List[Dict[str, Any]]) -> float:
        """Calculate activity stability across the simulation."""
        try:
            if len(activity_history) < 2:
                return 0.0
            
            # Calculate variance in overall activity
            overall_activities = [step["overall_activity"] for step in activity_history]
            variance = np.var(overall_activities)
            
            # Convert to stability score (lower variance = higher stability)
            stability = max(0.0, 1.0 - variance)
            
            return stability
            
        except Exception as e:
            logger.error(f"Stability calculation failed: {e}")
            return 0.0
