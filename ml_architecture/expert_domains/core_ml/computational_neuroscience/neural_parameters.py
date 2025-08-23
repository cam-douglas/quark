"""
Neural Parameters and Neuromodulator Systems for Pillar 1.

Purpose: Tune neural parameters for realistic firing rates and implement neuromodulator systems
Inputs: Neural components, biological benchmarks, neuromodulator levels
Outputs: Tuned neural parameters, neuromodulator effects, homeostatic adjustments
Seeds: Biological parameter ranges from neuroscience literature
Deps: neural_components, biological_validator
"""

import numpy as np
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class NeuromodulatorType(Enum):
    """Types of neuromodulators."""
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    ACETYLCHOLINE = "acetylcholine"
    NOREPINEPHRINE = "norepinephrine"
    GABA = "gaba"


@dataclass
class NeuromodulatorLevel:
    """Neuromodulator concentration level."""
    type: NeuromodulatorType
    concentration: float  # nM
    baseline: float  # nM
    max_level: float  # nM
    decay_rate: float  # 1/s
    last_update: float = 0.0  # s


@dataclass
class NeuralParameters:
    """Tuned neural parameters for realistic firing rates."""
    # Neuron parameters
    membrane_threshold: float = -55.0  # mV
    resting_potential: float = -70.0  # mV
    reset_potential: float = -65.0  # mV
    membrane_time_constant: float = 20.0  # ms
    refractory_period: float = 2.0  # ms
    
    # Synaptic parameters
    excitatory_weight_range: Tuple[float, float] = (0.1, 0.5)
    inhibitory_weight_range: Tuple[float, float] = (-0.3, -0.1)
    learning_rate_range: Tuple[float, float] = (0.001, 0.01)
    
    # Population parameters
    connection_probability: float = 0.1
    max_connections_per_neuron: int = 100
    
    # Firing rate targets (Hz)
    target_firing_rate: float = 8.0
    target_firing_rate_std: float = 2.0
    
    # Synchrony targets
    target_synchrony: float = 0.3
    target_synchrony_std: float = 0.1
    
    # Oscillation targets
    target_oscillation_power: float = 0.2
    target_oscillation_frequency: float = 10.0  # Hz


class NeuromodulatorSystem:
    """Neuromodulator system for regulating neural dynamics."""
    
    def __init__(self):
        """Initialize neuromodulator system."""
        self.modulators = {
            NeuromodulatorType.DOPAMINE: NeuromodulatorLevel(
                type=NeuromodulatorType.DOPAMINE,
                concentration=50.0,  # nM
                baseline=50.0,
                max_level=200.0,
                decay_rate=0.1  # 1/s
            ),
            NeuromodulatorType.SEROTONIN: NeuromodulatorLevel(
                type=NeuromodulatorType.SEROTONIN,
                concentration=30.0,  # nM
                baseline=30.0,
                max_level=150.0,
                decay_rate=0.08  # 1/s
            ),
            NeuromodulatorType.ACETYLCHOLINE: NeuromodulatorLevel(
                type=NeuromodulatorType.ACETYLCHOLINE,
                concentration=20.0,  # nM
                baseline=20.0,
                max_level=100.0,
                decay_rate=0.15  # 1/s
            ),
            NeuromodulatorType.NOREPINEPHRINE: NeuromodulatorLevel(
                type=NeuromodulatorType.NOREPINEPHRINE,
                concentration=25.0,  # nM
                baseline=25.0,
                max_level=120.0,
                decay_rate=0.12  # 1/s
            ),
            NeuromodulatorType.GABA: NeuromodulatorLevel(
                type=NeuromodulatorType.GABA,
                concentration=40.0,  # nM
                baseline=40.0,
                max_level=180.0,
                decay_rate=0.2  # 1/s
            )
        }
        
        # Neuromodulator effects on neural parameters
        self.modulator_effects = {
            NeuromodulatorType.DOPAMINE: {
                "learning_rate_multiplier": 1.5,
                "threshold_modulation": -2.0,  # mV
                "excitability_boost": 1.2
            },
            NeuromodulatorType.SEROTONIN: {
                "learning_rate_multiplier": 0.8,
                "threshold_modulation": 1.0,  # mV
                "excitability_boost": 0.9
            },
            NeuromodulatorType.ACETYLCHOLINE: {
                "learning_rate_multiplier": 1.3,
                "threshold_modulation": -1.5,  # mV
                "excitability_boost": 1.1
            },
            NeuromodulatorType.NOREPINEPHRINE: {
                "learning_rate_multiplier": 1.4,
                "threshold_modulation": -1.0,  # mV
                "excitability_boost": 1.15
            },
            NeuromodulatorType.GABA: {
                "learning_rate_multiplier": 0.7,
                "threshold_modulation": 2.0,  # mV
                "excitability_boost": 0.8
            }
        }
        
    def update_modulator_levels(self, dt: float, current_time: float):
        """Update neuromodulator levels with decay."""
        for modulator in self.modulators.values():
            # Decay towards baseline
            decay_factor = np.exp(-modulator.decay_rate * dt)
            modulator.concentration = (
                modulator.baseline + 
                (modulator.concentration - modulator.baseline) * decay_factor
            )
            modulator.last_update = current_time
            
    def release_modulator(self, modulator_type: NeuromodulatorType, 
                         amount: float, current_time: float):
        """Release neuromodulator."""
        if modulator_type in self.modulators:
            modulator = self.modulators[modulator_type]
            modulator.concentration = min(
                modulator.max_level,
                modulator.concentration + amount
            )
            modulator.last_update = current_time
            
    def get_modulator_level(self, modulator_type: NeuromodulatorType) -> float:
        """Get current modulator level."""
        if modulator_type in self.modulators:
            return self.modulators[modulator_type].concentration
        return 0.0
        
    def get_modulated_parameters(self, base_params: NeuralParameters) -> NeuralParameters:
        """Get neural parameters modulated by current neuromodulator levels."""
        modulated_params = NeuralParameters()
        
        # Copy base parameters
        for field in base_params.__dataclass_fields__:
            setattr(modulated_params, field, getattr(base_params, field))
            
        # Apply neuromodulator effects
        for modulator_type, effects in self.modulator_effects.items():
            level = self.get_modulator_level(modulator_type)
            baseline = self.modulators[modulator_type].baseline
            
            # Normalize level (0-1)
            normalized_level = (level - baseline) / (self.modulators[modulator_type].max_level - baseline)
            normalized_level = np.clip(normalized_level, 0.0, 1.0)
            
            # Apply effects
            if "learning_rate_multiplier" in effects:
                factor = 1.0 + (effects["learning_rate_multiplier"] - 1.0) * normalized_level
                modulated_params.learning_rate_range = (
                    modulated_params.learning_rate_range[0] * factor,
                    modulated_params.learning_rate_range[1] * factor
                )
                
            if "threshold_modulation" in effects:
                threshold_shift = effects["threshold_modulation"] * normalized_level
                modulated_params.membrane_threshold += threshold_shift
                
            if "excitability_boost" in effects:
                factor = 1.0 + (effects["excitability_boost"] - 1.0) * normalized_level
                modulated_params.target_firing_rate *= factor
                
        return modulated_params


class HomeostaticPlasticity:
    """Homeostatic plasticity mechanisms."""
    
    def __init__(self, target_firing_rate: float = 8.0):
        """Initialize homeostatic plasticity."""
        self.target_firing_rate = target_firing_rate
        self.scale_factor = 1.0
        self.adaptation_rate = 0.01
        self.min_scale = 0.1
        self.max_scale = 10.0
        
    def update_scale_factor(self, current_firing_rate: float, dt: float):
        """Update synaptic scaling factor based on firing rate."""
        if current_firing_rate > 0:
            # Calculate error
            error = np.log(current_firing_rate / self.target_firing_rate)
            
            # Update scale factor
            self.scale_factor += self.adaptation_rate * error * dt
            
            # Clamp scale factor
            self.scale_factor = np.clip(self.scale_factor, self.min_scale, self.max_scale)
            
    def get_scaled_weight(self, base_weight: float) -> float:
        """Get homeostatically scaled weight."""
        return base_weight * self.scale_factor


class Metaplasticity:
    """Metaplasticity mechanisms for regulating plasticity."""
    
    def __init__(self):
        """Initialize metaplasticity."""
        self.plasticity_threshold = 0.5
        self.metaplasticity_rate = 0.001
        self.learning_rate_modulation = 1.0
        
    def update_plasticity_threshold(self, activity_level: float, dt: float):
        """Update plasticity threshold based on activity."""
        # BCM-like rule: threshold moves towards activity
        threshold_error = activity_level - self.plasticity_threshold
        self.plasticity_threshold += self.metaplasticity_rate * threshold_error * dt
        
        # Clamp threshold
        self.plasticity_threshold = np.clip(self.plasticity_threshold, 0.1, 2.0)
        
    def get_modulated_learning_rate(self, base_learning_rate: float, 
                                  activity_level: float) -> float:
        """Get metaplastically modulated learning rate."""
        # Reduce learning rate when activity is far from threshold
        distance_from_threshold = abs(activity_level - self.plasticity_threshold)
        modulation_factor = np.exp(-distance_from_threshold)
        
        return base_learning_rate * modulation_factor * self.learning_rate_modulation


class NeuralParameterTuner:
    """Tune neural parameters for realistic firing rates."""
    
    def __init__(self, target_params: NeuralParameters):
        """Initialize parameter tuner."""
        self.target_params = target_params
        self.current_params = NeuralParameters()
        self.tuning_history = []
        
    def tune_parameters(self, current_firing_rate: float, 
                       current_synchrony: float,
                       current_oscillation_power: float) -> NeuralParameters:
        """Tune parameters based on current performance."""
        # Calculate errors
        firing_rate_error = current_firing_rate - self.target_params.target_firing_rate
        synchrony_error = current_synchrony - self.target_params.target_synchrony
        oscillation_error = current_oscillation_power - self.target_params.target_oscillation_power
        
        # Tune membrane threshold based on firing rate
        if abs(firing_rate_error) > 1.0:  # More than 1 Hz error
            threshold_adjustment = -firing_rate_error * 0.5  # mV per Hz
            self.current_params.membrane_threshold += threshold_adjustment
            self.current_params.membrane_threshold = np.clip(
                self.current_params.membrane_threshold, -60.0, -45.0
            )
            
        # Tune connection probability based on synchrony
        if abs(synchrony_error) > 0.1:  # More than 0.1 error
            connection_adjustment = synchrony_error * 0.05
            self.current_params.connection_probability += connection_adjustment
            self.current_params.connection_probability = np.clip(
                self.current_params.connection_probability, 0.01, 0.3
            )
            
        # Tune learning rate based on oscillation power
        if abs(oscillation_error) > 0.05:  # More than 0.05 error
            learning_rate_adjustment = oscillation_error * 0.1
            self.current_params.learning_rate_range = (
                self.current_params.learning_rate_range[0] + learning_rate_adjustment,
                self.current_params.learning_rate_range[1] + learning_rate_adjustment
            )
            
        # Record tuning history
        self.tuning_history.append({
            "firing_rate_error": firing_rate_error,
            "synchrony_error": synchrony_error,
            "oscillation_error": oscillation_error,
            "membrane_threshold": self.current_params.membrane_threshold,
            "connection_probability": self.current_params.connection_probability,
            "learning_rate_range": self.current_params.learning_rate_range
        })
        
        return self.current_params
        
    def get_tuning_summary(self) -> Dict:
        """Get summary of parameter tuning."""
        if not self.tuning_history:
            return {}
            
        recent_history = self.tuning_history[-10:]  # Last 10 adjustments
        
        return {
            "total_adjustments": len(self.tuning_history),
            "recent_firing_rate_error": np.mean([h["firing_rate_error"] for h in recent_history]),
            "recent_synchrony_error": np.mean([h["synchrony_error"] for h in recent_history]),
            "recent_oscillation_error": np.mean([h["oscillation_error"] for h in recent_history]),
            "current_membrane_threshold": self.current_params.membrane_threshold,
            "current_connection_probability": self.current_params.connection_probability,
            "current_learning_rate_range": self.current_params.learning_rate_range
        }


def create_optimized_neural_parameters() -> NeuralParameters:
    """Create optimized neural parameters for realistic firing rates."""
    return NeuralParameters(
        membrane_threshold=-52.0,  # Slightly higher threshold for realistic rates
        resting_potential=-70.0,
        reset_potential=-65.0,
        membrane_time_constant=15.0,  # Faster dynamics
        refractory_period=1.5,  # Shorter refractory period
        
        excitatory_weight_range=(0.15, 0.4),  # Moderate weights
        inhibitory_weight_range=(-0.25, -0.08),  # Balanced inhibition
        learning_rate_range=(0.005, 0.015),  # Moderate learning rates
        
        connection_probability=0.12,  # Slightly higher connectivity
        max_connections_per_neuron=80,
        
        target_firing_rate=8.0,  # Realistic cortical firing rate
        target_firing_rate_std=2.0,
        target_synchrony=0.3,
        target_synchrony_std=0.1,
        target_oscillation_power=0.2,
        target_oscillation_frequency=10.0
    )
