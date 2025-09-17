"""Notch Oscillator Module

Implements Notch-Delta signaling oscillations that influence symmetric vs
asymmetric division decisions. Based on Kageyama 2008 and Imayoshi 2013
showing Notch oscillations control neurogenesis timing.

Author: Quark AI
Date: 2025-01-30
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class OscillatorPhase(Enum):
    """Notch oscillator phases"""
    HIGH = "high"
    LOW = "low" 
    RISING = "rising"
    FALLING = "falling"


@dataclass
class NotchState:
    """Notch signaling state for a cell"""
    notch_level: float  # 0.0 to 1.0
    delta_level: float  # 0.0 to 1.0
    oscillator_phase: OscillatorPhase
    period: float  # Oscillation period in hours
    amplitude: float  # Oscillation amplitude
    last_update_time: float


class NotchOscillator:
    """Notch-Delta oscillator controlling division bias"""
    
    def __init__(self):
        """Initialize Notch oscillator"""
        # Oscillator parameters (Kageyama 2008)
        self.default_period = 2.5  # hours
        self.default_amplitude = 0.8
        self.noise_level = 0.1
        
        # Cell states
        self.cell_states: Dict[str, NotchState] = {}
        
        # Division bias parameters
        self.symmetric_threshold = 0.3  # Below this: symmetric division
        self.asymmetric_threshold = 0.7  # Above this: asymmetric division
    
    def initialize_cell(self, cell_id: str, initial_phase: float = 0.0) -> NotchState:
        """Initialize Notch state for a cell"""
        # Add some randomness to avoid synchronization
        phase_offset = np.random.uniform(-0.2, 0.2)
        period = self.default_period * (1.0 + np.random.uniform(-0.1, 0.1))
        
        notch_level = 0.5 + self.default_amplitude * 0.5 * np.sin(2 * np.pi * (initial_phase + phase_offset))
        notch_level = max(0.0, min(1.0, notch_level))
        
        # Delta is typically anti-correlated with Notch
        delta_level = 1.0 - notch_level
        
        state = NotchState(
            notch_level=notch_level,
            delta_level=delta_level,
            oscillator_phase=self._determine_phase(notch_level),
            period=period,
            amplitude=self.default_amplitude,
            last_update_time=0.0
        )
        
        self.cell_states[cell_id] = state
        return state
    
    def _determine_phase(self, notch_level: float) -> OscillatorPhase:
        """Determine oscillator phase from current Notch level"""
        if notch_level > 0.7:
            return OscillatorPhase.HIGH
        elif notch_level < 0.3:
            return OscillatorPhase.LOW
        elif notch_level > 0.5:
            return OscillatorPhase.RISING
        else:
            return OscillatorPhase.FALLING
    
    def update_oscillations(self, current_time: float) -> Dict[str, float]:
        """Update Notch oscillations for all cells"""
        division_biases = {}
        
        for cell_id, state in self.cell_states.items():
            dt = current_time - state.last_update_time
            
            # Update oscillation
            omega = 2 * np.pi / state.period
            phase = omega * current_time
            
            # Add noise
            noise = np.random.normal(0, self.noise_level)
            
            # Calculate new Notch level
            new_notch = 0.5 + state.amplitude * 0.5 * np.sin(phase) + noise
            new_notch = max(0.0, min(1.0, new_notch))
            
            # Update Delta (anti-correlated)
            new_delta = 1.0 - new_notch
            
            # Update state
            state.notch_level = new_notch
            state.delta_level = new_delta
            state.oscillator_phase = self._determine_phase(new_notch)
            state.last_update_time = current_time
            
            # Calculate division bias
            division_biases[cell_id] = self._calculate_division_bias(state)
        
        return division_biases
    
    def _calculate_division_bias(self, state: NotchState) -> float:
        """Calculate division bias from Notch state
        
        Returns:
            Float from 0.0 (symmetric) to 1.0 (asymmetric)
        """
        notch = state.notch_level
        
        if notch < self.symmetric_threshold:
            # Low Notch -> symmetric division (both daughters similar)
            return 0.2
        elif notch > self.asymmetric_threshold:
            # High Notch -> asymmetric division (one neurogenic, one progenitor)
            return 0.8
        else:
            # Intermediate -> linear interpolation
            range_size = self.asymmetric_threshold - self.symmetric_threshold
            relative_pos = (notch - self.symmetric_threshold) / range_size
            return 0.2 + 0.6 * relative_pos
    
    def get_division_bias(self, cell_id: str) -> float:
        """Get current division bias for a cell"""
        if cell_id not in self.cell_states:
            return 0.5  # Default neutral bias
        
        return self._calculate_division_bias(self.cell_states[cell_id])
    
    def get_oscillation_statistics(self) -> Dict[str, float]:
        """Get statistics about current oscillations"""
        if not self.cell_states:
            return {}
        
        notch_levels = [state.notch_level for state in self.cell_states.values()]
        biases = [self._calculate_division_bias(state) for state in self.cell_states.values()]
        
        return {
            "mean_notch_level": float(np.mean(notch_levels)),
            "notch_std": float(np.std(notch_levels)),
            "mean_division_bias": float(np.mean(biases)),
            "bias_std": float(np.std(biases)),
            "symmetric_fraction": float(np.mean([b < 0.4 for b in biases])),
            "asymmetric_fraction": float(np.mean([b > 0.6 for b in biases]))
        }
