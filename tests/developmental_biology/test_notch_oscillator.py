"""Test Notch oscillator functionality"""

import pytest
import numpy as np
from brain.modules.developmental_biology.notch_oscillator import NotchOscillator, OscillatorPhase


def test_notch_oscillator_initialization():
    """Test Notch oscillator initializes correctly"""
    oscillator = NotchOscillator()
    assert len(oscillator.cell_states) == 0
    
    # Initialize a cell
    state = oscillator.initialize_cell("test_cell")
    assert state.notch_level >= 0.0
    assert state.notch_level <= 1.0
    assert state.delta_level >= 0.0
    assert state.delta_level <= 1.0


def test_division_bias_calculation():
    """Test division bias calculation from Notch levels"""
    oscillator = NotchOscillator()
    
    # Initialize cell
    oscillator.initialize_cell("cell1")
    
    # Get division bias
    bias = oscillator.get_division_bias("cell1")
    assert 0.0 <= bias <= 1.0


def test_oscillation_updates():
    """Test oscillation updates over time"""
    oscillator = NotchOscillator()
    
    # Initialize multiple cells
    for i in range(5):
        oscillator.initialize_cell(f"cell_{i}")
    
    # Update oscillations
    biases_t0 = oscillator.update_oscillations(0.0)
    biases_t1 = oscillator.update_oscillations(1.0)
    
    assert len(biases_t0) == 5
    assert len(biases_t1) == 5
    
    # Values should have changed (oscillation)
    changed = False
    for cell_id in biases_t0:
        if abs(biases_t0[cell_id] - biases_t1[cell_id]) > 0.01:
            changed = True
            break
    assert changed  # At least one cell should have changed


def test_oscillation_statistics():
    """Test oscillation statistics"""
    oscillator = NotchOscillator()
    
    # Initialize cells
    for i in range(10):
        oscillator.initialize_cell(f"cell_{i}")
    
    # Update oscillations
    oscillator.update_oscillations(0.0)
    
    # Get statistics
    stats = oscillator.get_oscillation_statistics()
    
    assert "mean_notch_level" in stats
    assert "mean_division_bias" in stats
    assert "symmetric_fraction" in stats
    assert "asymmetric_fraction" in stats
    
    # Fractions should sum to <= 1.0 (some may be intermediate)
    assert stats["symmetric_fraction"] + stats["asymmetric_fraction"] <= 1.0
