"""Test apoptosis manager functionality"""

import pytest
import numpy as np
from brain.modules.developmental_biology.apoptosis_manager import ApoptosisManager, ApoptosisParameters


def test_baseline_death_probability():
    """Test baseline death probability calculation"""
    manager = ApoptosisManager()
    prob = manager.calculate_death_probability(dt_hours=1.0)
    
    # Should be close to baseline rate for 1 hour
    expected = 1.0 - np.exp(-0.002)  # baseline_rate = 0.002
    assert abs(prob - expected) < 0.0001


def test_bmp_increases_death():
    """Test that BMP increases death probability"""
    manager = ApoptosisManager()
    
    prob_no_bmp = manager.calculate_death_probability(bmp_level=0.0, dt_hours=1.0)
    prob_with_bmp = manager.calculate_death_probability(bmp_level=1.0, dt_hours=1.0)
    
    assert prob_with_bmp > prob_no_bmp


def test_apoptosis_fraction_24h():
    """Test expected apoptosis fraction over 24 hours"""
    manager = ApoptosisManager()
    
    # Create mock cells
    cells = {f"cell_{i}": type('Cell', (), {'position': (0, 0, 0)}) for i in range(1000)}
    initial_count = len(cells)
    
    # Apply apoptosis for 24 hours
    removed = manager.apply_apoptosis(cells, dt_hours=24.0)
    
    # Expected fraction: ~4.8% for baseline rate over 24h
    expected_fraction = 1.0 - np.exp(-0.002 * 24)
    actual_fraction = len(removed) / initial_count
    
    # Should be within 2% absolute error (stochastic)
    assert abs(actual_fraction - expected_fraction) < 0.02


def test_human_data_validation():
    """Test validation against human data ranges"""
    manager = ApoptosisManager()
    
    # 8-16 pcw should accept 2-5%
    assert manager.validate_against_human_data(0.03, pcw=10) == True
    assert manager.validate_against_human_data(0.01, pcw=10) == False
    assert manager.validate_against_human_data(0.06, pcw=10) == False
    
    # Outside range should have wider tolerance
    assert manager.validate_against_human_data(0.015, pcw=6) == True
