#!/usr/bin/env python3
"""
🧠 Shared Test Configuration
Common test fixtures and configuration for all test categories
"""

import sys
import os
import pytest
import tempfile
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def brain_config():
    """Standard brain configuration for testing"""
    return {
        "modules": {
            "architecture_agent": {"type": "architecture_agent"},
            "pfc": {"type": "pfc", "num_neurons": 50},
            "basal_ganglia": {"type": "basal_ganglia"},
            "thalamus": {"type": "thalamus"},
            "working_memory": {"type": "working_memory", "slots": 3, "num_neurons": 30},
            "dmn": {"type": "dmn"},
            "salience": {"type": "salience"},
            "attention": {"type": "attention"}
        }
    }

@pytest.fixture
def temp_metrics_file():
    """Temporary metrics file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_file = f.name
    yield temp_file
    if os.path.exists(temp_file):
        os.unlink(temp_file)

@pytest.fixture
def neural_population_config():
    """Standard neural population configuration"""
    return {
        "population_id": "test_pop",
        "num_neurons": 20,
        "neuron_type": "regular_spiking",
        "connectivity": 0.3
    }

@pytest.fixture
def spiking_neuron_config():
    """Standard spiking neuron configuration"""
    return {
        "neuron_id": 0,
        "neuron_type": "regular_spiking",
        "v0": -65.0,
        "u0": 0.0
    }

@pytest.fixture
def hebbian_synapse_config():
    """Standard Hebbian synapse configuration"""
    return {
        "pre_neuron_id": 0,
        "post_neuron_id": 1,
        "initial_weight": 0.5,
        "learning_rate": 0.01
    }

@pytest.fixture
def stdp_config():
    """Standard STDP configuration"""
    return {
        "tau_plus": 20.0,
        "tau_minus": 20.0,
        "A_plus": 0.01,
        "A_minus": 0.01
    }

@pytest.fixture
def biological_benchmarks():
    """Biological benchmarks for validation"""
    return {
        "firing_rate_range": (0.1, 50.0),
        "membrane_potential_range": (-100.0, 50.0),
        "synaptic_weight_range": (0.0, 2.0),
        "synchrony_range": (0.0, 1.0)
    }

@pytest.fixture
def simulation_parameters():
    """Standard simulation parameters"""
    return {
        "time_step": 1.0,  # ms
        "simulation_duration": 1000.0,  # ms
        "warmup_steps": 50,
        "measurement_steps": 100,
        "sampling_rate": 1000.0  # Hz
    }

@pytest.fixture
def test_data_dir():
    """Test data directory"""
    return Path(__file__).parent / "data"

@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests"""
    seed = 42
    np.random.seed(seed)
    return seed

# Test markers for categorization
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "simulation: mark test as a simulation test"
    )
    config.addinivalue_line(
        "markers", "validation: mark test as a validation test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "biological: mark test as biological validation"
    )

# Helper functions for tests
def assert_biological_range(value, min_val, max_val, tolerance=0.1):
    """Assert that a value is within biological range"""
    assert min_val - tolerance <= value <= max_val + tolerance, \
        f"Value {value} outside biological range [{min_val}, {max_val}]"

def assert_firing_rate_biological(rate):
    """Assert firing rate is biologically plausible"""
    assert_biological_range(rate, 0.1, 50.0)

def assert_membrane_potential_biological(potential):
    """Assert membrane potential is biologically plausible"""
    assert_biological_range(potential, -100.0, 50.0)

def assert_synaptic_weight_biological(weight):
    """Assert synaptic weight is biologically plausible"""
    assert_biological_range(weight, 0.0, 2.0)

def assert_synchrony_biological(synchrony):
    """Assert neural synchrony is biologically plausible"""
    assert_biological_range(synchrony, 0.0, 1.0)

def run_simulation_steps(brain, num_steps, step_size=50):
    """Run brain simulation for specified number of steps"""
    telemetry_data = []
    for step in range(num_steps):
        telemetry = brain.step(step_size)
        telemetry_data.append(telemetry)
    return telemetry_data

def calculate_statistics(data_list, key):
    """Calculate statistics for a list of data points"""
    values = [data.get(key, 0) for data in data_list if key in data]
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values)
    }
