#!/usr/bin/env python3
"""
🧠 Brain Simulation Tests
Tests brain integration with neural dynamics using simulation technologies

**Model:** Claude (Functional Implementation & Testing)
**Purpose:** Simulation testing of brain integration
**Validation Level:** System-level simulation verification
"""

import sys
import os
import pytest
import tempfile
import json
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.brain_launcher_v3 import Brain

class TestBrainSimulation:
    """Test brain simulation with neural dynamics"""
    
    def setup_method(self):
        """Setup test brain configuration"""
        self.brain_config = {
            "modules": {
                "architecture_agent": {"type": "architecture_agent"},
                "pfc": {"type": "pfc", "num_neurons": 50},
                "basal_ganglia": {"type": "basal_ganglia"},
                "thalamus": {"type": "thalamus"},
                "working_memory": {"type": "working_memory", "slots": 3, "num_neurons": 30},
                "dmn": {"type": "dmn"},
                "salience": {"type": "salience"},
                "attention": {"type": "attention"}
            },
            "curriculum": {
                "ticks_per_week": 50,
                "schedule": [
                    {"week": 0, "wm_slots": 3, "moe_k": 1},
                    {"week": 4, "wm_slots": 4, "moe_k": 2}
                ]
            }
        }
    
    def test_brain_initialization(self):
        """Test brain initialization with neural components"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_metrics_file = f.name
        
        try:
            brain = Brain(self.brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Check that brain has all required modules
            required_modules = ["pfc", "working_memory", "basal_ganglia", "thalamus"]
            for module_name in required_modules:
                assert module_name in brain.modules
                assert brain.modules[module_name] is not None
            
            # Check that PFC has neural population
            pfc = brain.modules["pfc"]
            assert hasattr(pfc, 'neural_population')
            assert pfc.neural_population.num_neurons == 50
            
            # Check that WM has neural population
            wm = brain.modules["working_memory"]
            assert hasattr(wm, 'neural_population')
            assert wm.neural_population.num_neurons == 30
            
        finally:
            if os.path.exists(temp_metrics_file):
                os.unlink(temp_metrics_file)
    
    def test_neural_dynamics_integration(self):
        """Test that neural dynamics are properly integrated"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_metrics_file = f.name
        
        try:
            brain = Brain(self.brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Run simulation for multiple steps to build up neural activity
            neural_metrics = []
            for step in range(50):  # Increased steps for neural activity buildup
                telemetry = brain.step(50)
                
                # Check PFC neural dynamics
                pfc_tel = telemetry.get("pfc", {})
                if "firing_rate" in pfc_tel:
                    neural_metrics.append({
                        "step": step,
                        "pfc_firing_rate": pfc_tel["firing_rate"],
                        "pfc_spike_count": pfc_tel.get("spike_count", 0),
                        "pfc_synchrony": pfc_tel.get("neural_synchrony", 0.0)
                    })
            
            # Should have neural dynamics data
            assert len(neural_metrics) > 0
            
            # Check that firing rates are reasonable
            firing_rates = [m["pfc_firing_rate"] for m in neural_metrics]
            assert all(rate >= 0.0 for rate in firing_rates)
            
            # At least some steps should have neural activity
            assert any(rate > 0.0 for rate in firing_rates)
            
        finally:
            if os.path.exists(temp_metrics_file):
                os.unlink(temp_metrics_file)
    
    def test_working_memory_neural_dynamics(self):
        """Test working memory neural dynamics"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_metrics_file = f.name
        
        try:
            brain = Brain(self.brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Run simulation to build up WM activity
            wm_metrics = []
            for step in range(20):
                telemetry = brain.step(50)
                
                # Check WM neural dynamics
                wm_tel = telemetry.get("working_memory", {})
                if "firing_rate" in wm_tel:
                    wm_metrics.append({
                        "step": step,
                        "wm_firing_rate": wm_tel["firing_rate"],
                        "wm_persistent_activity": wm_tel.get("persistent_activity", False),
                        "wm_synchrony": wm_tel.get("neural_synchrony", 0.0)
                    })
            
            # Should have WM neural dynamics data
            assert len(wm_metrics) > 0
            
            # Check that firing rates are reasonable
            firing_rates = [m["wm_firing_rate"] for m in wm_metrics]
            assert all(rate >= 0.0 for rate in firing_rates)
            
        finally:
            if os.path.exists(temp_metrics_file):
                os.unlink(temp_metrics_file)
    
    def test_neural_synchrony_evolution(self):
        """Test neural synchrony evolution over time"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_metrics_file = f.name
        
        try:
            brain = Brain(self.brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Run longer simulation to observe synchrony evolution
            synchrony_data = []
            for step in range(50):
                telemetry = brain.step(50)
                
                pfc_tel = telemetry.get("pfc", {})
                wm_tel = telemetry.get("working_memory", {})
                
                if "neural_synchrony" in pfc_tel:
                    synchrony_data.append({
                        "step": step,
                        "pfc_synchrony": pfc_tel["neural_synchrony"],
                        "wm_synchrony": wm_tel.get("neural_synchrony", 0.0)
                    })
            
            # Should have synchrony data
            assert len(synchrony_data) > 0
            
            # Check synchrony values are in valid range
            pfc_sync = [d["pfc_synchrony"] for d in synchrony_data]
            assert all(0.0 <= sync <= 1.0 for sync in pfc_sync)
            
        finally:
            if os.path.exists(temp_metrics_file):
                os.unlink(temp_metrics_file)
    
    def test_oscillation_power_analysis(self):
        """Test oscillation power analysis in different frequency bands"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_metrics_file = f.name
        
        try:
            brain = Brain(self.brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Run simulation to build up oscillation data
            oscillation_data = []
            for step in range(30):
                telemetry = brain.step(50)
                
                pfc_tel = telemetry.get("pfc", {})
                if "alpha_power" in pfc_tel:
                    oscillation_data.append({
                        "step": step,
                        "alpha_power": pfc_tel["alpha_power"],
                        "beta_power": pfc_tel["beta_power"],
                        "gamma_power": pfc_tel["gamma_power"]
                    })
            
            # Should have oscillation data
            assert len(oscillation_data) > 0
            
            # Check power values are non-negative
            for data in oscillation_data:
                assert data["alpha_power"] >= 0.0
                assert data["beta_power"] >= 0.0
                assert data["gamma_power"] >= 0.0
            
        finally:
            if os.path.exists(temp_metrics_file):
                os.unlink(temp_metrics_file)
    
    def test_membrane_potential_distribution(self):
        """Test membrane potential distribution across neurons"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_metrics_file = f.name
        
        try:
            brain = Brain(self.brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Run simulation and collect membrane potentials
            potential_data = []
            for step in range(10):
                telemetry = brain.step(50)
                
                pfc_tel = telemetry.get("pfc", {})
                wm_tel = telemetry.get("working_memory", {})
                
                if "membrane_potentials" in pfc_tel:
                    potential_data.append({
                        "step": step,
                        "pfc_potentials": pfc_tel["membrane_potentials"],
                        "wm_potentials": wm_tel.get("membrane_potentials", [])
                    })
            
            # Should have potential data
            assert len(potential_data) > 0
            
            # Check membrane potentials are in reasonable range
            for data in potential_data:
                pfc_pots = data["pfc_potentials"]
                if pfc_pots:
                    assert all(-100.0 <= pot <= 50.0 for pot in pfc_pots)
                
                wm_pots = data["wm_potentials"]
                if wm_pots:
                    assert all(-100.0 <= pot <= 50.0 for pot in wm_pots)
            
        finally:
            if os.path.exists(temp_metrics_file):
                os.unlink(temp_metrics_file)
    
    def test_synaptic_weight_evolution(self):
        """Test synaptic weight evolution over time"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_metrics_file = f.name
        
        try:
            brain = Brain(self.brain_config, stage="F", log_csv=temp_metrics_file)
            
            # Get initial synaptic weights
            pfc = brain.modules["pfc"]
            initial_weights = pfc.neural_population.get_synaptic_weights()
            
            # Run simulation to allow weight changes
            for step in range(20):
                brain.step(50)
            
            # Get final synaptic weights
            final_weights = pfc.neural_population.get_synaptic_weights()
            
            # Should have synaptic weights
            assert len(initial_weights) > 0
            assert len(final_weights) > 0
            
            # Weights should be in valid range
            for (pre, post), weight in final_weights.items():
                assert 0.0 <= weight <= 2.0  # Based on HebbianSynapse bounds
            
        finally:
            if os.path.exists(temp_metrics_file):
                os.unlink(temp_metrics_file)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
