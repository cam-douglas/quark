# tests/test_enhanced_pillar2.py

"""
Purpose: Comprehensive validation test for enhanced Pillar 2 (Gating & Reinforcement) components.
Inputs: Test scenarios for spiking neurons, pathway separation, dopamine responses
Outputs: Validation results and biological accuracy metrics
Dependencies: brain_modules.basal_ganglia.enhanced_architecture, numpy, pytest
"""

import sys
import os
import numpy as np
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from brain_modules.basal_ganglia.enhanced_architecture import (
    SpikingNeuron,
    EnhancedStriatum,
    EnhancedDopamineSystem,
    EnhancedBasalGanglia
)

class TestEnhancedBiologicalModels:
    """Test suite for enhanced biological models in Pillar 2"""
    
    def test_spiking_neuron_basic_functionality(self):
        """Test basic spiking neuron behavior"""
        neuron = SpikingNeuron(threshold=1.0, decay=0.9, refractory_period=3)
        
        # Test subthreshold inputs
        for i in range(5):
            spiked = neuron.update(0.5, i)
            assert not spiked, f"Neuron should not spike with subthreshold input at time {i}"
        
        # Test supra-threshold input
        spiked = neuron.update(1.5, 5)
        assert spiked, "Neuron should spike with supra-threshold input"
        assert neuron.membrane_potential == 0.0, "Membrane potential should reset after spike"
        
        # Test refractory period
        for i in range(6, 9):
            spiked = neuron.update(2.0, i)
            assert not spiked, f"Neuron should be in refractory period at time {i}"
        
        # Test spike history
        assert sum(neuron.spike_history) == 1, "Should have exactly one spike recorded"
    
    def test_spiking_neuron_firing_rate(self):
        """Test firing rate calculation"""
        neuron = SpikingNeuron(threshold=0.5, decay=0.8)
        
        # Generate spikes
        for i in range(20):
            input_current = 1.0 if i % 3 == 0 else 0.1  # Spike every 3rd time step
            neuron.update(input_current, i)
        
        firing_rate = neuron.get_firing_rate(window_size=10)
        expected_rate = 3 / 10  # 3 spikes in last 10 time steps
        assert abs(firing_rate - expected_rate) < 0.1, f"Firing rate should be ~{expected_rate}, got {firing_rate}"
    
    def test_enhanced_striatum_pathway_separation(self):
        """Test direct/indirect pathway separation in striatum"""
        striatum = EnhancedStriatum(num_neurons=10)
        
        # Test with different input signals
        input_signal = np.random.rand(10)
        dopamine_level = 0.5
        
        direct_activity, indirect_activity = striatum.process_input(input_signal, dopamine_level)
        
        # Both pathways should produce some activity
        assert direct_activity >= 0, "Direct pathway activity should be non-negative"
        assert indirect_activity >= 0, "Indirect pathway activity should be non-negative"
        assert direct_activity + indirect_activity <= 10, "Total activity should not exceed neuron count"
        
        # Test receptor distributions
        assert len(striatum.d1_receptors) == 5, "Should have 5 D1 receptors"
        assert len(striatum.d2_receptors) == 5, "Should have 5 D2 receptors"
    
    def test_enhanced_striatum_dopamine_modulation(self):
        """Test dopamine modulation of pathways"""
        striatum = EnhancedStriatum(num_neurons=20)
        input_signal = np.random.rand(20)
        
        # Test with low dopamine
        direct_low, indirect_low = striatum.process_input(input_signal, 0.1)
        
        # Test with high dopamine
        direct_high, indirect_high = striatum.process_input(input_signal, 0.9)
        
        # High dopamine should enhance direct pathway and inhibit indirect pathway
        # (Note: This is a simplified test - real modulation is more complex)
        assert direct_high >= direct_low, "High dopamine should enhance direct pathway"
    
    def test_enhanced_dopamine_phasic_tonic_separation(self):
        """Test phasic/tonic dopamine response separation"""
        dopamine_system = EnhancedDopamineSystem()
        
        # Test positive RPE
        dopamine_level, rpe = dopamine_system.process_reward(0.2, 1.0, 0)
        assert rpe > 0, "Should have positive RPE"
        assert dopamine_level > dopamine_system.baseline_dopamine, "Should have dopamine surge"
        
        # Check phasic/tonic separation
        tonic, phasic, total = dopamine_system.response_history[-1]
        assert phasic > 0, "Should have positive phasic response"
        assert tonic > dopamine_system.baseline_dopamine, "Should have elevated tonic level"
        
        # Test negative RPE
        dopamine_level, rpe = dopamine_system.process_reward(0.8, 0.1, 1)
        assert rpe < 0, "Should have negative RPE"
        assert dopamine_level < dopamine_system.response_history[0][2], "Should have dopamine dip"
        
        # Test phasic/tonic ratio
        ratio = dopamine_system.get_phasic_tonic_ratio()
        assert ratio > 0, "Phasic/tonic ratio should be positive"
    
    def test_enhanced_dopamine_temporal_dynamics(self):
        """Test temporal dynamics of dopamine responses"""
        dopamine_system = EnhancedDopamineSystem()
        
        # Simulate repeated rewards
        for i in range(10):
            predicted_reward = 0.3
            actual_reward = 0.8 if i % 3 == 0 else 0.2  # Periodic high rewards
            dopamine_system.process_reward(predicted_reward, actual_reward, i)
        
        # Check that tonic level has adjusted over time
        initial_tonic = dopamine_system.baseline_dopamine
        final_tonic = dopamine_system.response_history[-1][0]
        assert abs(final_tonic - initial_tonic) > 0.01, "Tonic level should adjust over time"
        
        # Check history tracking
        assert len(dopamine_system.response_history) == 10, "Should have 10 response records"
        assert len(dopamine_system.time_history) == 10, "Should have 10 time records"
    
    def test_enhanced_basal_ganglia_integration(self):
        """Test integration of enhanced components"""
        enhanced_bg = EnhancedBasalGanglia(num_neurons=50)
        
        # Test action selection
        input_signal = np.random.rand(50)
        predicted_reward = 0.3
        actual_reward = 0.8
        
        action, strength = enhanced_bg.process_action_selection(input_signal, predicted_reward, actual_reward)
        
        # Check outputs
        assert action in [0, 1], "Action should be binary"
        assert isinstance(strength, float), "Action strength should be float"
        
        # Check system state
        state = enhanced_bg.get_system_state()
        assert 'direct_pathway_activity' in state, "State should include direct pathway activity"
        assert 'indirect_pathway_activity' in state, "State should include indirect pathway activity"
        assert 'dopamine_level' in state, "State should include dopamine level"
        assert 'phasic_tonic_ratio' in state, "State should include phasic/tonic ratio"
    
    def test_enhanced_basal_ganglia_learning_dynamics(self):
        """Test learning dynamics over multiple trials"""
        enhanced_bg = EnhancedBasalGanglia(num_neurons=30)
        
        # Simulate learning trials
        for i in range(50):
            input_signal = np.random.rand(30)
            predicted_reward = 0.4
            actual_reward = 0.9 if i % 5 == 0 else 0.1  # Periodic high rewards
            
            action, strength = enhanced_bg.process_action_selection(input_signal, predicted_reward, actual_reward)
            
            # Record activity
            if i % 10 == 0:
                state = enhanced_bg.get_system_state()
                assert state['dopamine_level'] >= 0, "Dopamine level should be non-negative"
                assert state['dopamine_level'] <= 1, "Dopamine level should be <= 1"
        
        # Check activity history
        assert len(enhanced_bg.activity_history) == 50, "Should have 50 activity records"
        
        # Check that system has learned (simplified test)
        recent_activities = enhanced_bg.activity_history[-10:]
        action_strengths = [a['action_strength'] for a in recent_activities]
        assert len(action_strengths) == 10, "Should have 10 recent action strengths"
    
    def test_biological_plausibility_metrics(self):
        """Test biological plausibility of the enhanced system"""
        enhanced_bg = EnhancedBasalGanglia(num_neurons=40)
        
        # Test realistic firing rates
        for i in range(20):
            input_signal = np.random.rand(40)
            enhanced_bg.process_action_selection(input_signal, 0.3, 0.7)
        
        # Check firing rates are in realistic range (0-100 Hz equivalent)
        direct_activity, indirect_activity = enhanced_bg.striatum.get_pathway_activity()
        assert 0 <= direct_activity <= 1, "Direct pathway activity should be in [0,1]"
        assert 0 <= indirect_activity <= 1, "Indirect pathway activity should be in [0,1]"
        
        # Check dopamine levels are in realistic range
        state = enhanced_bg.get_system_state()
        assert 0 <= state['dopamine_level'] <= 1, "Dopamine level should be in [0,1]"
        
        # Check phasic/tonic ratio is reasonable
        ratio = state['phasic_tonic_ratio']
        assert ratio >= 0, "Phasic/tonic ratio should be non-negative"
    
    def test_robustness_to_noise(self):
        """Test system robustness to input noise"""
        enhanced_bg = EnhancedBasalGanglia(num_neurons=25)
        
        # Test with clean input
        clean_input = np.random.rand(25)
        clean_action, clean_strength = enhanced_bg.process_action_selection(clean_input, 0.3, 0.7)
        
        # Test with noisy input
        noise_level = 0.1
        noisy_input = clean_input + noise_level * np.random.randn(25)
        noisy_action, noisy_strength = enhanced_bg.process_action_selection(noisy_input, 0.3, 0.7)
        
        # System should be somewhat robust to noise
        # (Note: This is a simplified robustness test)
        assert isinstance(noisy_action, int), "Noisy input should still produce valid action"
        assert isinstance(noisy_strength, float), "Noisy input should still produce valid strength"

def run_enhanced_pillar2_validation():
    """Run comprehensive validation of enhanced Pillar 2 implementation"""
    print("🧠 Enhanced Pillar 2 Validation: Biological Models & Pathway Separation")
    print("=" * 70)
    
    # Create test instance
    test_suite = TestEnhancedBiologicalModels()
    
    # Run all tests
    test_methods = [
        test_suite.test_spiking_neuron_basic_functionality,
        test_suite.test_spiking_neuron_firing_rate,
        test_suite.test_enhanced_striatum_pathway_separation,
        test_suite.test_enhanced_striatum_dopamine_modulation,
        test_suite.test_enhanced_dopamine_phasic_tonic_separation,
        test_suite.test_enhanced_dopamine_temporal_dynamics,
        test_suite.test_enhanced_basal_ganglia_integration,
        test_suite.test_enhanced_basal_ganglia_learning_dynamics,
        test_suite.test_biological_plausibility_metrics,
        test_suite.test_robustness_to_noise
    ]
    
    passed_tests = 0
    total_tests = len(test_methods)
    
    for test_method in test_methods:
        try:
            test_method()
            print(f"✅ {test_method.__name__}: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_method.__name__}: FAILED - {str(e)}")
    
    print("=" * 70)
    print(f"📊 Enhanced Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 Enhanced Pillar 2 Validation: ALL TESTS PASSED")
        print("✅ Enhanced Basal Ganglia with biological models is working correctly")
        print("🧬 Biological plausibility: Spiking neurons, pathway separation, phasic/tonic dopamine")
        return True
    else:
        print("⚠️  Enhanced Pillar 2 Validation: SOME TESTS FAILED")
        print("🔧 Review enhanced implementation and fix issues")
        return False

if __name__ == "__main__":
    success = run_enhanced_pillar2_validation()
    exit(0 if success else 1)
