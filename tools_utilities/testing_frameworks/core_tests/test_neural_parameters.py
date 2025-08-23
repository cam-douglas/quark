"""
Test suite for neural parameters and neuromodulator systems.

Purpose: Validate neural parameter tuning and neuromodulator functionality
Inputs: Neural parameters, neuromodulator systems, homeostatic plasticity
Outputs: Test results and validation reports
Seeds: Fixed random seeds for reproducible testing
Deps: neural_parameters, numpy
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.neural_parameters import (
    NeuromodulatorType, NeuromodulatorLevel, NeuralParameters,
    NeuromodulatorSystem, HomeostaticPlasticity, Metaplasticity,
    NeuralParameterTuner, create_optimized_neural_parameters
)


class TestNeuralParameters(unittest.TestCase):
    """Test suite for neural parameters and neuromodulator systems."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # Fixed seed for reproducible tests
        
    def test_neuromodulator_type_enum(self):
        """Test neuromodulator type enumeration."""
        self.assertEqual(NeuromodulatorType.DOPAMINE.value, "dopamine")
        self.assertEqual(NeuromodulatorType.SEROTONIN.value, "serotonin")
        self.assertEqual(NeuromodulatorType.ACETYLCHOLINE.value, "acetylcholine")
        self.assertEqual(NeuromodulatorType.NOREPINEPHRINE.value, "norepinephrine")
        self.assertEqual(NeuromodulatorType.GABA.value, "gaba")
        
    def test_neuromodulator_level_creation(self):
        """Test neuromodulator level creation."""
        level = NeuromodulatorLevel(
            type=NeuromodulatorType.DOPAMINE,
            concentration=50.0,
            baseline=50.0,
            max_level=200.0,
            decay_rate=0.1
        )
        
        self.assertEqual(level.type, NeuromodulatorType.DOPAMINE)
        self.assertEqual(level.concentration, 50.0)
        self.assertEqual(level.baseline, 50.0)
        self.assertEqual(level.max_level, 200.0)
        self.assertEqual(level.decay_rate, 0.1)
        
    def test_neural_parameters_creation(self):
        """Test neural parameters creation."""
        params = NeuralParameters()
        
        self.assertEqual(params.membrane_threshold, -55.0)
        self.assertEqual(params.resting_potential, -70.0)
        self.assertEqual(params.reset_potential, -65.0)
        self.assertEqual(params.membrane_time_constant, 20.0)
        self.assertEqual(params.refractory_period, 2.0)
        self.assertEqual(params.target_firing_rate, 8.0)
        self.assertEqual(params.target_synchrony, 0.3)
        self.assertEqual(params.target_oscillation_power, 0.2)
        
    def test_neuromodulator_system_creation(self):
        """Test neuromodulator system creation."""
        system = NeuromodulatorSystem()
        
        # Check all neuromodulators exist
        self.assertIn(NeuromodulatorType.DOPAMINE, system.modulators)
        self.assertIn(NeuromodulatorType.SEROTONIN, system.modulators)
        self.assertIn(NeuromodulatorType.ACETYLCHOLINE, system.modulators)
        self.assertIn(NeuromodulatorType.NOREPINEPHRINE, system.modulators)
        self.assertIn(NeuromodulatorType.GABA, system.modulators)
        
        # Check modulator effects exist
        self.assertIn(NeuromodulatorType.DOPAMINE, system.modulator_effects)
        self.assertIn(NeuromodulatorType.SEROTONIN, system.modulator_effects)
        
    def test_modulator_level_update(self):
        """Test neuromodulator level update with decay."""
        system = NeuromodulatorSystem()
        
        # Get initial dopamine level
        initial_level = system.get_modulator_level(NeuromodulatorType.DOPAMINE)
        
        # Update levels
        system.update_modulator_levels(dt=1.0, current_time=1.0)
        
        # Check decay occurred
        new_level = system.get_modulator_level(NeuromodulatorType.DOPAMINE)
        self.assertLessEqual(new_level, initial_level)
        
    def test_modulator_release(self):
        """Test neuromodulator release."""
        system = NeuromodulatorSystem()
        
        # Get initial level
        initial_level = system.get_modulator_level(NeuromodulatorType.DOPAMINE)
        
        # Release dopamine
        system.release_modulator(NeuromodulatorType.DOPAMINE, 50.0, 0.0)
        
        # Check level increased
        new_level = system.get_modulator_level(NeuromodulatorType.DOPAMINE)
        self.assertGreater(new_level, initial_level)
        
    def test_modulated_parameters(self):
        """Test parameter modulation by neuromodulators."""
        system = NeuromodulatorSystem()
        base_params = NeuralParameters()
        
        # Release some neuromodulators to ensure modulation
        system.release_modulator(NeuromodulatorType.DOPAMINE, 25.0, 0.0)
        
        # Get modulated parameters
        modulated_params = system.get_modulated_parameters(base_params)
        
        # Check parameters were modulated
        self.assertIsInstance(modulated_params, NeuralParameters)
        self.assertNotEqual(modulated_params.membrane_threshold, base_params.membrane_threshold)
        
    def test_homeostatic_plasticity_creation(self):
        """Test homeostatic plasticity creation."""
        homeo = HomeostaticPlasticity(target_firing_rate=8.0)
        
        self.assertEqual(homeo.target_firing_rate, 8.0)
        self.assertEqual(homeo.scale_factor, 1.0)
        self.assertEqual(homeo.adaptation_rate, 0.01)
        
    def test_homeostatic_scale_factor_update(self):
        """Test homeostatic scale factor update."""
        homeo = HomeostaticPlasticity(target_firing_rate=8.0)
        
        # Update with high firing rate
        homeo.update_scale_factor(current_firing_rate=16.0, dt=0.001)
        
        # Scale factor should decrease (allow for small numerical differences)
        self.assertLessEqual(homeo.scale_factor, 1.0 + 1e-5)
        
        # Update with low firing rate
        homeo.update_scale_factor(current_firing_rate=4.0, dt=0.001)
        
        # Scale factor should increase
        self.assertGreater(homeo.scale_factor, 0.5)
        
    def test_homeostatic_weight_scaling(self):
        """Test homeostatic weight scaling."""
        homeo = HomeostaticPlasticity(target_firing_rate=8.0)
        homeo.scale_factor = 1.5
        
        base_weight = 0.1
        scaled_weight = homeo.get_scaled_weight(base_weight)
        
        self.assertEqual(scaled_weight, base_weight * 1.5)
        
    def test_metaplasticity_creation(self):
        """Test metaplasticity creation."""
        meta = Metaplasticity()
        
        self.assertEqual(meta.plasticity_threshold, 0.5)
        self.assertEqual(meta.metaplasticity_rate, 0.001)
        self.assertEqual(meta.learning_rate_modulation, 1.0)
        
    def test_metaplasticity_threshold_update(self):
        """Test metaplasticity threshold update."""
        meta = Metaplasticity()
        
        # Update with high activity
        meta.update_plasticity_threshold(activity_level=0.8, dt=0.001)
        
        # Threshold should increase
        self.assertGreater(meta.plasticity_threshold, 0.5)
        
        # Update with low activity
        meta.update_plasticity_threshold(activity_level=0.2, dt=0.001)
        
        # Threshold should decrease
        self.assertLess(meta.plasticity_threshold, 0.6)
        
    def test_metaplasticity_learning_rate_modulation(self):
        """Test metaplasticity learning rate modulation."""
        meta = Metaplasticity()
        
        base_lr = 0.01
        activity_level = 0.3
        
        modulated_lr = meta.get_modulated_learning_rate(base_lr, activity_level)
        
        # Learning rate should be modulated
        self.assertNotEqual(modulated_lr, base_lr)
        self.assertGreater(modulated_lr, 0.0)
        
    def test_neural_parameter_tuner_creation(self):
        """Test neural parameter tuner creation."""
        target_params = NeuralParameters()
        tuner = NeuralParameterTuner(target_params)
        
        self.assertEqual(tuner.target_params, target_params)
        self.assertIsInstance(tuner.current_params, NeuralParameters)
        self.assertEqual(len(tuner.tuning_history), 0)
        
    def test_parameter_tuning(self):
        """Test neural parameter tuning."""
        target_params = NeuralParameters()
        tuner = NeuralParameterTuner(target_params)
        
        # Tune parameters
        tuned_params = tuner.tune_parameters(
            current_firing_rate=4.0,  # Low firing rate
            current_synchrony=0.1,    # Low synchrony
            current_oscillation_power=0.05  # Low oscillation power
        )
        
        # Check parameters were tuned
        self.assertIsInstance(tuned_params, NeuralParameters)
        self.assertGreater(len(tuner.tuning_history), 0)
        
    def test_tuning_summary(self):
        """Test parameter tuning summary."""
        target_params = NeuralParameters()
        tuner = NeuralParameterTuner(target_params)
        
        # Perform some tuning
        tuner.tune_parameters(4.0, 0.1, 0.05)
        tuner.tune_parameters(6.0, 0.2, 0.1)
        
        # Get summary
        summary = tuner.get_tuning_summary()
        
        self.assertIn("total_adjustments", summary)
        self.assertIn("recent_firing_rate_error", summary)
        self.assertIn("current_membrane_threshold", summary)
        
    def test_optimized_neural_parameters(self):
        """Test optimized neural parameters creation."""
        params = create_optimized_neural_parameters()
        
        # Check optimized values
        self.assertEqual(params.membrane_threshold, -52.0)
        self.assertEqual(params.membrane_time_constant, 15.0)
        self.assertEqual(params.refractory_period, 1.5)
        self.assertEqual(params.target_firing_rate, 8.0)
        self.assertEqual(params.target_synchrony, 0.3)
        self.assertEqual(params.target_oscillation_power, 0.2)
        
    def test_neuromodulator_effects(self):
        """Test neuromodulator effects on parameters."""
        system = NeuromodulatorSystem()
        base_params = NeuralParameters()
        
        # Release high levels of dopamine
        system.release_modulator(NeuromodulatorType.DOPAMINE, 100.0, 0.0)
        
        # Get modulated parameters
        modulated_params = system.get_modulated_parameters(base_params)
        
        # Check dopamine effects
        self.assertLess(modulated_params.membrane_threshold, base_params.membrane_threshold)
        self.assertGreater(modulated_params.target_firing_rate, base_params.target_firing_rate)
        
    def test_multiple_modulator_interaction(self):
        """Test interaction between multiple neuromodulators."""
        system = NeuromodulatorSystem()
        base_params = NeuralParameters()
        
        # Release multiple modulators
        system.release_modulator(NeuromodulatorType.DOPAMINE, 50.0, 0.0)
        system.release_modulator(NeuromodulatorType.SEROTONIN, 30.0, 0.0)
        system.release_modulator(NeuromodulatorType.GABA, 40.0, 0.0)
        
        # Get modulated parameters
        modulated_params = system.get_modulated_parameters(base_params)
        
        # Check combined effects
        self.assertNotEqual(modulated_params.membrane_threshold, base_params.membrane_threshold)
        self.assertNotEqual(modulated_params.target_firing_rate, base_params.target_firing_rate)
        
    def test_homeostatic_metaplasticity_integration(self):
        """Test integration of homeostatic plasticity and metaplasticity."""
        homeo = HomeostaticPlasticity(target_firing_rate=8.0)
        meta = Metaplasticity()
        
        # Simulate activity
        current_firing_rate = 12.0
        activity_level = 0.7
        
        # Update both systems
        homeo.update_scale_factor(current_firing_rate, dt=0.001)
        meta.update_plasticity_threshold(activity_level, dt=0.001)
        
        # Check both systems updated
        self.assertNotEqual(homeo.scale_factor, 1.0)
        self.assertNotEqual(meta.plasticity_threshold, 0.5)
        
        # Test combined effect on learning rate
        base_lr = 0.01
        scaled_weight = homeo.get_scaled_weight(1.0)
        modulated_lr = meta.get_modulated_learning_rate(base_lr, activity_level)
        
        # Both should affect the system
        self.assertNotEqual(scaled_weight, 1.0)
        self.assertNotEqual(modulated_lr, base_lr)


def run_neural_parameters_test_suite():
    """Run the neural parameters test suite."""
    print("🧠 Running Neural Parameters Test Suite")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNeuralParameters)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_neural_parameters_test_suite()
    sys.exit(0 if success else 1)
