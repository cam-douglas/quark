"""
Test suite for neural dynamics integration components.

Purpose: Validate neural integration layer and biological validation framework
Inputs: Neural components, integration layer, biological validator
Outputs: Test results and validation reports
Seeds: Fixed random seeds for reproducible testing
Deps: neural_components, neural_integration_layer, biological_validator
"""

import unittest
import numpy as np
import json
import tempfile
import os, sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.neural_integration_layer import (
    MessageType, NeuralMessage, NeuralPopulation, 
    CorticalSubcorticalLoop, NeuralIntegrationLayer
)
from src.core.biological_validator import (
    ValidationLevel, BiologicalBenchmark, ValidationResult, 
    BiologicalValidator
)
from src.core.neural_components import (
    SpikingNeuron, HebbianSynapse, STDP, NeuralPopulation as NeuralPopulationComponent
)


class TestNeuralDynamicsIntegration(unittest.TestCase):
    """Test suite for neural dynamics integration components."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # Fixed seed for reproducible tests
        
    def test_neural_population_creation(self):
        """Test neural population creation and basic functionality."""
        # Test PFC population creation
        pfc_pop = NeuralPopulation("PFC", 100, "excitatory")
        self.assertEqual(pfc_pop.name, "PFC")
        self.assertEqual(pfc_pop.neuron_count, 100)
        self.assertEqual(pfc_pop.neuron_type, "excitatory")
        
        # Test BG population creation
        bg_pop = NeuralPopulation("BG", 50, "inhibitory")
        self.assertEqual(bg_pop.name, "BG")
        self.assertEqual(bg_pop.neuron_count, 50)
        self.assertEqual(bg_pop.neuron_type, "inhibitory")
        
    def test_cortical_subcortical_loop_creation(self):
        """Test cortical-subcortical loop creation."""
        loop = CorticalSubcorticalLoop()
        
        # Check populations exist
        self.assertIsNotNone(loop.pfc_population)
        self.assertIsNotNone(loop.bg_population)
        self.assertIsNotNone(loop.thalamus_population)
        
        # Check synapses exist
        self.assertIsNotNone(loop.pfc_to_bg)
        self.assertIsNotNone(loop.bg_to_thalamus)
        self.assertIsNotNone(loop.thalamus_to_pfc)
        
        # Check STDP mechanisms exist
        self.assertIsNotNone(loop.pfc_bg_stdp)
        self.assertIsNotNone(loop.bg_thalamus_stdp)
        self.assertIsNotNone(loop.thalamus_pfc_stdp)
        
    def test_loop_dynamics(self):
        """Test cortical-subcortical loop dynamics."""
        loop = CorticalSubcorticalLoop()
        
        # Test single step
        dt = 0.001  # 1ms
        external_input = 0.1
        
        result = loop.step(dt, external_input)
        
        # Check result structure
        self.assertIn("pfc_output", result)
        self.assertIn("bg_output", result)
        self.assertIn("thalamus_output", result)
        self.assertIn("loop_metrics", result)
        
        # Check output structure
        pfc_output = result["pfc_output"]
        self.assertIn("spikes", pfc_output)
        self.assertIn("firing_rate", pfc_output)
        self.assertIn("synchrony", pfc_output)
        self.assertIn("oscillation_power", pfc_output)
        
    def test_message_conversion(self):
        """Test message-to-spike and spike-to-message conversion."""
        integration_layer = NeuralIntegrationLayer()
        
        # Test message to spike conversion
        message = NeuralMessage(
            message_type=MessageType.EXECUTIVE_CONTROL,
            content="test_command",
            priority=0.8,
            source="PFC",
            target="BG"
        )
        
        spike_pattern = integration_layer.message_to_spikes(message)
        self.assertIsInstance(spike_pattern, list)
        self.assertTrue(len(spike_pattern) > 0)
        
        # Test spike to message conversion
        converted_message = integration_layer.spikes_to_message(spike_pattern)
        self.assertIsInstance(converted_message, NeuralMessage)
        self.assertEqual(converted_message.message_type, message.message_type)
        
    def test_neural_integration_layer_creation(self):
        """Test neural integration layer creation."""
        integration_layer = NeuralIntegrationLayer()
        
        # Check components exist
        self.assertIsNotNone(integration_layer.cortical_loop)
        self.assertIsNotNone(integration_layer.message_queue)
        self.assertIsNotNone(integration_layer.neural_state)
        
    def test_biological_validator_creation(self):
        """Test biological validator creation."""
        validator = BiologicalValidator()
        
        # Check benchmarks exist
        self.assertGreater(len(validator.benchmarks), 0)
        
        # Check validation levels
        self.assertIn(ValidationLevel.BASIC, validator.validation_levels)
        self.assertIn(ValidationLevel.ADVANCED, validator.validation_levels)
        self.assertIn(ValidationLevel.EXPERT, validator.validation_levels)
        
    def test_firing_rate_validation(self):
        """Test firing rate validation against biological benchmarks."""
        validator = BiologicalValidator()
        
        # Create test data
        test_data = {
            "firing_rates": [5.0, 10.0, 15.0],  # Hz
            "synchrony": [0.3, 0.5, 0.7],
            "oscillation_power": [0.1, 0.2, 0.3]
        }
        
        # Validate firing rates
        result = validator.validate_firing_rates(test_data["firing_rates"])
        self.assertIsInstance(result, ValidationResult)
        self.assertIn("firing_rate", result.metrics)
        
    def test_synchrony_validation(self):
        """Test synchrony validation."""
        validator = BiologicalValidator()
        
        # Test synchrony values
        synchrony_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = validator.validate_synchrony(synchrony_values)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertIn("synchrony", result.metrics)
        
    def test_oscillation_validation(self):
        """Test oscillation power validation."""
        validator = BiologicalValidator()
        
        # Test oscillation power values
        oscillation_values = [0.05, 0.1, 0.2, 0.3]
        result = validator.validate_oscillation_power(oscillation_values)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertIn("oscillation_power", result.metrics)
        
    def test_loop_stability_validation(self):
        """Test cortical-subcortical loop stability validation."""
        validator = BiologicalValidator()
        
        # Create test loop data
        loop_data = {
            "pfc_firing_rate": 8.0,
            "bg_firing_rate": 12.0,
            "thalamus_firing_rate": 6.0,
            "loop_gain": 0.8,
            "stability_metric": 0.7
        }
        
        result = validator.validate_loop_stability(loop_data)
        self.assertIsInstance(result, ValidationResult)
        self.assertIn("loop_stability", result.metrics)
        
    def test_plasticity_validation(self):
        """Test synaptic plasticity validation."""
        validator = BiologicalValidator()
        
        # Test plasticity data
        plasticity_data = {
            "weight_changes": [0.01, 0.02, -0.01, 0.03],
            "learning_rates": [0.01, 0.008, 0.012],
            "plasticity_events": 150
        }
        
        result = validator.validate_plasticity(plasticity_data)
        self.assertIsInstance(result, ValidationResult)
        self.assertIn("plasticity", result.metrics)
        
    def test_comprehensive_validation(self):
        """Test comprehensive validation of neural dynamics."""
        validator = BiologicalValidator()
        
        # Create comprehensive test data
        test_data = {
            "firing_rates": [5.0, 10.0, 15.0],
            "synchrony": [0.3, 0.5, 0.7],
            "oscillation_power": [0.1, 0.2, 0.3],
            "loop_stability": 0.8,
            "plasticity_metrics": {
                "weight_changes": [0.01, 0.02, -0.01],
                "learning_rates": [0.01, 0.008, 0.012]
            }
        }
        
        result = validator.validate_comprehensive(test_data)
        self.assertIsInstance(result, ValidationResult)
        self.assertGreater(len(result.metrics), 0)
        
    def test_validation_data_export(self):
        """Test validation data export functionality."""
        validator = BiologicalValidator()
        
        # Create test validation result
        test_result = ValidationResult(
            level=ValidationLevel.BASIC,
            metrics={"test_metric": 0.5},
            status="PASS",
            details="Test validation completed successfully"
        )
        
        # Test export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
            
        try:
            validator.export_validation_data([test_result], temp_filename)
            
            # Check file was created and contains data
            self.assertTrue(os.path.exists(temp_filename))
            
            with open(temp_filename, 'r') as f:
                data = json.load(f)
                self.assertIn("validation_results", data)
                
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
                
    def test_neural_population_step(self):
        """Test neural population stepping with realistic input."""
        # Create population with realistic parameters
        pop = NeuralPopulation("Test", 10, "excitatory")
        
        # Step with sufficient input to trigger spikes
        dt = 0.001
        input_current = 1.0  # Higher input to ensure spikes
        
        result = pop.step(dt, input_current)
        
        # Check result structure
        self.assertIn("spikes", result)
        self.assertIn("firing_rate", result)
        self.assertIn("synchrony", result)
        self.assertIn("oscillation_power", result)
        self.assertIn("active_neurons", result)
        
        # Check data types
        self.assertIsInstance(result["spikes"], list)
        self.assertIsInstance(result["firing_rate"], float)
        self.assertIsInstance(result["synchrony"], float)
        self.assertIsInstance(result["oscillation_power"], float)
        self.assertIsInstance(result["active_neurons"], int)


def run_neural_dynamics_test_suite():
    """Run the complete neural dynamics integration test suite."""
    print("🧠 Running Neural Dynamics Integration Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNeuralDynamicsIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
            
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_neural_dynamics_test_suite()
    sys.exit(0 if success else 1)
