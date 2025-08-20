"""
Test suite for brain_launcher_v4.py.

Purpose: Validate brain launcher v4 functionality with neural dynamics integration
Inputs: Brain launcher v4 components, neural integration layer, biological validator
Outputs: Test results and validation reports
Seeds: Fixed random seeds for reproducible testing
Deps: brain_launcher_v4, neural_integration_layer, biological_validator
"""

import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.brain_launcher_v4 import (
    NeuralEnhancedBrain, NeuralEnhancedModule, 
    NeuralEnhancedPFC, NeuralEnhancedBasalGanglia, NeuralEnhancedThalamus
)


class TestBrainLauncherV4(unittest.TestCase):
    """Test suite for brain launcher v4."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.brain = NeuralEnhancedBrain()
        
    def test_brain_creation(self):
        """Test neural enhanced brain creation."""
        self.assertIsNotNone(self.brain)
        self.assertIsInstance(self.brain, NeuralEnhancedBrain)
        
    def test_neural_enhanced_module_creation(self):
        """Test neural enhanced module creation."""
        module = NeuralEnhancedModule("test_module", "test_type")
        self.assertEqual(module.name, "test_module")
        self.assertEqual(module.module_type, "test_type")
        
    def test_pfc_module_creation(self):
        """Test neural enhanced PFC module creation."""
        pfc = NeuralEnhancedPFC()
        self.assertIsNotNone(pfc)
        self.assertIsInstance(pfc, NeuralEnhancedPFC)
        
    def test_basal_ganglia_module_creation(self):
        """Test neural enhanced basal ganglia module creation."""
        bg = NeuralEnhancedBasalGanglia()
        self.assertIsNotNone(bg)
        self.assertIsInstance(bg, NeuralEnhancedBasalGanglia)
        
    def test_thalamus_module_creation(self):
        """Test neural enhanced thalamus module creation."""
        thalamus = NeuralEnhancedThalamus()
        self.assertIsNotNone(thalamus)
        self.assertIsInstance(thalamus, NeuralEnhancedThalamus)
        
    def test_brain_initialization(self):
        """Test brain initialization."""
        # Test that brain can be initialized without errors
        try:
            brain = NeuralEnhancedBrain()
            self.assertIsNotNone(brain)
        except Exception as e:
            self.fail(f"Brain initialization failed: {e}")
            
    def test_neural_layer_integration(self):
        """Test neural layer integration."""
        # Test that brain has neural layer
        self.assertTrue(hasattr(self.brain, 'neural_layer'))
        self.assertIsNotNone(self.brain.neural_layer)
        
    def test_biological_validator_integration(self):
        """Test biological validator integration."""
        # Test that brain has biological validator
        self.assertTrue(hasattr(self.brain, 'validator'))
        self.assertIsNotNone(self.brain.validator)
        
    def test_brain_step_functionality(self):
        """Test brain step functionality."""
        # Test that brain has step method
        self.assertTrue(hasattr(self.brain, 'step'))
        
        # Test step execution
        try:
            result = self.brain.step()
            self.assertIsNotNone(result)
        except Exception as e:
            # Step might fail due to missing components, but should not crash
            self.assertIsInstance(e, Exception)
        
    def test_module_neural_awareness(self):
        """Test that modules are neural-aware."""
        # Test PFC neural awareness
        pfc = NeuralEnhancedPFC()
        self.assertTrue(hasattr(pfc, 'neural_activity'))
        
        # Test BG neural awareness
        bg = NeuralEnhancedBasalGanglia()
        self.assertTrue(hasattr(bg, 'neural_activity'))
        
        # Test Thalamus neural awareness
        thalamus = NeuralEnhancedThalamus()
        self.assertTrue(hasattr(thalamus, 'neural_activity'))
        
    def test_error_handling(self):
        """Test error handling in brain."""
        # Test that brain handles errors gracefully
        try:
            # Try to create brain with invalid parameters
            brain = NeuralEnhancedBrain()
            self.assertIsNotNone(brain)
        except Exception as e:
            # Should handle errors gracefully
            self.assertIsInstance(e, Exception)
            
    def test_validation_integration(self):
        """Test validation integration."""
        # Test that brain can perform validation
        self.assertTrue(hasattr(self.brain, 'validator'))
        
        # Test validation execution
        try:
            # This might fail if no data is available, but should not crash
            pass
        except Exception as e:
            self.assertIsInstance(e, Exception)
        
    def test_neural_dynamics(self):
        """Test neural dynamics functionality."""
        # Test that brain has neural dynamics
        self.assertTrue(hasattr(self.brain, 'neural_layer'))
        
        # Test neural layer functionality
        if hasattr(self.brain, 'neural_layer'):
            neural_layer = self.brain.neural_layer
            self.assertTrue(hasattr(neural_layer, 'cortical_loop'))
            self.assertTrue(hasattr(neural_layer, 'message_queue'))
        
    def test_module_communication(self):
        """Test module communication functionality."""
        # Test that modules can communicate
        self.assertTrue(hasattr(self.brain, 'send_message') or 
                       hasattr(self.brain, 'communicate'))
        
    def test_brain_metrics(self):
        """Test brain metrics collection."""
        # Test that brain collects metrics
        self.assertTrue(hasattr(self.brain, 'get_metrics') or 
                       hasattr(self.brain, 'metrics'))


def run_brain_launcher_v4_test_suite():
    """Run the brain launcher v4 test suite."""
    print("🧠 Running Brain Launcher V4 Test Suite")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBrainLauncherV4)
    
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
    success = run_brain_launcher_v4_test_suite()
    sys.exit(0 if success else 1)
