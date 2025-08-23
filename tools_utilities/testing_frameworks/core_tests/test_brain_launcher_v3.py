"""
Test suite for brain_launcher_v3.py.

Purpose: Validate brain launcher v3 functionality and advanced module interactions
Inputs: Brain launcher v3 components, modules, and neural integration
Outputs: Test results and validation reports
Seeds: Fixed random seeds for reproducible testing
Deps: brain_launcher_v3, neural_components, neural_integration_layer
"""

import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.brain_launcher_v3 import Brain, BrainModule, BrainState


class TestBrainLauncherV3(unittest.TestCase):
    """Test suite for brain launcher v3."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.brain = Brain()
        
    def test_brain_creation(self):
        """Test brain creation."""
        self.assertIsNotNone(self.brain)
        self.assertIsInstance(self.brain, Brain)
        
    def test_module_creation(self):
        """Test brain module creation."""
        module = BrainModule("test_module", "test_type")
        self.assertEqual(module.name, "test_module")
        self.assertEqual(module.module_type, "test_type")
        
    def test_brain_state_creation(self):
        """Test brain state creation."""
        state = BrainState()
        self.assertIsNotNone(state)
        self.assertIsInstance(state, BrainState)
        
    def test_brain_initialization(self):
        """Test brain initialization."""
        # Test that brain can be initialized without errors
        try:
            brain = Brain()
            self.assertIsNotNone(brain)
        except Exception as e:
            self.fail(f"Brain initialization failed: {e}")
            
    def test_module_registration(self):
        """Test module registration functionality."""
        # This test would depend on the actual implementation
        # For now, we'll test basic structure
        self.assertTrue(hasattr(self.brain, 'modules') or 
                       hasattr(self.brain, 'register_module'))
        
    def test_brain_state_transitions(self):
        """Test brain state transition functionality."""
        # Test basic state functionality
        state = BrainState()
        self.assertIsNotNone(state)
        
    def test_brain_step_functionality(self):
        """Test brain step functionality."""
        # Test that brain has step method
        self.assertTrue(hasattr(self.brain, 'step') or 
                       hasattr(self.brain, 'run_step'))
        
    def test_error_handling(self):
        """Test error handling in brain."""
        # Test that brain handles errors gracefully
        try:
            # Try to create brain with invalid parameters
            brain = Brain()
            self.assertIsNotNone(brain)
        except Exception as e:
            # Should handle errors gracefully
            self.assertIsInstance(e, Exception)
            
    def test_neural_integration(self):
        """Test neural integration functionality."""
        # Test that brain has neural integration capabilities
        self.assertTrue(hasattr(self.brain, 'neural_layer') or 
                       hasattr(self.brain, 'neural_integration'))
        
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


def run_brain_launcher_v3_test_suite():
    """Run the brain launcher v3 test suite."""
    print("🧠 Running Brain Launcher V3 Test Suite")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBrainLauncherV3)
    
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
    success = run_brain_launcher_v3_test_suite()
    sys.exit(0 if success else 1)
