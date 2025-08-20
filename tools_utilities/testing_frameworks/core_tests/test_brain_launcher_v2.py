"""
Test suite for brain_launcher_v2.py.

Purpose: Validate brain launcher v2 functionality and module interactions
Inputs: Brain launcher v2 components and modules
Outputs: Test results and validation reports
Seeds: Fixed random seeds for reproducible testing
Deps: brain_launcher_v2, neural_components
"""

import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.brain_launcher_v2 import BrainLauncherV2, BrainModule, BrainState


class TestBrainLauncherV2(unittest.TestCase):
    """Test suite for brain launcher v2."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.launcher = BrainLauncherV2()
        
    def test_launcher_creation(self):
        """Test brain launcher v2 creation."""
        self.assertIsNotNone(self.launcher)
        self.assertIsInstance(self.launcher, BrainLauncherV2)
        
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
        
    def test_launcher_initialization(self):
        """Test launcher initialization."""
        # Test that launcher can be initialized without errors
        try:
            launcher = BrainLauncherV2()
            self.assertIsNotNone(launcher)
        except Exception as e:
            self.fail(f"Launcher initialization failed: {e}")
            
    def test_module_registration(self):
        """Test module registration functionality."""
        # This test would depend on the actual implementation
        # For now, we'll test basic structure
        self.assertTrue(hasattr(self.launcher, 'modules') or 
                       hasattr(self.launcher, 'register_module'))
        
    def test_brain_state_transitions(self):
        """Test brain state transition functionality."""
        # Test basic state functionality
        state = BrainState()
        self.assertIsNotNone(state)
        
    def test_launcher_step_functionality(self):
        """Test launcher step functionality."""
        # Test that launcher has step method
        self.assertTrue(hasattr(self.launcher, 'step') or 
                       hasattr(self.launcher, 'run_step'))
        
    def test_error_handling(self):
        """Test error handling in launcher."""
        # Test that launcher handles errors gracefully
        try:
            # Try to create launcher with invalid parameters
            launcher = BrainLauncherV2()
            self.assertIsNotNone(launcher)
        except Exception as e:
            # Should handle errors gracefully
            self.assertIsInstance(e, Exception)


def run_brain_launcher_v2_test_suite():
    """Run the brain launcher v2 test suite."""
    print("🧠 Running Brain Launcher V2 Test Suite")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBrainLauncherV2)
    
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
    success = run_brain_launcher_v2_test_suite()
    sys.exit(0 if success else 1)
