#!/usr/bin/env python3
"""
Test Brainstem Segmentation Integration

Tests the Phase 4 Step 2.O2 integration of brainstem segmentation into brain simulator startup.

Author: Quark AI
Date: 2025-09-16
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_segmentation_hook_import():
    """Test that the segmentation hook can be imported."""
    try:
        from brain.modules.brainstem_segmentation.segmentation_hook import BrainstemSegmentationHook, install_segmentation_hook
        print("✅ Segmentation hook imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import segmentation hook: {e}")
        return False

def test_inference_engine_import():
    """Test that the inference engine can be imported."""
    try:
        from brain.modules.brainstem_segmentation.inference_engine import BrainstemInferenceEngine, InferenceConfig
        print("✅ Inference engine imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import inference engine: {e}")
        return False

def test_hook_initialization():
    """Test that the segmentation hook can be initialized."""
    try:
        from brain.modules.brainstem_segmentation.segmentation_hook import BrainstemSegmentationHook

        hook = BrainstemSegmentationHook(auto_segment=True)
        print("✅ Segmentation hook initializes successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize segmentation hook: {e}")
        return False

def test_brain_simulator_integration():
    """Test integration with brain simulator (limited test)."""
    try:
        # Try to import brain simulator
        from brain.core.brain_simulator_init import BrainSimulator
        print("✅ Brain simulator can be imported")

        # Test hook installation (without actually running simulator)
        from brain.modules.brainstem_segmentation.segmentation_hook import install_segmentation_hook

        # Create mock brain simulator for testing
        class MockBrainSimulator:
            def __init__(self):
                self.hooks = []

        mock_brain = MockBrainSimulator()
        hook = install_segmentation_hook(mock_brain, auto_segment=True)

        if hasattr(mock_brain, 'hooks') and len(mock_brain.hooks) > 0:
            print("✅ Hook installation works with mock brain simulator")
            return True
        else:
            print("⚠️  Hook installation may have issues")
            return False

    except ImportError as e:
        print(f"⚠️  Brain simulator not available for testing: {e}")
        return False
    except Exception as e:
        print(f"❌ Brain simulator integration test failed: {e}")
        return False

def test_synthetic_segmentation():
    """Test segmentation with synthetic data."""
    try:
        from brain.modules.brainstem_segmentation.inference_engine import BrainstemInferenceEngine, InferenceConfig
        import numpy as np

        # Create inference engine
        config = InferenceConfig()
        engine = BrainstemInferenceEngine(config)

        # Create synthetic voxel data
        synthetic_data = np.random.rand(64, 64, 64).astype(np.float32)

        # Try segmentation
        result = engine.segment_volume(synthetic_data)

        if result is not None:
            print("✅ Synthetic data segmentation successful")
            print(f"   Segmentation shape: {result.shape}")
            print(f"   Unique labels: {len(np.unique(result))}")
            return True
        else:
            print("⚠️  Synthetic data segmentation returned None")
            return False

    except ImportError as e:
        print(f"⚠️  Inference engine not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Synthetic segmentation test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧠 Testing Brainstem Segmentation Integration")
    print("=" * 50)

    tests = [
        ("Hook Import", test_segmentation_hook_import),
        ("Inference Engine Import", test_inference_engine_import),
        ("Hook Initialization", test_hook_initialization),
        ("Brain Simulator Integration", test_brain_simulator_integration),
        ("Synthetic Segmentation", test_synthetic_segmentation)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1

    print(f"\n{'='*50}")
    print(f"📊 Integration Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All integration tests passed!")
        print("🚀 Phase 4 Step 2.O2 integration is ready for production")
        return True
    else:
        print("⚠️  Some integration tests failed - check for missing dependencies")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
