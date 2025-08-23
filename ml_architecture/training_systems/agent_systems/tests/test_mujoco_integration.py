#!/usr/bin/env python3
"""
Test script for MuJoCo integration

This script tests the basic functionality of the MuJoCo physics integration
to ensure everything is working correctly.
"""

import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_mujoco_import():
    """Test MuJoCo import"""
    try:
        import mujoco
        print(f"✅ MuJoCo imported successfully: version {mujoco.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import MuJoCo: {e}")
        return False

def test_physics_simulation_import():
    """Test physics simulation module import"""
    try:
        from physics_simulation import (
            MuJoCoInterface, 
            BrainPhysicsSimulator, 
            TissueMechanics, 
            MorphogenPhysics
        )
        print("✅ Physics simulation modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import physics simulation modules: {e}")
        return False

def test_mujoco_interface():
    """Test MuJoCo interface creation"""
    try:
        from physics_simulation import MuJoCoInterface
        
        # Create interface
        interface = MuJoCoInterface()
        print("✅ MuJoCo interface created successfully")
        
        # Test model creation
        brain_regions = ["cortex", "hippocampus"]
        cell_types = ["neurons", "glial_cells"]
        
        model_path = interface.create_brain_development_model(brain_regions, cell_types)
        print(f"✅ Brain development model created: {model_path}")
        
        # Test model loading
        success = interface.load_model(model_path)
        if success:
            print("✅ Model loaded successfully")
        else:
            print("❌ Model loading failed")
            return False
        
        # Test simulation stats
        stats = interface.get_simulation_stats()
        print(f"✅ Simulation stats retrieved: {stats['num_bodies']} bodies")
        
        # Cleanup
        interface.close()
        print("✅ MuJoCo interface closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ MuJoCo interface test failed: {e}")
        traceback.print_exc()
        return False

def test_tissue_mechanics():
    """Test tissue mechanics"""
    try:
        from physics_simulation import TissueMechanics
        import numpy as np
        
        # Create tissue mechanics
        tissue_mechanics = TissueMechanics()
        print("✅ Tissue mechanics created successfully")
        
        # Test elastic deformation
        deformation = tissue_mechanics.calculate_elastic_deformation(
            'cortex',
            np.array([100.0, 0.0, 0.0]),
            0.001
        )
        print(f"✅ Elastic deformation calculated: {float(deformation['deformation']):.6f} m")
        
        # Test tissue growth
        growth = tissue_mechanics.simulate_tissue_growth(
            'cortex', 0.01, 2.0, 0.001
        )
        print(f"✅ Tissue growth simulated: {len(growth['time_points'])} time points")
        
        return True
        
    except Exception as e:
        print(f"❌ Tissue mechanics test failed: {e}")
        traceback.print_exc()
        return False

def test_morphogen_physics():
    """Test morphogen physics"""
    try:
        from physics_simulation import MorphogenPhysics
        
        # Create morphogen physics
        morphogen_physics = MorphogenPhysics(grid_size=20, domain_size=0.01)
        print("✅ Morphogen physics created successfully")
        
        # Add sources and sinks
        morphogen_physics.add_morphogen_source('shh', 10, 10)
        morphogen_physics.add_morphogen_sink('shh', 0, 0)
        print("✅ Morphogen sources and sinks added")
        
        # Test diffusion
        results = morphogen_physics.step_diffusion(steps=5)
        print(f"✅ Diffusion simulation completed: {len(results['time'])} time steps")
        
        # Test patterning simulation
        pattern_results = morphogen_physics.simulate_developmental_patterning(duration=1.0)
        print(f"✅ Patterning simulation completed: {len(pattern_results['time_points'])} time points")
        
        return True
        
    except Exception as e:
        print(f"❌ Morphogen physics test failed: {e}")
        traceback.print_exc()
        return False

def test_brain_physics_simulator():
    """Test brain physics simulator"""
    try:
        from physics_simulation import MuJoCoInterface, BrainPhysicsSimulator
        
        # Create interface and simulator
        interface = MuJoCoInterface()
        brain_simulator = BrainPhysicsSimulator(interface)
        print("✅ Brain physics simulator created successfully")
        
        # Setup model
        brain_regions = ["cortex", "hippocampus"]
        cell_types = ["neurons", "glial_cells"]
        
        setup_success = brain_simulator.setup_brain_development_model(brain_regions, cell_types)
        if not setup_success:
            print("❌ Failed to setup brain development model")
            interface.close()
            return False
        
        print("✅ Brain development model setup completed")
        
        # Test growth simulation
        growth_results = brain_simulator.simulate_brain_growth(duration=1.0)
        print(f"✅ Brain growth simulation completed: {len(growth_results['time_points'])} time points")
        
        # Test metrics
        metrics = brain_simulator.get_development_metrics()
        print(f"✅ Development metrics retrieved: {metrics['region_count']} regions")
        
        # Cleanup
        interface.close()
        print("✅ Brain physics simulator test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Brain physics simulator test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing MuJoCo Integration for SmallMind")
    print("=" * 50)
    
    tests = [
        ("MuJoCo Import", test_mujoco_import),
        ("Physics Simulation Import", test_physics_simulation_import),
        ("MuJoCo Interface", test_mujoco_interface),
        ("Tissue Mechanics", test_tissue_mechanics),
        ("Morphogen Physics", test_morphogen_physics),
        ("Brain Physics Simulator", test_brain_physics_simulator),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MuJoCo integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
