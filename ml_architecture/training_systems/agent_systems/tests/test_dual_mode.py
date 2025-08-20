#!/usr/bin/env python3
"""
Simple test for dual-mode brain physics simulator
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from physics_simulation.mujoco_interface import MuJoCoInterface
from physics_simulation.brain_physics import BrainPhysicsSimulator

def test_mujoco_mode():
    """Test MuJoCo mode"""
    print("🔬 Testing MuJoCo mode...")
    
    mujoco_interface = MuJoCoInterface()
    brain_sim = BrainPhysicsSimulator(
        simulation_mode="mujoco",
        mujoco_interface=mujoco_interface
    )
    
    brain_regions = ["cortex", "hippocampus"]
    cell_types = ["neurons", "glia"]
    
    success = brain_sim.setup_brain_development_model(brain_regions, cell_types)
    if success:
        results = brain_sim.simulate_brain_growth(2.0)
        print(f"✅ MuJoCo mode: {len(results['time_points'])} time points")
        return True
    return False

def test_nest_mode():
    """Test NEST mode"""
    print("🧠 Testing NEST mode...")
    
    brain_sim = BrainPhysicsSimulator(simulation_mode="nest")
    
    brain_regions = ["cortex", "hippocampus"]
    cell_types = ["excitatory", "inhibitory"]
    region_sizes = {"cortex": 500, "hippocampus": 300}
    
    success = brain_sim.setup_brain_development_model(brain_regions, cell_types, region_sizes)
    if success:
        results = brain_sim.simulate_brain_growth(50.0)
        print(f"✅ NEST mode: {len(results['time_points'])} time points")
        return True
    return False

def test_hybrid_mode():
    """Test hybrid mode"""
    print("🚀 Testing hybrid mode...")
    
    mujoco_interface = MuJoCoInterface()
    brain_sim = BrainPhysicsSimulator(
        simulation_mode="hybrid",
        mujoco_interface=mujoco_interface
    )
    
    brain_regions = ["cortex", "hippocampus"]
    cell_types = ["excitatory", "inhibitory"]
    region_sizes = {"cortex": 300, "hippocampus": 200}
    
    success = brain_sim.setup_brain_development_model(brain_regions, cell_types, region_sizes)
    if success:
        results = brain_sim.simulate_brain_growth(1.0)
        print(f"✅ Hybrid mode: {len(results['time_points'])} time points")
        return True
    return False

def main():
    """Run all tests"""
    print("🧪 Testing Dual-Mode Brain Physics Simulator")
    print("=" * 50)
    
    tests = [
        ("MuJoCo Mode", test_mujoco_mode),
        ("NEST Mode", test_nest_mode),
        ("Hybrid Mode", test_hybrid_mode)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results[test_name] = False
    
    print("\n📊 Test Results:")
    print("-" * 30)
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
