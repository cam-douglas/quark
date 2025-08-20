#!/usr/bin/env python3
"""
Simple Brain Simulation Approach Demo

Shows the differences between:
1. MuJoCo: Physical brain development
2. NEST: Neural network development  
3. Hybrid: Combined approach
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_mujoco():
    """Test MuJoCo approach"""
    print("🔬 Testing MuJoCo (Physical) approach...")
    try:
        from physics_simulation.mujoco_interface import MuJoCoInterface
        from physics_simulation.dual_mode_simulator import DualModeBrainSimulator
        
        mujoco_interface = MuJoCoInterface()
        simulator = DualModeBrainSimulator(simulation_mode="mujoco", mujoco_interface=mujoco_interface)
        
        brain_regions = ["cortex", "hippocampus"]
        cell_types = ["neurons", "glia"]
        
        success = simulator.setup_brain_development_model(brain_regions, cell_types)
        if success:
            results = simulator.simulate_brain_growth(2.0)
            print(f"✅ MuJoCo: {len(results['time_points'])} time points")
            return True
        return False
    except Exception as e:
        print(f"❌ MuJoCo failed: {e}")
        return False

def test_nest():
    """Test NEST approach"""
    print("🧠 Testing NEST (Neural) approach...")
    try:
        from physics_simulation.dual_mode_simulator import DualModeBrainSimulator
        
        simulator = DualModeBrainSimulator(simulation_mode="nest")
        
        brain_regions = ["cortex", "hippocampus"]
        cell_types = ["excitatory", "inhibitory"]
        region_sizes = {"cortex": 500, "hippocampus": 300}
        
        success = simulator.setup_brain_development_model(brain_regions, cell_types, region_sizes)
        if success:
            results = simulator.simulate_brain_growth(50.0)
            print(f"✅ NEST: {len(results['time_points'])} time points")
            return True
        return False
    except Exception as e:
        print(f"❌ NEST failed: {e}")
        return False

def test_hybrid():
    """Test hybrid approach"""
    print("🚀 Testing Hybrid approach...")
    try:
        from physics_simulation.mujoco_interface import MuJoCoInterface
        from physics_simulation.dual_mode_simulator import DualModeBrainSimulator
        
        mujoco_interface = MuJoCoInterface()
        simulator = DualModeBrainSimulator(simulation_mode="hybrid", mujoco_interface=mujoco_interface)
        
        brain_regions = ["cortex", "hippocampus"]
        cell_types = ["excitatory", "inhibitory"]
        region_sizes = {"cortex": 300, "hippocampus": 200}
        
        success = simulator.setup_brain_development_model(brain_regions, cell_types, region_sizes)
        if success:
            results = simulator.simulate_brain_growth(1.0)
            print(f"✅ Hybrid: {len(results['time_points'])} time points")
            return True
        return False
    except Exception as e:
        print(f"❌ Hybrid failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Brain Simulation Approaches")
    print("=" * 50)
    
    results = {
        "MuJoCo (Physical)": test_mujoco(),
        "NEST (Neural)": test_nest(),
        "Hybrid (Combined)": test_hybrid()
    }
    
    print("\n📊 Results:")
    print("-" * 30)
    for approach, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{approach}: {status}")
    
    working = [k for k, v in results.items() if v]
    if working:
        print(f"\n🎉 Working approaches: {', '.join(working)}")
        print("\n💡 When to use each approach:")
        print("🔬 MuJoCo: Tissue mechanics, spatial development, biomechanics")
        print("🧠 NEST: Neural circuits, learning, network dynamics")
        print("🚀 Hybrid: Multi-scale effects, physical-neural interactions")
    else:
        print("\n❌ No approaches worked. Check your setup.")

if __name__ == "__main__":
    main()
