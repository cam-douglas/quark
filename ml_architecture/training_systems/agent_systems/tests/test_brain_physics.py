#!/usr/bin/env python3
"""
Test script for brain physics simulation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from physics_simulation.mujoco_interface import MuJoCoInterface
from physics_simulation.brain_physics import BrainPhysicsSimulator
import numpy as np

def test_brain_physics():
    """Test the brain physics simulation"""
    print("🧠 Testing Brain Physics Simulation...")
    
    try:
        # Initialize MuJoCo interface
        print("  📦 Initializing MuJoCo interface...")
        mujoco_interface = MuJoCoInterface()
        
        # Initialize brain physics simulator
        print("  🧬 Creating brain physics simulator...")
        brain_sim = BrainPhysicsSimulator(mujoco_interface)
        
        # Setup brain development model
        brain_regions = ["cortex", "hippocampus", "thalamus", "cerebellum"]
        cell_types = ["neurons", "glia", "endothelial"]
        
        print(f"  🏗️  Setting up brain model with regions: {brain_regions}")
        print(f"  🧫 Cell types: {cell_types}")
        
        success = brain_sim.setup_brain_development_model(brain_regions, cell_types)
        if not success:
            print("  ❌ Failed to setup brain development model")
            return False
        
        print("  ✅ Brain development model setup successful")
        
        # Run a short simulation
        print("  ⏱️  Running brain growth simulation...")
        duration = 1.0  # 1 second simulation
        results = brain_sim.simulate_brain_growth(duration)
        
        print(f"  📊 Simulation completed:")
        print(f"    - Time points: {len(results['time_points'])}")
        print(f"    - Region sizes tracked: {len(results['region_sizes'])}")
        print(f"    - Growth rates tracked: {len(results['growth_rates'])}")
        
        # Get development metrics
        metrics = brain_sim.get_development_metrics()
        print(f"  📈 Development metrics:")
        print(f"    - Development stage: {metrics['development_stage']:.3f}")
        print(f"    - Region count: {metrics['region_count']}")
        print(f"    - Cell population: {metrics['cell_population']}")
        print(f"    - Average growth rate: {metrics['average_growth_rate']:.3f}")
        
        # Get simulation stats
        stats = mujoco_interface.get_simulation_stats()
        print(f"  🔬 Simulation statistics:")
        print(f"    - Bodies: {stats['num_bodies']}")
        print(f"    - Joints: {stats['num_joints']}")
        print(f"    - Actuators: {stats['num_actuators']}")
        print(f"    - Total energy: {stats['total_energy']:.6f}")
        
        print("  🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'mujoco_interface' in locals():
            mujoco_interface.close()

if __name__ == "__main__":
    success = test_brain_physics()
    sys.exit(0 if success else 1)
