#!/usr/bin/env python3
"""
MuJoCo Brain Development Physics Demo

Demonstrates the integration of MuJoCo physics engine with brain development simulation.
This demo shows:
- Brain region physics simulation
- Tissue mechanics
- Morphogen diffusion
- Integrated brain development modeling
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from physics_simulation import (
    MuJoCoInterface, 
    BrainPhysicsSimulator, 
    TissueMechanics, 
    MorphogenPhysics
)

def main():
    """Main demo function"""
    print("🚀 MuJoCo Brain Development Physics Demo")
    print("=" * 50)
    
    try:
        # Initialize MuJoCo interface
        print("\n1. Initializing MuJoCo Physics Interface...")
        mujoco_interface = MuJoCoInterface()
        
        # Setup brain development model
        print("\n2. Setting up Brain Development Model...")
        brain_regions = [
            "cerebral_cortex",
            "hippocampus", 
            "amygdala",
            "thalamus",
            "cerebellum"
        ]
        
        cell_types = [
            "excitatory_neurons",
            "inhibitory_neurons",
            "glial_cells",
            "endothelial_cells"
        ]
        
        # Create and load the brain development model
        success = mujoco_interface.create_brain_development_model(brain_regions, cell_types)
        if not success:
            print("❌ Failed to create brain development model")
            return
        
        print(f"✅ Created brain development model with {len(brain_regions)} regions and {len(cell_types)} cell types")
        
        # Initialize brain physics simulator
        print("\n3. Initializing Brain Physics Simulator...")
        brain_simulator = BrainPhysicsSimulator(mujoco_interface)
        
        setup_success = brain_simulator.setup_brain_development_model(brain_regions, cell_types)
        if not setup_success:
            print("❌ Failed to setup brain physics simulator")
            return
        
        print("✅ Brain physics simulator initialized")
        
        # Initialize tissue mechanics
        print("\n4. Initializing Tissue Mechanics...")
        tissue_mechanics = TissueMechanics()
        
        # Test tissue properties
        print("   Testing tissue mechanics...")
        cortex_deformation = tissue_mechanics.calculate_elastic_deformation(
            'cortex', 
            np.array([100.0, 0.0, 0.0]), 
            0.001
        )
        print(f"   Cortex deformation: {cortex_deformation['deformation']:.6f} m")
        
        # Initialize morphogen physics
        print("\n5. Initializing Morphogen Physics...")
        morphogen_physics = MorphogenPhysics(grid_size=50, domain_size=0.01)
        
        # Add morphogen sources
        print("   Setting up morphogen sources...")
        morphogen_physics.add_morphogen_source('shh', 25, 25)  # Center
        morphogen_physics.add_morphogen_source('bmp', 10, 10)  # Top-left
        morphogen_physics.add_morphogen_source('wnt', 40, 40)  # Bottom-right
        
        # Add morphogen sinks
        morphogen_physics.add_morphogen_sink('shh', 0, 0)
        morphogen_physics.add_morphogen_sink('bmp', 49, 49)
        morphogen_physics.add_morphogen_sink('wnt', 0, 49)
        
        print("✅ Morphogen sources and sinks configured")
        
        # Run integrated simulation
        print("\n6. Running Integrated Brain Development Simulation...")
        
        # Step 1: Brain growth simulation
        print("   Simulating brain growth...")
        growth_results = brain_simulator.simulate_brain_growth(duration=5.0)
        print(f"   Growth simulation completed: {len(growth_results['time_points'])} time points")
        
        # Step 2: Morphogen diffusion
        print("   Simulating morphogen diffusion...")
        diffusion_results = morphogen_physics.simulate_developmental_patterning(duration=2.0)
        print(f"   Diffusion simulation completed: {len(diffusion_results['time_points'])} time points")
        
        # Step 3: Tissue mechanics analysis
        print("   Analyzing tissue mechanics...")
        tissue_growth = tissue_mechanics.simulate_tissue_growth(
            'cortex', 
            growth_rate=0.01, 
            time_duration=5.0, 
            initial_volume=0.001
        )
        print(f"   Tissue growth simulation completed: {len(tissue_growth['time_points'])} time points")
        
        # Get simulation statistics
        print("\n7. Simulation Results:")
        print("-" * 30)
        
        # Brain development metrics
        brain_metrics = brain_simulator.get_development_metrics()
        print(f"   Development Stage: {brain_metrics['development_stage']:.3f}")
        print(f"   Brain Regions: {brain_metrics['region_count']}")
        print(f"   Total Cell Population: {brain_metrics['cell_population']}")
        print(f"   Average Growth Rate: {brain_metrics['average_growth_rate']:.6f}")
        
        # MuJoCo simulation stats
        mujoco_stats = mujoco_interface.get_simulation_stats()
        print(f"   Simulation Time: {mujoco_stats['simulation_time']:.3f} s")
        print(f"   Number of Bodies: {mujoco_stats['num_bodies']}")
        print(f"   Total Energy: {mujoco_stats['total_energy']:.6f}")
        
        # Morphogen summary
        morphogen_summary = morphogen_physics.get_morphogen_summary()
        print(f"   Morphogen Types: {len(morphogen_summary)}")
        for morphogen_type, summary in morphogen_summary.items():
            print(f"     {morphogen_type.upper()}: {summary['total_concentration']:.6f} total")
        
        # Tissue mechanics feedback
        cortex_feedback = tissue_mechanics.get_mechanical_feedback('cortex')
        if cortex_feedback:
            print(f"   Cortex Mechanical Feedback:")
            print(f"     Cumulative Deformation: {cortex_feedback['cumulative_deformation']:.6f} m")
            print(f"     Stress Cycles: {cortex_feedback['stress_cycles']}")
        
        # Visualization
        print("\n8. Generating Visualizations...")
        create_visualizations(growth_results, diffusion_results, tissue_growth)
        
        # Save simulation state
        print("\n9. Saving Simulation State...")
        save_simulation_state(brain_simulator, morphogen_physics, tissue_mechanics)
        
        print("\n✅ MuJoCo Brain Development Physics Demo completed successfully!")
        print("\n📊 Check the generated plots and saved data for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'mujoco_interface' in locals():
            mujoco_interface.close()

def create_visualizations(growth_results, diffusion_results, tissue_growth):
    """Create visualization plots for simulation results"""
    
    # Create output directory
    output_dir = Path("mujoco_simulation_output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Brain Growth Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for region in growth_results['region_sizes'][0].keys():
        sizes = [step[region] for step in growth_results['region_sizes']]
        plt.plot(growth_results['time_points'], sizes, label=region, marker='o')
    plt.title('Brain Region Growth Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Region Size')
    plt.legend()
    plt.grid(True)
    
    # 2. Growth Rates
    plt.subplot(2, 2, 2)
    for region in growth_results['growth_rates'][0].keys():
        rates = [step[region] for step in growth_results['growth_rates']]
        plt.plot(growth_results['time_points'], rates, label=region, marker='s')
    plt.title('Growth Rates Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Growth Rate')
    plt.legend()
    plt.grid(True)
    
    # 3. Tissue Growth
    plt.subplot(2, 2, 3)
    plt.plot(tissue_growth['time_points'], tissue_growth['volumes'], 'b-', label='Volume')
    plt.title('Cortex Tissue Growth')
    plt.xlabel('Time (s)')
    plt.ylabel('Volume (m³)')
    plt.grid(True)
    
    # 4. Tissue Stress
    plt.subplot(2, 2, 4)
    plt.plot(tissue_growth['time_points'], tissue_growth['stresses'], 'r-', label='Stress')
    plt.title('Cortex Tissue Stress')
    plt.xlabel('Time (s)')
    plt.ylabel('Stress (Pa)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "brain_development_physics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Visualizations saved to {output_dir}/")

def save_simulation_state(brain_simulator, morphogen_physics, tissue_mechanics):
    """Save simulation state to files"""
    
    output_dir = Path("mujoco_simulation_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save brain development state
    brain_simulator.save_development_state(output_dir / "brain_development_state.json")
    
    # Save morphogen state
    morphogen_summary = morphogen_physics.get_morphogen_summary()
    import json
    with open(output_dir / "morphogen_state.json", 'w') as f:
        json.dump(morphogen_summary, f, indent=2)
    
    # Save tissue mechanics state
    tissue_feedback = {}
    for tissue_type in ['cortex', 'white_matter', 'ventricles']:
        feedback = tissue_mechanics.get_mechanical_feedback(tissue_type)
        if feedback:
            tissue_feedback[tissue_type] = feedback
    
    with open(output_dir / "tissue_mechanics_state.json", 'w') as f:
        json.dump(tissue_feedback, f, indent=2)
    
    print(f"   💾 Simulation state saved to {output_dir}/")

if __name__ == "__main__":
    main()
