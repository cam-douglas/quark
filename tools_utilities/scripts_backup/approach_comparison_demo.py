#!/usr/bin/env python3
"""
Brain Simulation Approach Comparison Demo

This demo shows the practical differences between:
1. MuJoCo: Physical brain development
2. NEST: Neural network development  
3. Hybrid: Combined approach

Run this to see which approach best fits your research needs!
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demonstrate_mujoco_approach():
    """Demonstrate MuJoCo physical simulation capabilities"""
    print("\n🔬 MuJoCo Approach Demonstration")
    print("=" * 50)
    print("Research Focus: Physical brain development")
    print("Best for: Tissue mechanics, spatial organization, biomechanics")
    print("Scale: Macroscopic (brain regions, tissue layers)")
    print("Time: Developmental (days to months)")
    print("-" * 50)
    
    try:
        from physics_simulation.mujoco_interface import MuJoCoInterface
        from physics_simulation.dual_mode_simulator import DualModeBrainSimulator
        
        # Initialize MuJoCo interface
        mujoco_interface = MuJoCoInterface()
        
        # Create MuJoCo simulator
        simulator = DualModeBrainSimulator(
            simulation_mode="mujoco",
            mujoco_interface=mujoco_interface
        )
        
        # Setup physical brain model
        brain_regions = ["cortex", "hippocampus", "thalamus", "cerebellum"]
        cell_types = ["neurons", "glia", "endothelial"]
        
        print("🏗️  Setting up physical brain model...")
        success = simulator.setup_brain_development_model(brain_regions, cell_types)
        
        if not success:
            print("❌ Failed to setup MuJoCo model")
            return None
        
        # Run physical simulation
        print("⏱️  Running physical growth simulation...")
        duration = 3.0  # seconds
        results = simulator.simulate_brain_growth(duration)
        
        # Show physical results
        print(f"📊 Physical simulation results:")
        print(f"   - Time points: {len(results['time_points'])}")
        print(f"   - Physical data: {len(results['physical_data'])}")
        
        if 'physical_data' in results and 'region_sizes' in results['physical_data']:
            region_sizes = results['physical_data']['region_sizes']
            print(f"   - Region size tracking: {len(region_sizes)} steps")
            print(f"   - Growth rate tracking: {len(results['physical_data'].get('growth_rates', []))} steps")
        
        # Get development metrics
        metrics = simulator.get_development_metrics()
        print(f"📈 Physical development metrics:")
        print(f"   - Development stage: {metrics['development_stage']:.3f}")
        print(f"   - Region count: {metrics['region_count']}")
        print(f"   - Average growth rate: {metrics['average_growth_rate']:.3f}")
        
        print("✅ MuJoCo demonstration completed!")
        print("💡 This approach is ideal for:")
        print("   - Studying tissue mechanics and growth")
        print("   - Understanding spatial brain organization")
        print("   - Modeling biomechanical constraints")
        print("   - Investigating physical developmental disorders")
        
        return results
        
    except Exception as e:
        print(f"❌ MuJoCo demonstration failed: {e}")
        return None
    finally:
        if 'mujoco_interface' in locals():
            mujoco_interface.close()

def demonstrate_nest_approach():
    """Demonstrate NEST neural simulation capabilities"""
    print("\n🧠 NEST Approach Demonstration")
    print("=" * 50)
    print("Research Focus: Neural network development")
    print("Best for: Connectivity, learning, network dynamics")
    print("Scale: Microscopic (neurons, synapses, circuits)")
    print("Time: Neural (milliseconds to hours)")
    print("-" * 50)
    
    try:
        from physics_simulation.dual_mode_simulator import DualModeBrainSimulator
        
        # Create NEST simulator
        simulator = DualModeBrainSimulator(simulation_mode="nest")
        
        # Setup neural model
        brain_regions = ["cortex", "hippocampus", "thalamus"]
        cell_types = ["excitatory", "inhibitory"]
        region_sizes = {"cortex": 1500, "hippocampus": 1000, "thalamus": 800}
        
        print("🏗️  Setting up neural network model...")
        print(f"   - Regions: {brain_regions}")
        print(f"   - Cell types: {cell_types}")
        print(f"   - Total neurons: {sum(region_sizes.values())}")
        
        success = simulator.setup_brain_development_model(brain_regions, cell_types, region_sizes)
        
        if not success:
            print("❌ Failed to setup NEST model")
            return None
        
        # Run neural simulation
        print("⏱️  Running neural development simulation...")
        duration = 200.0  # milliseconds
        results = simulator.simulate_brain_growth(duration)
        
        # Show neural results
        print(f"📊 Neural simulation results:")
        print(f"   - Time points: {len(results['time_points'])}")
        print(f"   - Neural data: {len(results['neural_data'])}")
        
        if 'neural_data' in results and 'firing_rates' in results['neural_data']:
            firing_rates = results['neural_data']['firing_rates']
            print(f"   - Firing rates tracked: {len(firing_rates)} regions")
            print(f"   - Average firing rate: {np.mean(firing_rates):.2f} Hz")
        
        # Get development metrics
        metrics = simulator.get_development_metrics()
        print(f"📈 Neural development metrics:")
        print(f"   - Development stage: {metrics['development_stage']:.3f}")
        print(f"   - Neural populations: {metrics['neural_populations']}")
        print(f"   - Total neurons: {metrics['total_neurons']}")
        
        print("✅ NEST demonstration completed!")
        print("💡 This approach is ideal for:")
        print("   - Modeling neural circuit development")
        print("   - Studying learning and plasticity")
        print("   - Investigating network dynamics")
        print("   - Understanding neurological disorders")
        
        return results
        
    except Exception as e:
        print(f"❌ NEST demonstration failed: {e}")
        return None

def demonstrate_hybrid_approach():
    """Demonstrate hybrid simulation capabilities"""
    print("\n🚀 Hybrid Approach Demonstration")
    print("=" * 50)
    print("Research Focus: Multi-scale brain development")
    print("Best for: Physical-neural interactions, comprehensive modeling")
    print("Scale: Multi-scale (tissue + neural levels)")
    print("Time: Multiple scales (physical + neural)")
    print("-" * 50)
    
    try:
        from physics_simulation.mujoco_interface import MuJoCoInterface
        from physics_simulation.dual_mode_simulator import DualModeBrainSimulator
        
        # Initialize MuJoCo interface
        mujoco_interface = MuJoCoInterface()
        
        # Create hybrid simulator
        simulator = DualModeBrainSimulator(
            simulation_mode="hybrid",
            mujoco_interface=mujoco_interface
        )
        
        # Setup combined model
        brain_regions = ["cortex", "hippocampus", "thalamus"]
        cell_types = ["excitatory", "inhibitory"]
        region_sizes = {"cortex": 800, "hippocampus": 600, "thalamus": 400}
        
        print("🏗️  Setting up hybrid brain model...")
        print(f"   - Physical regions: {brain_regions}")
        print(f"   - Neural cell types: {cell_types}")
        print(f"   - Total neurons: {sum(region_sizes.values())}")
        
        success = simulator.setup_brain_development_model(brain_regions, cell_types, region_sizes)
        
        if not success:
            print("❌ Failed to setup hybrid model")
            return None
        
        # Run hybrid simulation
        print("⏱️  Running hybrid simulation...")
        duration = 2.0  # seconds
        results = simulator.simulate_brain_growth(duration)
        
        # Show hybrid results
        print(f"📊 Hybrid simulation results:")
        print(f"   - Time points: {len(results['time_points'])}")
        print(f"   - Physical data: {len(results['physical_data'])}")
        print(f"   - Neural data: {len(results['neural_data'])}")
        print(f"   - Coupling data: {len(results['coupling_data'])}")
        
        # Get development metrics
        metrics = simulator.get_development_metrics()
        print(f"📈 Hybrid development metrics:")
        print(f"   - Development stage: {metrics['development_stage']:.3f}")
        print(f"   - Physical regions: {metrics['region_count']}")
        print(f"   - Neural populations: {metrics['neural_populations']}")
        print(f"   - Total neurons: {metrics['total_neurons']}")
        print(f"   - Average growth rate: {metrics['average_growth_rate']:.3f}")
        
        print("✅ Hybrid demonstration completed!")
        print("💡 This approach is ideal for:")
        print("   - Studying multi-scale brain development")
        print("   - Investigating physical-neural interactions")
        print("   - Modeling complex developmental processes")
        print("   - Understanding developmental disorders holistically")
        
        return results
        
    except Exception as e:
        print(f"❌ Hybrid demonstration failed: {e}")
        return None
    finally:
        if 'mujoco_interface' in locals():
            mujoco_interface.close()

def create_approach_comparison(results_dict):
    """Create visualization comparing all three approaches"""
    print("\n📊 Creating approach comparison visualization...")
    
    output_dir = Path("outputs/approach_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Brain Simulation Approach Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: MuJoCo physical growth
    if 'mujoco' in results_dict and results_dict['mujoco']:
        ax = axes[0, 0]
        results = results_dict['mujoco']
        if 'physical_data' in results and 'region_sizes' in results['physical_data']:
            region_sizes_data = results['physical_data']['region_sizes']
            time_points = results['time_points']
            
            for i, region_name in enumerate(["cortex", "hippocampus", "thalamus", "cerebellum"]):
                if i < len(region_sizes_data):
                    sizes = [step.get(region_name, 0.1) for step in region_sizes_data]
                    if len(sizes) == len(time_points):
                        ax.plot(time_points, sizes, marker='o', label=region_name, markersize=3)
            
            ax.set_title('MuJoCo: Physical Growth')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Region Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Plot 2: NEST neural activity
    if 'nest' in results_dict and results_dict['nest']:
        ax = axes[0, 1]
        results = results_dict['nest']
        if 'neural_data' in results and 'firing_rates' in results['neural_data']:
            firing_rates = results['neural_data']['firing_rates']
            regions = ["cortex", "hippocampus", "thalamus"]
            
            bars = ax.bar(regions[:len(firing_rates)], firing_rates, alpha=0.7, color='green')
            ax.set_title('NEST: Neural Firing Rates')
            ax.set_xlabel('Brain Region')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, rate in zip(bars, firing_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{rate:.1f}', ha='center', va='bottom')
    
    # Plot 3: Hybrid coupling
    if 'hybrid' in results_dict and results_dict['hybrid']:
        ax = axes[0, 2]
        results = results_dict['hybrid']
        if 'coupling_data' in results and 'physical_constraints' in results['coupling_data']:
            constraints = results['coupling_data']['physical_constraints']
            regions = list(constraints.keys())
            values = list(constraints.values())
            
            bars = ax.bar(regions, values, alpha=0.7, color='purple')
            ax.set_title('Hybrid: Physical Constraints')
            ax.set_xlabel('Brain Region')
            ax.set_ylabel('Constraint Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    # Plot 4: Development stage comparison
    ax = axes[1, 0]
    approaches = []
    stages = []
    
    for approach, results in results_dict.items():
        if results:
            approaches.append(approach.upper())
            stages.append(results.get('development_stage', 0.0))
    
    if approaches:
        bars = ax.bar(approaches, stages, alpha=0.7, color=['blue', 'green', 'purple'])
        ax.set_title('Development Stage Comparison')
        ax.set_ylabel('Development Stage')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        for bar, stage in zip(bars, stages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{stage:.3f}', ha='center', va='bottom')
    
    # Plot 5: Data richness comparison
    ax = axes[1, 1]
    approaches = []
    data_points = []
    
    for approach, results in results_dict.items():
        if results:
            approaches.append(approach.upper())
            total_data = len(results.get('time_points', []))
            if 'physical_data' in results:
                total_data += len(results['physical_data'])
            if 'neural_data' in results:
                total_data += len(results['neural_data'])
            if 'coupling_data' in results:
                total_data += len(results['coupling_data'])
            data_points.append(total_data)
    
    if approaches:
        bars = ax.bar(approaches, data_points, alpha=0.7, color=['blue', 'green', 'purple'])
        ax.set_title('Data Richness Comparison')
        ax.set_ylabel('Total Data Points')
        ax.grid(True, alpha=0.3)
        
        for bar, points in zip(bars, data_points):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(points), ha='center', va='bottom')
    
    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Approach Summary:\n\n"
    
    for approach, results in results_dict.items():
        if results:
            summary_text += f"{approach.upper()}:\n"
            summary_text += f"  • Time points: {len(results.get('time_points', []))}\n"
            summary_text += f"  • Development stage: {results.get('development_stage', 0.0):.3f}\n"
            if 'physical_data' in results:
                summary_text += f"  • Physical data: {len(results['physical_data'])}\n"
            if 'neural_data' in results:
                summary_text += f"  • Neural data: {len(results['neural_data'])}\n"
            if 'coupling_data' in results:
                summary_text += f"  • Coupling data: {len(results['coupling_data'])}\n"
            summary_text += "\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save visualization
    plot_path = output_dir / "approach_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparison visualization saved to {plot_path}")
    
    # Export data
    data_path = output_dir / "approach_comparison_results.json"
    export_data = {
        'simulation_approaches': list(results_dict.keys()),
        'results_summary': {}
    }
    
    for approach, results in results_dict.items():
        if results:
            export_data['results_summary'][approach] = {
                'development_stage': results.get('development_stage', 0.0),
                'time_points': len(results.get('time_points', [])),
                'metrics': results.get('development_metrics', {})
            }
    
    with open(data_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"💾 Results exported to {data_path}")

def main():
    """Run the approach comparison demo"""
    print("🚀 Brain Simulation Approach Comparison Demo")
    print("=" * 70)
    print("This demo shows the practical differences between three approaches:")
    print("1. 🔬 MuJoCo: Physical brain development (tissue mechanics)")
    print("2. 🧠 NEST: Neural network development (connectivity, learning)")
    print("3. 🚀 Hybrid: Combined approach (multi-scale modeling)")
    print("=" * 70)
    
    results_dict = {}
    
    # Demonstrate each approach
    results_dict['mujoco'] = demonstrate_mujoco_approach()
    results_dict['nest'] = demonstrate_nest_approach()
    results_dict['hybrid'] = demonstrate_hybrid_approach()
    
    # Create comparison visualization
    if any(results_dict.values()):
        create_approach_comparison(results_dict)
        
        print("\n🎉 Approach comparison completed!")
        print("\n💡 Key Takeaways:")
        
        working_approaches = [k for k, v in results_dict.items() if v]
        
        if 'mujoco' in working_approaches:
            print("🔬 MuJoCo: Great for physical brain development studies")
        if 'nest' in working_approaches:
            print("🧠 NEST: Excellent for neural network development research")
        if 'hybrid' in working_approaches:
            print("🚀 Hybrid: Perfect for comprehensive multi-scale modeling")
        
        print("\n📚 Next Steps:")
        print("1. Review the comparison visualization")
        print("2. Check the detailed guide: simulation_approach_guide.md")
        print("3. Choose the approach that fits your research needs")
        print("4. Customize parameters for your specific study")
        
    else:
        print("\n❌ All approaches failed. Please check your setup.")

if __name__ == "__main__":
    main()
