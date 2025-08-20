#!/usr/bin/env python3
"""
Dual-Mode Brain Physics Simulation Demo

Demonstrates three simulation modes:
1. MuJoCo Mode: Physical brain development (tissue growth, mechanics)
2. NEST Mode: Neural network development (connectivity, activity)
3. Hybrid Mode: Combined physical and neural simulation
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from physics_simulation.mujoco_interface import MuJoCoInterface
from physics_simulation.brain_physics import BrainPhysicsSimulator

def run_mujoco_mode_demo():
    """Demo MuJoCo physical simulation mode"""
    print("\n🔬 MuJoCo Mode Demo - Physical Brain Development")
    print("=" * 50)
    
    try:
        # Initialize MuJoCo interface
        mujoco_interface = MuJoCoInterface()
        
        # Create brain physics simulator in MuJoCo mode
        brain_sim = BrainPhysicsSimulator(
            simulation_mode="mujoco",
            mujoco_interface=mujoco_interface
        )
        
        # Setup brain model
        brain_regions = ["cortex", "hippocampus", "thalamus", "cerebellum"]
        cell_types = ["neurons", "glia", "endothelial"]
        
        print(f"🏗️  Setting up physical brain model with {len(brain_regions)} regions")
        success = brain_sim.setup_brain_development_model(brain_regions, cell_types)
        
        if not success:
            print("❌ Failed to setup MuJoCo model")
            return
        
        # Run physical simulation
        print("⏱️  Running physical growth simulation...")
        duration = 5.0
        results = brain_sim.simulate_brain_growth(duration)
        
        # Display results
        print(f"📊 Physical simulation completed:")
        print(f"   - Time points: {len(results['time_points'])}")
        print(f"   - Physical data: {len(results['physical_data'])}")
        
        # Get metrics
        metrics = brain_sim.get_development_metrics()
        print(f"📈 Physical development metrics:")
        print(f"   - Development stage: {metrics['development_stage']:.3f}")
        print(f"   - Region count: {metrics['region_count']}")
        print(f"   - Average growth rate: {metrics['average_growth_rate']:.3f}")
        
        print("✅ MuJoCo mode demo completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ MuJoCo mode demo failed: {e}")
        return None
    finally:
        if 'mujoco_interface' in locals():
            mujoco_interface.close()

def run_nest_mode_demo():
    """Demo NEST neural simulation mode"""
    print("\n🧠 NEST Mode Demo - Neural Network Development")
    print("=" * 50)
    
    try:
        # Create brain physics simulator in NEST mode
        brain_sim = BrainPhysicsSimulator(simulation_mode="nest")
        
        # Setup neural model
        brain_regions = ["cortex", "hippocampus", "thalamus"]
        cell_types = ["excitatory", "inhibitory", "glia"]
        region_sizes = {"cortex": 2000, "hippocampus": 1500, "thalamus": 1000}
        
        print(f"🏗️  Setting up neural network model:")
        print(f"   - Regions: {brain_regions}")
        print(f"   - Cell types: {cell_types}")
        print(f"   - Total neurons: {sum(region_sizes.values())}")
        
        success = brain_sim.setup_brain_development_model(brain_regions, cell_types, region_sizes)
        
        if not success:
            print("❌ Failed to setup NEST model")
            return
        
        # Run neural simulation
        print("⏱️  Running neural development simulation...")
        duration = 100.0  # milliseconds
        results = brain_sim.simulate_brain_growth(duration)
        
        # Display results
        print(f"📊 Neural simulation completed:")
        print(f"   - Time points: {len(results['time_points'])}")
        print(f"   - Neural data: {len(results['neural_data'])}")
        
        # Get metrics
        metrics = brain_sim.get_development_metrics()
        print(f"📈 Neural development metrics:")
        print(f"   - Development stage: {metrics['development_stage']:.3f}")
        print(f"   - Neural populations: {metrics['neural_populations']}")
        print(f"   - Total neurons: {metrics['total_neurons']}")
        
        print("✅ NEST mode demo completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ NEST mode demo failed: {e}")
        return None

def run_hybrid_mode_demo():
    """Demo hybrid physical + neural simulation mode"""
    print("\n🚀 Hybrid Mode Demo - Combined Physical & Neural Simulation")
    print("=" * 60)
    
    try:
        # Initialize MuJoCo interface
        mujoco_interface = MuJoCoInterface()
        
        # Create brain physics simulator in hybrid mode
        brain_sim = BrainPhysicsSimulator(
            simulation_mode="hybrid",
            mujoco_interface=mujoco_interface
        )
        
        # Setup combined model
        brain_regions = ["cortex", "hippocampus", "thalamus"]
        cell_types = ["excitatory", "inhibitory"]
        region_sizes = {"cortex": 1000, "hippocampus": 800, "thalamus": 600}
        
        print(f"🏗️  Setting up hybrid brain model:")
        print(f"   - Physical regions: {brain_regions}")
        print(f"   - Neural cell types: {cell_types}")
        print(f"   - Total neurons: {sum(region_sizes.values())}")
        
        success = brain_sim.setup_brain_development_model(brain_regions, cell_types, region_sizes)
        
        if not success:
            print("❌ Failed to setup hybrid model")
            return
        
        # Run hybrid simulation
        print("⏱️  Running hybrid simulation...")
        duration = 3.0
        results = brain_sim.simulate_brain_growth(duration)
        
        # Display results
        print(f"📊 Hybrid simulation completed:")
        print(f"   - Time points: {len(results['time_points'])}")
        print(f"   - Physical data: {len(results['physical_data'])}")
        print(f"   - Neural data: {len(results['neural_data'])}")
        print(f"   - Coupling data: {len(results['coupling_data'])}")
        
        # Get metrics
        metrics = brain_sim.get_development_metrics()
        print(f"📈 Hybrid development metrics:")
        print(f"   - Development stage: {metrics['development_stage']:.3f}")
        print(f"   - Physical regions: {metrics['region_count']}")
        print(f"   - Neural populations: {metrics['neural_populations']}")
        print(f"   - Total neurons: {metrics['total_neurons']}")
        print(f"   - Average growth rate: {metrics['average_growth_rate']:.3f}")
        
        print("✅ Hybrid mode demo completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Hybrid mode demo failed: {e}")
        return None
    finally:
        if 'mujoco_interface' in locals():
            mujoco_interface.close()

def create_comparison_visualization(results_dict):
    """Create visualization comparing all three modes"""
    print("\n📊 Creating comparison visualization...")
    
    output_dir = Path("outputs/dual_mode_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dual-Mode Brain Physics Simulation Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Physical growth (MuJoCo)
    if 'mujoco' in results_dict and results_dict['mujoco']:
        ax = axes[0, 0]
        results = results_dict['mujoco']
        if 'physical_data' in results and 'region_sizes' in results['physical_data']:
            region_sizes_data = results['physical_data']['region_sizes']
            time_points = results['time_points']
            
            # Plot each region's growth over time
            for i, region_name in enumerate(results.get('brain_regions', [])):
                if i < len(region_sizes_data):
                    sizes = [step.get(region_name, 0.1) for step in region_sizes_data]
                    if len(sizes) == len(time_points):
                        ax.plot(time_points, sizes, marker='o', label=region_name, markersize=3)
            
            ax.set_title('MuJoCo: Physical Growth')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Region Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Plot 2: Neural activity (NEST)
    if 'nest' in results_dict and results_dict['nest']:
        ax = axes[0, 1]
        results = results_dict['nest']
        if 'neural_data' in results and 'firing_rates' in results['neural_data']:
            firing_rates = results['neural_data']['firing_rates']
            ax.bar(range(len(firing_rates)), firing_rates, alpha=0.7)
            ax.set_title('NEST: Neural Firing Rates')
            ax.set_xlabel('Brain Region')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.grid(True, alpha=0.3)
    
    # Plot 3: Hybrid coupling
    if 'hybrid' in results_dict and results_dict['hybrid']:
        ax = axes[0, 2]
        results = results_dict['hybrid']
        if 'coupling_data' in results and 'physical_constraints' in results['coupling_data']:
            constraints = results['coupling_data']['physical_constraints']
            regions = list(constraints.keys())
            values = list(constraints.values())
            ax.bar(regions, values, alpha=0.7, color='purple')
            ax.set_title('Hybrid: Physical Constraints')
            ax.set_xlabel('Brain Region')
            ax.set_ylabel('Constraint Value')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    # Plot 4: Development stage comparison
    ax = axes[1, 0]
    modes = []
    stages = []
    for mode, results in results_dict.items():
        if results:
            modes.append(mode.upper())
            stages.append(results.get('development_stage', 0.0))
    
    if modes:
        bars = ax.bar(modes, stages, alpha=0.7, color=['blue', 'green', 'purple'])
        ax.set_title('Development Stage Comparison')
        ax.set_ylabel('Development Stage')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, stage in zip(bars, stages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{stage:.3f}', ha='center', va='bottom')
    
    # Plot 5: Region counts
    ax = axes[1, 1]
    region_counts = []
    for mode, results in results_dict.items():
        if results:
            metrics = results.get('development_metrics', {})
            region_counts.append(metrics.get('region_count', 0))
    
    if region_counts:
        bars = ax.bar(modes, region_counts, alpha=0.7, color=['blue', 'green', 'purple'])
        ax.set_title('Brain Region Counts')
        ax.set_ylabel('Number of Regions')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, region_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(count), ha='center', va='bottom')
    
    # Plot 6: Simulation summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Simulation Summary:\n\n"
    for mode, results in results_dict.items():
        if results:
            summary_text += f"{mode.upper()} Mode:\n"
            summary_text += f"  • Time points: {len(results.get('time_points', []))}\n"
            summary_text += f"  • Development stage: {results.get('development_stage', 0.0):.3f}\n"
            if 'physical_data' in results:
                summary_text += f"  • Physical data: {len(results['physical_data'])}\n"
            if 'neural_data' in results:
                summary_text += f"  • Neural data: {len(results['neural_data'])}\n"
            summary_text += "\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save visualization
    plot_path = output_dir / "dual_mode_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparison visualization saved to {plot_path}")
    
    # Export data
    data_path = output_dir / "dual_mode_results.json"
    export_data = {
        'simulation_modes': list(results_dict.keys()),
        'results_summary': {}
    }
    
    for mode, results in results_dict.items():
        if results:
            export_data['results_summary'][mode] = {
                'development_stage': results.get('development_stage', 0.0),
                'time_points': len(results.get('time_points', [])),
                'metrics': results.get('development_metrics', {})
            }
    
    with open(data_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"💾 Results exported to {data_path}")

def main():
    """Run the complete dual-mode demo"""
    print("🚀 Dual-Mode Brain Physics Simulation Demo")
    print("=" * 60)
    print("This demo showcases three simulation approaches:")
    print("1. 🔬 MuJoCo: Physical brain development simulation")
    print("2. 🧠 NEST: Neural network development simulation") 
    print("3. 🚀 Hybrid: Combined physical + neural simulation")
    print("=" * 60)
    
    results_dict = {}
    
    # Run MuJoCo mode demo
    results_dict['mujoco'] = run_mujoco_mode_demo()
    
    # Run NEST mode demo
    results_dict['nest'] = run_nest_mode_demo()
    
    # Run hybrid mode demo
    results_dict['hybrid'] = run_hybrid_mode_demo()
    
    # Create comparison visualization
    if any(results_dict.values()):
        create_comparison_visualization(results_dict)
        
        print("\n🎉 Demo completed successfully!")
        print("💡 You can now:")
        print("   - Compare the three simulation approaches")
        print("   - Analyze the visualization results")
        print("   - Export data for further analysis")
        print("   - Use ParaView for advanced 3D visualization")
    else:
        print("\n❌ All demo modes failed. Please check your setup.")

if __name__ == "__main__":
    main()
