#!/usr/bin/env python3
"""
VisIt Integration Demo for Brain Development Simulation

This script demonstrates the integration of VisIt scientific visualization
with the brain physics simulator for advanced brain development analysis.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add source path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from physics_simulation.brain_physics import BrainPhysicsSimulator
from physics_simulation.visit_interface import VisItInterface, visualize_brain_data

def create_realistic_brain_data():
    """Create realistic brain development data for demonstration"""
    print("🧠 Creating realistic brain development data...")
    
    # Define brain regions with realistic properties
    brain_regions = {
        "frontal_lobe": {
            "position": [0, 0, 0],
            "size": 1200,
            "development_rate": 0.8,
            "cell_types": ["excitatory_pyramidal", "inhibitory_interneuron", "glia"]
        },
        "temporal_lobe": {
            "position": [1.5, 0, 0],
            "size": 900,
            "development_rate": 0.7,
            "cell_types": ["excitatory_pyramidal", "inhibitory_interneuron", "glia"]
        },
        "parietal_lobe": {
            "position": [3, 0, 0],
            "size": 1000,
            "development_rate": 0.75,
            "cell_types": ["excitatory_pyramidal", "inhibitory_interneuron", "glia"]
        },
        "occipital_lobe": {
            "position": [4.5, 0, 0],
            "size": 800,
            "development_rate": 0.6,
            "cell_types": ["excitatory_pyramidal", "inhibitory_interneuron", "glia"]
        },
        "hippocampus": {
            "position": [1.5, 1, 0],
            "size": 600,
            "development_rate": 0.9,
            "cell_types": ["excitatory_pyramidal", "inhibitory_interneuron", "glia"]
        }
    }
    
    # Generate neuron data
    neurons = []
    neuron_id = 0
    
    for region_name, region_data in brain_regions.items():
        region_neurons = region_data["size"]
        cell_types = region_data["cell_types"]
        
        for i in range(region_neurons):
            # Determine cell type
            cell_type_idx = i % len(cell_types)
            cell_type = cell_types[cell_type_idx]
            
            # Generate realistic position within region
            base_pos = region_data["position"]
            x = base_pos[0] + np.random.normal(0, 0.3)
            y = base_pos[1] + np.random.normal(0, 0.3)
            z = base_pos[2] + np.random.normal(0, 0.2)
            
            # Generate realistic activity based on development stage
            development_stage = min(1.0, np.random.exponential(0.5))
            activity = np.random.beta(2, 5) * development_stage
            
            neurons.append({
                "id": neuron_id,
                "type": cell_type,
                "region": region_name,
                "position": [x, y, z],
                "activity": activity,
                "development_stage": development_stage
            })
            neuron_id += 1
    
    # Generate time series data
    time_points = np.linspace(0, 100, 50)  # 100 time units, 50 points
    network_activity = []
    
    for t in time_points:
        # Simulate developmental growth with some noise
        base_activity = 0.1 + 0.8 * (1 - np.exp(-t / 20))  # Sigmoid growth
        noise = np.random.normal(0, 0.05)
        activity = max(0, min(1, base_activity + noise))
        network_activity.append(activity)
    
    brain_data = {
        "regions": brain_regions,
        "neurons": neurons,
        "time_series": {
            "time": time_points.tolist(),
            "activity": network_activity
        },
        "metadata": {
            "total_neurons": len(neurons),
            "total_regions": len(brain_regions),
            "simulation_time": 100,
            "development_complete": False
        }
    }
    
    print(f"✅ Created brain data with {len(neurons)} neurons across {len(brain_regions)} regions")
    return brain_data

def demonstrate_visit_visualization(brain_data):
    """Demonstrate VisIt visualization capabilities"""
    print("\n🔬 Demonstrating VisIt Visualization...")
    
    try:
        # Test if VisIt is available
        from physics_simulation.visit_interface import VISIT_AVAILABLE
        if not VISIT_AVAILABLE:
            print("⚠️  VisIt not available, skipping visualization demo")
            return False
        
        # Create VisIt interface
        visit_interface = VisItInterface()
        print("✅ VisIt interface created")
        
        # Create different types of visualizations
        visualization_types = ["3D", "2D", "time_series"]
        
        for vis_type in visualization_types:
            print(f"📊 Creating {vis_type} visualization...")
            
            success = visit_interface.create_brain_visualization(brain_data, vis_type)
            if success:
                # Export visualization
                output_file = f"brain_visualization_{vis_type.lower()}.png"
                export_success = visit_interface.export_visualization(output_file)
                
                if export_success:
                    print(f"✅ {vis_type} visualization exported to {output_file}")
                else:
                    print(f"⚠️  Failed to export {vis_type} visualization")
            else:
                print(f"❌ Failed to create {vis_type} visualization")
        
        # Perform data analysis
        print("\n📈 Performing data analysis...")
        analysis_types = ["statistics", "spatial", "temporal"]
        
        for analysis_type in analysis_types:
            results = visit_interface.analyze_brain_data(brain_data, analysis_type)
            if results:
                print(f"✅ {analysis_type.capitalize()} analysis completed:")
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value}")
                    elif isinstance(value, dict):
                        print(f"   {key}: {len(value)} items")
                    else:
                        print(f"   {key}: {value}")
            else:
                print(f"⚠️  {analysis_type.capitalize()} analysis failed")
        
        # Cleanup
        visit_interface.close()
        print("✅ VisIt interface closed")
        
        return True
        
    except Exception as e:
        print(f"❌ VisIt visualization demo failed: {e}")
        return False

def demonstrate_brain_physics_integration():
    """Demonstrate integration with brain physics simulator"""
    print("\n🧠 Demonstrating Brain Physics + VisIt Integration...")
    
    try:
        # Create brain physics simulator
        brain_sim = BrainPhysicsSimulator(simulation_time=50.0, time_step=0.1)
        
        # Setup brain development model
        brain_regions = ["cortex", "hippocampus", "cerebellum"]
        cell_types = ["excitatory", "inhibitory", "glia"]
        region_sizes = {"cortex": 200, "hippocampus": 100, "cerebellum": 150}
        
        success = brain_sim.setup_brain_development_model(brain_regions, cell_types, region_sizes)
        if not success:
            print("❌ Failed to setup brain development model")
            return False
        
        print("✅ Brain development model setup successfully")
        
        # Test VisIt integration
        if brain_sim.visit_interface:
            print("✅ VisIt interface available")
            
            # Create visualization
            vis_success = brain_sim.visualize_brain_development("3D", "integrated_brain_dev.png")
            if vis_success:
                print("✅ Integrated brain development visualization created")
            else:
                print("⚠️  Integrated visualization failed")
            
            # Perform analysis
            analysis_results = brain_sim.analyze_brain_data_with_visit("statistics")
            if analysis_results:
                print("✅ Integrated data analysis completed")
                print(f"   Development stage: {analysis_results.get('development_stage', 'N/A')}")
                print(f"   Total regions: {analysis_results.get('total_regions', 'N/A')}")
                print(f"   Total neurons: {analysis_results.get('total_neurons', 'N/A')}")
            else:
                print("⚠️  Integrated data analysis failed")
        else:
            print("⚠️  VisIt interface not available in brain simulator")
        
        # Cleanup
        brain_sim.cleanup()
        print("✅ Brain simulator cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Brain physics integration demo failed: {e}")
        return False

def create_summary_plots(brain_data):
    """Create summary plots using matplotlib"""
    print("\n📊 Creating Summary Plots...")
    
    try:
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Brain Development Analysis Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Region sizes
        regions = list(brain_data["regions"].keys())
        sizes = [brain_data["regions"][r]["size"] for r in regions]
        ax1.bar(regions, sizes, color='skyblue', alpha=0.7)
        ax1.set_title('Brain Region Sizes')
        ax1.set_ylabel('Neuron Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Development rates
        dev_rates = [brain_data["regions"][r]["development_rate"] for r in regions]
        ax2.bar(regions, dev_rates, color='lightgreen', alpha=0.7)
        ax2.set_title('Development Rates by Region')
        ax2.set_ylabel('Development Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Time series activity
        time_data = brain_data["time_series"]
        ax3.plot(time_data["time"], time_data["activity"], 'b-', linewidth=2)
        ax3.set_title('Network Activity Over Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Activity Level')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Neuron distribution by type
        neuron_types = {}
        for neuron in brain_data["neurons"]:
            cell_type = neuron["type"]
            if cell_type not in neuron_types:
                neuron_types[cell_type] = 0
            neuron_types[cell_type] += 1
        
        if neuron_types:
            ax4.pie(neuron_types.values(), labels=neuron_types.keys(), autopct='%1.1f%%')
            ax4.set_title('Neuron Distribution by Type')
        
        plt.tight_layout()
        
        # Save plot
        output_file = "brain_development_summary.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Summary plots saved to {output_file}")
        
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ Summary plot creation failed: {e}")
        return False

def main():
    """Main demonstration function"""
    print("🚀 VisIt Integration Demo for Brain Development Simulation")
    print("=" * 70)
    
    # Create realistic brain data
    brain_data = create_realistic_brain_data()
    
    # Demonstrate VisIt visualization
    visit_success = demonstrate_visit_visualization(brain_data)
    
    # Demonstrate brain physics integration
    integration_success = demonstrate_brain_physics_integration()
    
    # Create summary plots
    plots_success = create_summary_plots(brain_data)
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 DEMONSTRATION SUMMARY")
    print("=" * 70)
    
    results = {
        "VisIt Visualization": visit_success,
        "Brain Physics Integration": integration_success,
        "Summary Plots": plots_success
    }
    
    for feature, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{feature}: {status}")
    
    all_successful = all(results.values())
    
    if all_successful:
        print("\n🎉 All demonstrations completed successfully!")
        print("\n💡 Next Steps:")
        print("1. Explore the generated visualization files")
        print("2. Modify brain_data to experiment with different configurations")
        print("3. Use brain_sim.visualize_brain_development() in your own simulations")
        print("4. Integrate VisIt analysis into your brain development workflows")
    else:
        print("\n⚠️  Some demonstrations failed. Check the output above for details.")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure VisIt is properly installed")
        print("2. Check that all dependencies are available")
        print("3. Verify the brain physics simulator is working")
    
    return all_successful

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
