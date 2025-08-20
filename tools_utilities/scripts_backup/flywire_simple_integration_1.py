#!/usr/bin/env python3
"""
Simple FlyWire Integration for Neuroscience Simulation
====================================================

This script demonstrates how to integrate FlyWire data with your existing
neuroscience simulation frameworks.
"""

import os, sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

# Import our FlyWire integration
from flywire_integration import FlyWireDataManager, FlyNeuronSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_with_nest_simulator(data_manager: FlyWireDataManager):
    """Integrate FlyWire data with NEST simulator."""
    print("Integrating with NEST simulator...")
    
    try:
        import nest
        
        # Initialize NEST
        nest.ResetKernel()
        nest.SetKernelStatus({'resolution': 0.1, 'print_time': True})
        
        # Create neurons based on FlyWire data
        neurons = data_manager.neuron_data
        nest_neurons = {}
        
        # Group neurons by cell type for efficient creation
        for cell_type in neurons['cell_type'].unique():
            cell_neurons = neurons[neurons['cell_type'] == cell_type]
            
            # Create NEST neurons with appropriate model
            if 'Kenyon' in cell_type:
                model = 'iaf_cond_alpha'
                params = {'V_th': -50.0, 't_ref': 1.0}
            elif 'Mushroom body' in cell_type:
                model = 'iaf_cond_alpha'
                params = {'V_th': -60.0, 't_ref': 3.0}
            else:
                model = 'iaf_cond_alpha'
                params = {'V_th': -55.0, 't_ref': 2.0}
            
            neuron_list = nest.Create(model, len(cell_neurons), params)
            
            # Store mapping
            for i, (_, neuron) in enumerate(cell_neurons.iterrows()):
                nest_neurons[neuron['neuron_id']] = neuron_list[i]
            
            print(f"  Created {len(cell_neurons)} {cell_type} neurons")
        
        # Create connections
        connections = data_manager.connectivity_data
        connection_count = 0
        
        for _, connection in connections.iterrows():
            pre_id = connection['pre_neuron_id']
            post_id = connection['post_neuron_id']
            
            if pre_id in nest_neurons and post_id in nest_neurons:
                weight = connection['synapse_strength'] * 0.1
                delay = 1.0
                
                if connection['connection_type'] == 'excitatory':
                    nest.Connect(nest_neurons[pre_id], nest_neurons[post_id],
                               syn_spec={'weight': weight, 'delay': delay})
                else:
                    nest.Connect(nest_neurons[pre_id], nest_neurons[post_id],
                               syn_spec={'weight': -weight, 'delay': delay})
                
                connection_count += 1
        
        print(f"  Created {connection_count} connections")
        
        # Run simulation
        print("  Running NEST simulation...")
        nest.Simulate(50.0)  # 50ms simulation
        
        print("  NEST integration completed successfully!")
        return True
        
    except ImportError:
        print("  NEST not available")
        return False
    except Exception as e:
        print(f"  NEST integration failed: {e}")
        return False

def integrate_with_mujoco_simulator(data_manager: FlyWireDataManager):
    """Integrate FlyWire data with MuJoCo simulator."""
    print("Integrating with MuJoCo simulator...")
    
    try:
        import mujoco
        
        # Create a simple MuJoCo model for visualization
        neurons = data_manager.neuron_data.sample(min(100, len(data_manager.neuron_data)))
        
        # Generate MuJoCo XML
        xml_content = generate_mujoco_xml(neurons)
        
        # Load model
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)
        
        print(f"  Created MuJoCo model with {len(neurons)} neurons")
        print("  MuJoCo integration completed successfully!")
        return True
        
    except ImportError:
        print("  MuJoCo not available")
        return False
    except Exception as e:
        print(f"  MuJoCo integration failed: {e}")
        return False

def generate_mujoco_xml(neurons: pd.DataFrame) -> str:
    """Generate MuJoCo XML for fly brain visualization."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<mujoco model="fly_brain">',
        '  <compiler coordinate="local" angle="radian"/>',
        '  <option timestep="0.01" iterations="50" solver="Newton"/>',
        '  <worldbody>',
        '    <light diffuse=".8 .8 .8" pos="0 0 3" dir="0 0 -1"/>',
        '    <camera name="top" pos="0 0 5" quat="1 0 0 0"/>',
        '    <geom type="plane" size="1000 1000 0.1" rgba=".9 .9 .9 1"/>'
    ]
    
    # Add neurons as spheres
    for _, neuron in neurons.iterrows():
        x, y, z = neuron['x_coord'] / 1000, neuron['y_coord'] / 1000, neuron['z_coord'] / 1000
        size = neuron['soma_size'] / 1000
        
        # Color based on cell type
        color = get_cell_type_color(neuron['cell_type'])
        
        xml_parts.append(
            f'    <body name="neuron_{neuron["neuron_id"]}" pos="{x} {y} {z}">'
            f'      <geom type="sphere" size="{size}" rgba="{color}"/>'
            f'    </body>'
        )
    
    xml_parts.extend([
        '  </worldbody>',
        '</mujoco>'
    ])
    
    return '\n'.join(xml_parts)

def get_cell_type_color(cell_type: str) -> str:
    """Get color for cell type visualization."""
    colors = {
        'Kenyon cell': '1 0 0 1',  # Red
        'Mushroom body output neuron': '0 1 0 1',  # Green
        'Central complex neuron': '0 0 1 1',  # Blue
        'Optic lobe neuron': '1 1 0 1',  # Yellow
        'Antennal lobe neuron': '1 0 1 1',  # Magenta
        'Lateral horn neuron': '0 1 1 1',  # Cyan
        'Subesophageal ganglion neuron': '0.5 0.5 0.5 1',  # Gray
        'Ventral nerve cord neuron': '1 0.5 0 1'  # Orange
    }
    return colors.get(cell_type, '0.5 0.5 0.5 1')

def create_visualization(data_manager: FlyWireDataManager):
    """Create basic visualization of the fly brain network."""
    print("Creating visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Sample neurons for visualization
        sample_size = min(200, len(data_manager.neuron_data))
        sample_neurons = data_manager.neuron_data.sample(sample_size)
        
        # Create 2D plot (top-down view)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot neurons by cell type
        for cell_type in sample_neurons['cell_type'].unique():
            neurons = sample_neurons[sample_neurons['cell_type'] == cell_type]
            ax.scatter(neurons['x_coord'], neurons['y_coord'], 
                      label=cell_type, s=20, alpha=0.7)
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Fly Brain Network - Top View')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        output_file = 'flywire_data/fly_brain_2d.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  2D visualization saved to {output_file}")
        
    except ImportError:
        print("  Matplotlib not available for visualization")
    except Exception as e:
        print(f"  Visualization failed: {e}")

def main():
    """Main function demonstrating FlyWire integration."""
    print("FlyWire Integration with Neuroscience Simulation Frameworks")
    print("=" * 60)
    
    # Initialize data manager
    data_manager = FlyWireDataManager()
    
    # Setup data
    print("\n1. Setting up FlyWire data...")
    if not data_manager.download_sample_data():
        print("Failed to create sample data")
        return
    
    if not data_manager.load_data():
        print("Failed to load data")
        return
    
    # Display data summary
    stats = data_manager.get_network_statistics()
    print(f"\nData Summary:")
    print(f"  Total neurons: {stats['total_neurons']}")
    print(f"  Total connections: {stats['total_connections']}")
    print(f"  Connection density: {stats['connection_density']:.6f}")
    
    # Integration with existing frameworks
    print("\n2. Framework Integration:")
    
    # NEST Integration
    nest_success = integrate_with_nest_simulator(data_manager)
    
    # MuJoCo Integration
    mujoco_success = integrate_with_mujoco_simulator(data_manager)
    
    # Visualization
    create_visualization(data_manager)
    
    # Export data
    print("\n3. Exporting data...")
    output_file = data_manager.export_for_simulation()
    if output_file:
        print(f"  Data exported to: {output_file}")
    
    print("\nIntegration completed!")
    print("\nNext steps:")
    print("1. Use the exported data in your existing simulations")
    print("2. Extend the integrations with more sophisticated models")
    print("3. Add more visualization capabilities")
    print("4. Integrate with your other neuroscience frameworks")

if __name__ == "__main__":
    main()
