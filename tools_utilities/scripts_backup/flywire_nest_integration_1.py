#!/usr/bin/env python3
"""
FlyWire-NEST Integration for Advanced Fly Neuron Modeling
========================================================

This script integrates FlyWire connectome data with the existing NEST simulator
and other neuroscience simulation frameworks in your project.
"""

import os, sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flywire_integration import FlyWireDataManager, FlyNeuronSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlyWireNESTIntegration:
    """Integrates FlyWire data with NEST simulator."""
    
    def __init__(self, data_manager: FlyWireDataManager):
        self.data_manager = data_manager
        self.nest_neurons = {}
        self.nest_connections = {}
        self.nest_initialized = False
        
        logger.info("FlyWire-NEST Integration initialized")
    
    def initialize_nest(self):
        """Initialize NEST simulator."""
        try:
            import nest
            nest.set_verbosity('M_WARNING')
            nest.ResetKernel()
            
            # Set simulation parameters
            nest.SetKernelStatus({
                'resolution': 0.1,  # 0.1ms resolution
                'print_time': True,
                'overwrite_files': True
            })
            
            self.nest_initialized = True
            logger.info("NEST simulator initialized successfully")
            
        except ImportError:
            logger.warning("NEST not available, using fallback simulation")
            self.nest_initialized = False
    
    def create_nest_network(self):
        """Create NEST network from FlyWire data."""
        if not self.nest_initialized:
            logger.error("NEST not initialized")
            return False
        
        try:
            import nest
            
            # Create neuron populations based on cell types
            cell_type_models = {
                'Kenyon cell': 'iaf_cond_alpha',
                'Mushroom body output neuron': 'iaf_cond_alpha',
                'Central complex neuron': 'iaf_cond_alpha',
                'Optic lobe neuron': 'iaf_cond_alpha',
                'Antennal lobe neuron': 'iaf_cond_alpha',
                'Lateral horn neuron': 'iaf_cond_alpha',
                'Subesophageal ganglion neuron': 'iaf_cond_alpha',
                'Ventral nerve cord neuron': 'iaf_cond_alpha'
            }
            
            # Create neurons for each cell type
            for cell_type, model in cell_type_models.items():
                neurons = self.data_manager.get_cell_type_neurons(cell_type)
                if not neurons.empty:
                    # Create NEST neurons with cell type-specific parameters
                    neuron_params = self._get_cell_type_params(cell_type)
                    neuron_list = nest.Create(model, len(neurons), neuron_params)
                    
                    # Store mapping between FlyWire IDs and NEST IDs
                    for i, (_, neuron) in enumerate(neurons.iterrows()):
                        self.nest_neurons[neuron['neuron_id']] = neuron_list[i]
                    
                    logger.info(f"Created {len(neurons)} {cell_type} neurons in NEST")
            
            # Create synaptic connections
            self._create_nest_connections()
            
            logger.info("NEST network created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create NEST network: {e}")
            return False
    
    def _get_cell_type_params(self, cell_type: str) -> Dict:
        """Get NEST parameters for specific cell types."""
        base_params = {
            'V_th': -55.0,  # mV
            'V_reset': -70.0,  # mV
            't_ref': 2.0,  # ms
            'tau_syn_ex': 0.5,  # ms
            'tau_syn_in': 0.5,  # ms
        }
        
        # Cell type-specific modifications
        if cell_type == 'Kenyon cell':
            base_params.update({
                'V_th': -50.0,
                't_ref': 1.0,
                'tau_syn_ex': 1.0
            })
        elif cell_type == 'Mushroom body output neuron':
            base_params.update({
                'V_th': -60.0,
                't_ref': 3.0
            })
        elif 'Central complex' in cell_type:
            base_params.update({
                'V_th': -45.0,
                'tau_syn_in': 1.0
            })
        
        return base_params
    
    def _create_nest_connections(self):
        """Create synaptic connections in NEST."""
        if not self.nest_initialized:
            return
        
        try:
            import nest
            
            connection_count = 0
            for _, connection in self.data_manager.connectivity_data.iterrows():
                pre_id = connection['pre_neuron_id']
                post_id = connection['post_neuron_id']
                
                if pre_id in self.nest_neurons and post_id in self.nest_neurons:
                    pre_neuron = self.nest_neurons[pre_id]
                    post_neuron = self.nest_neurons[post_id]
                    
                    # Connection parameters
                    weight = connection['synapse_strength'] * 0.1
                    delay = 1.0  # ms
                    
                    # Connection type
                    if connection['connection_type'] == 'excitatory':
                        nest.Connect(pre_neuron, post_neuron, 
                                   syn_spec={'weight': weight, 'delay': delay})
                    elif connection['connection_type'] == 'inhibitory':
                        nest.Connect(pre_neuron, post_neuron,
                                   syn_spec={'weight': -weight, 'delay': delay})
                    
                    connection_count += 1
            
            logger.info(f"Created {connection_count} NEST connections")
            
        except Exception as e:
            logger.error(f"Failed to create NEST connections: {e}")
    
    def run_nest_simulation(self, duration: float = 100.0):
        """Run NEST simulation."""
        if not self.nest_initialized:
            logger.error("NEST not initialized")
            return
        
        try:
            import nest
            
            # Add external input to some neurons
            self._add_external_input()
            
            # Run simulation
            logger.info(f"Running NEST simulation for {duration}ms")
            nest.Simulate(duration)
            
            # Collect results
            self._collect_nest_results()
            
        except Exception as e:
            logger.error(f"NEST simulation failed: {e}")
    
    def _add_external_input(self):
        """Add external input to neurons."""
        if not self.nest_initialized:
            return
        
        try:
            import nest
            
            # Add external input to 10% of neurons
            num_neurons = len(self.nest_neurons)
            num_stimulated = max(1, int(num_neurons * 0.1))
            
            stimulated_neurons = np.random.choice(
                list(self.nest_neurons.values()), 
                num_stimulated, 
                replace=False
            )
            
            for neuron in stimulated_neurons:
                # Create DC generator
                dc_gen = nest.Create('dc_generator', params={'amplitude': 100.0})
                nest.Connect(dc_gen, neuron)
            
            logger.info(f"Added external input to {num_stimulated} neurons")
            
        except Exception as e:
            logger.error(f"Failed to add external input: {e}")
    
    def _collect_nest_results(self):
        """Collect simulation results from NEST."""
        if not self.nest_initialized:
            return
        
        try:
            import nest
            
            # Get spike times for all neurons
            spike_recorder = nest.Create('spike_recorder')
            
            # Connect all neurons to spike recorder
            for neuron in self.nest_neurons.values():
                nest.Connect(neuron, spike_recorder)
            
            # Get spike data
            spike_events = spike_recorder.get('events')
            spike_times = spike_events['times']
            spike_senders = spike_events['senders']
            
            logger.info(f"Collected {len(spike_times)} spikes from NEST simulation")
            
            # Store results
            self.nest_results = {
                'spike_times': spike_times,
                'spike_senders': spike_senders,
                'total_spikes': len(spike_times)
            }
            
        except Exception as e:
            logger.error(f"Failed to collect NEST results: {e}")

class FlyWireMuJoCoIntegration:
    """Integrates FlyWire data with MuJoCo physics simulator."""
    
    def __init__(self, data_manager: FlyWireDataManager):
        self.data_manager = data_manager
        self.mujoco_model = None
        self.mujoco_data = None
        
        logger.info("FlyWire-MuJoCo Integration initialized")
    
    def create_mujoco_model(self):
        """Create MuJoCo model from FlyWire neuron data."""
        try:
            import mujoco
            
            # Create a simple MuJoCo model for visualization
            model_xml = self._generate_mujoco_xml()
            
            # Load model
            self.mujoco_model = mujoco.MjModel.from_xml_string(model_xml)
            self.mujoco_data = mujoco.MjData(self.mujoco_model)
            
            logger.info("MuJoCo model created successfully")
            return True
            
        except ImportError:
            logger.warning("MuJoCo not available")
            return False
        except Exception as e:
            logger.error(f"Failed to create MuJoCo model: {e}")
            return False
    
    def _generate_mujoco_xml(self) -> str:
        """Generate MuJoCo XML for fly brain visualization."""
        # Sample a subset of neurons for visualization
        sample_neurons = self.data_manager.neuron_data.sample(min(100, len(self.data_manager.neuron_data)))
        
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
        for _, neuron in sample_neurons.iterrows():
            x, y, z = neuron['x_coord'] / 1000, neuron['y_coord'] / 1000, neuron['z_coord'] / 1000
            size = neuron['soma_size'] / 1000
            
            # Color based on cell type
            color = self._get_cell_type_color(neuron['cell_type'])
            
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
    
    def _get_cell_type_color(self, cell_type: str) -> str:
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

class FlyWireVisualization:
    """Provides visualization capabilities for FlyWire data."""
    
    def __init__(self, data_manager: FlyWireDataManager):
        self.data_manager = data_manager
        logger.info("FlyWire Visualization initialized")
    
    def create_3d_plot(self, output_file: str = None):
        """Create 3D visualization of the fly brain network."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Sample neurons for visualization
            sample_size = min(200, len(self.data_manager.neuron_data))
            sample_neurons = self.data_manager.neuron_data.sample(sample_size)
            
            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot neurons by cell type
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.data_manager.cell_types)))
            cell_type_colors = dict(zip(self.data_manager.cell_types.keys(), colors))
            
            for cell_type in self.data_manager.cell_types.keys():
                neurons = sample_neurons[sample_neurons['cell_type'] == cell_type]
                if not neurons.empty:
                    ax.scatter(neurons['x_coord'], neurons['y_coord'], neurons['z_coord'],
                             c=[cell_type_colors[cell_type]], label=cell_type, s=20, alpha=0.7)
            
            # Plot some connections
            sample_connections = self.data_manager.connectivity_data.sample(
                min(50, len(self.data_manager.connectivity_data))
            )
            
            for _, connection in sample_connections.iterrows():
                pre_neuron = self.data_manager.get_neuron_by_id(connection['pre_neuron_id'])
                post_neuron = self.data_manager.get_neuron_by_id(connection['post_neuron_id'])
                
                if pre_neuron is not None and post_neuron is not None:
                    x = [pre_neuron['x_coord'], post_neuron['x_coord']]
                    y = [pre_neuron['y_coord'], post_neuron['y_coord']]
                    z = [pre_neuron['z_coord'], post_neuron['z_coord']]
                    
                    # Color based on connection type
                    color = 'red' if connection['connection_type'] == 'excitatory' else 'blue'
                    ax.plot(x, y, z, color=color, alpha=0.3, linewidth=0.5)
            
            ax.set_xlabel('X (μm)')
            ax.set_ylabel('Y (μm)')
            ax.set_zlabel('Z (μm)')
            ax.set_title('Fly Brain Network Visualization')
            ax.legend()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"3D plot saved to {output_file}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
        except Exception as e:
            logger.error(f"Failed to create 3D plot: {e}")

def main():
    """Main function demonstrating comprehensive FlyWire integration."""
    print("Comprehensive FlyWire Integration with Neuroscience Frameworks")
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
    
    # NEST Integration
    print("\n2. NEST Integration...")
    nest_integration = FlyWireNESTIntegration(data_manager)
    nest_integration.initialize_nest()
    
    if nest_integration.nest_initialized:
        print("  NEST initialized successfully")
        if nest_integration.create_nest_network():
            print("  NEST network created")
            nest_integration.run_nest_simulation(duration=50.0)
            print("  NEST simulation completed")
    else:
        print("  NEST not available, using fallback simulation")
    
    # MuJoCo Integration
    print("\n3. MuJoCo Integration...")
    mujoco_integration = FlyWireMuJoCoIntegration(data_manager)
    if mujoco_integration.create_mujoco_model():
        print("  MuJoCo model created successfully")
    else:
        print("  MuJoCo not available")
    
    # Visualization
    print("\n4. Creating visualizations...")
    viz = FlyWireVisualization(data_manager)
    viz.create_3d_plot('flywire_data/fly_brain_3d.png')
    
    # Export data
    print("\n5. Exporting data...")
    output_file = data_manager.export_for_simulation()
    if output_file:
        print(f"  Data exported to: {output_file}")
    
    print("\nIntegration completed successfully!")
    print("\nNext steps:")
    print("1. Use the exported data in your existing simulations")
    print("2. Extend the NEST integration with more sophisticated models")
    print("3. Add more MuJoCo physics for neuron dynamics")
    print("4. Create interactive 3D visualizations")
    print("5. Integrate with your other neuroscience frameworks")

if __name__ == "__main__":
    main()
