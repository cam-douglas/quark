#!/usr/bin/env python3
"""
FlyWire Integration for Fly Neuron Modeling
==========================================

This script integrates FlyWire connectome data with the existing neuroscience
simulation framework for modeling Drosophila melanogaster neurons.
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlyWireDataManager:
    """Manages FlyWire connectome data access and processing."""
    
    def __init__(self, data_dir: str = "flywire_data"):
        self.data_dir = data_dir
        self.base_url = "https://codex.flywire.ai"
        self.api_url = "https://codex.flywire.ai/api"
        self.session = requests.Session()
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize data storage
        self.neuron_data = {}
        self.connectivity_data = {}
        self.cell_types = {}
        self.hemilineages = {}
        
        logger.info(f"FlyWire Data Manager initialized with data directory: {data_dir}")
    
    def download_sample_data(self) -> bool:
        """Download sample fly neuron data for demonstration."""
        try:
            logger.info("Creating synthetic FlyWire data based on published research...")
            
            # Create synthetic neuron data based on FlyWire research
            self._create_synthetic_neurons()
            self._create_synthetic_connectivity()
            self._create_cell_type_annotations()
            
            # Save data to files
            self._save_data()
            
            logger.info("Synthetic FlyWire data created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create synthetic data: {e}")
            return False

    def _create_synthetic_neurons(self):
        """Create synthetic neuron data based on FlyWire research."""
        # Based on FlyWire research: ~100,000 neurons in adult fly brain
        num_neurons = 1000  # Reduced for demonstration
        
        # Generate synthetic neuron IDs and positions
        neuron_ids = [f"720575940638757464_{i}" for i in range(num_neurons)]
        
        # Generate 3D coordinates (in microns) based on fly brain dimensions
        # Fly brain is roughly 500x400x200 microns
        x_coords = np.random.uniform(0, 500, num_neurons)
        y_coords = np.random.uniform(0, 400, num_neurons)
        z_coords = np.random.uniform(0, 200, num_neurons)
        
        # Generate neuron properties
        soma_sizes = np.random.uniform(3, 15, num_neurons)  # microns
        axon_lengths = np.random.uniform(100, 2000, num_neurons)  # microns
        dendrite_lengths = np.random.uniform(50, 1000, num_neurons)  # microns
        
        # Create neuron dataframe
        self.neuron_data = pd.DataFrame({
            'neuron_id': neuron_ids,
            'x_coord': x_coords,
            'y_coord': y_coords,
            'z_coord': z_coords,
            'soma_size': soma_sizes,
            'axon_length': axon_lengths,
            'dendrite_length': dendrite_lengths,
            'synapse_count': np.random.poisson(50, num_neurons),
            'branch_count': np.random.poisson(20, num_neurons)
        })
        
        logger.info(f"Created {num_neurons} synthetic neurons")
    
    def _create_synthetic_connectivity(self):
        """Create synthetic connectivity data."""
        num_neurons = len(self.neuron_data)
        
        # Generate synaptic connections
        # Average connectivity: ~10-20% of possible connections
        connection_probability = 0.15
        max_connections = int(num_neurons * connection_probability)
        
        connections = []
        for _ in range(max_connections):
            pre_neuron = np.random.randint(0, num_neurons)
            post_neuron = np.random.randint(0, num_neurons)
            
            if pre_neuron != post_neuron:
                # Generate synapse properties
                synapse_count = np.random.poisson(3)  # Multiple synapses per connection
                synapse_strength = np.random.uniform(0.1, 1.0)
                
                connections.append({
                    'pre_neuron_id': self.neuron_data.iloc[pre_neuron]['neuron_id'],
                    'post_neuron_id': self.neuron_data.iloc[post_neuron]['neuron_id'],
                    'synapse_count': synapse_count,
                    'synapse_strength': synapse_strength,
                    'connection_type': np.random.choice(['excitatory', 'inhibitory', 'modulatory'])
                })
        
        self.connectivity_data = pd.DataFrame(connections)
        logger.info(f"Created {len(connections)} synaptic connections")
    
    def _create_cell_type_annotations(self):
        """Create synthetic cell type and hemilineage annotations."""
        # Common fly brain cell types based on research
        cell_types = [
            'Kenyon cell', 'Mushroom body output neuron', 'Central complex neuron',
            'Optic lobe neuron', 'Antennal lobe neuron', 'Lateral horn neuron',
            'Subesophageal ganglion neuron', 'Ventral nerve cord neuron'
        ]
        
        hemilineages = [
            'ALad1', 'ALad2', 'ALl1', 'ALl2', 'ALv1', 'ALv2',
            'MBad1', 'MBad2', 'MBl1', 'MBl2', 'MBv1', 'MBv2',
            'CCad1', 'CCad2', 'CCl1', 'CCl2', 'CCv1', 'CCv2'
        ]
        
        # Assign cell types and hemilineages to neurons
        num_neurons = len(self.neuron_data)
        
        self.neuron_data['cell_type'] = np.random.choice(cell_types, num_neurons)
        self.neuron_data['hemilineage'] = np.random.choice(hemilineages, num_neurons)
        
        # Create cell type summary
        self.cell_types = self.neuron_data['cell_type'].value_counts().to_dict()
        self.hemilineages = self.neuron_data['hemilineage'].value_counts().to_dict()
        
        logger.info("Created cell type and hemilineage annotations")
    
    def _save_data(self):
        """Save data to files."""
        # Save neuron data
        neuron_file = os.path.join(self.data_dir, 'neurons.csv')
        self.neuron_data.to_csv(neuron_file, index=False)
        
        # Save connectivity data
        connectivity_file = os.path.join(self.data_dir, 'connectivity.csv')
        self.connectivity_data.to_csv(connectivity_file, index=False)
        
        # Save metadata
        metadata = {
            'dataset_name': 'FAFB (First Adult Fly Brain)',
            'source': 'FlyWire - Synthetic Data for Demonstration',
            'neuron_count': len(self.neuron_data),
            'connection_count': len(self.connectivity_data),
            'cell_types': self.cell_types,
            'hemilineages': self.hemilineages,
            'creation_date': pd.Timestamp.now().isoformat(),
            'notes': 'Synthetic data created for integration with neuroscience simulation framework'
        }
        
        metadata_file = os.path.join(self.data_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Data saved to {self.data_dir}")
    
    def load_data(self) -> bool:
        """Load previously saved data."""
        try:
            neuron_file = os.path.join(self.data_dir, 'neurons.csv')
            connectivity_file = os.path.join(self.data_dir, 'connectivity.csv')
            metadata_file = os.path.join(self.data_dir, 'metadata.json')
            
            if all(os.path.exists(f) for f in [neuron_file, connectivity_file, metadata_file]):
                self.neuron_data = pd.read_csv(neuron_file)
                self.connectivity_data = pd.read_csv(connectivity_file)
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.cell_types = metadata['cell_types']
                self.hemilineages = metadata['hemilineages']
                
                logger.info("Data loaded successfully")
                return True
            else:
                logger.warning("Data files not found, run download_sample_data() first")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def get_neuron_by_id(self, neuron_id: str) -> Optional[pd.Series]:
        """Get neuron data by ID."""
        if self.neuron_data.empty:
            return None
        
        neuron = self.neuron_data[self.neuron_data['neuron_id'] == neuron_id]
        return neuron.iloc[0] if not neuron.empty else None
    
    def get_connections(self, neuron_id: str, direction: str = 'both') -> pd.DataFrame:
        """
        Get connections for a specific neuron.
        
        Args:
            neuron_id: ID of the neuron
            direction: 'pre' (presynaptic), 'post' (postsynaptic), or 'both'
        """
        if self.connectivity_data.empty:
            return pd.DataFrame()
        
        if direction == 'pre':
            return self.connectivity_data[self.connectivity_data['pre_neuron_id'] == neuron_id]
        elif direction == 'post':
            return self.connectivity_data[self.connectivity_data['post_neuron_id'] == neuron_id]
        else:  # both
            pre_connections = self.connectivity_data[self.connectivity_data['pre_neuron_id'] == neuron_id]
            post_connections = self.connectivity_data[self.connectivity_data['post_neuron_id'] == neuron_id]
            return pd.concat([pre_connections, post_connections])
    
    def get_cell_type_neurons(self, cell_type: str) -> pd.DataFrame:
        """Get all neurons of a specific cell type."""
        if self.neuron_data.empty:
            return pd.DataFrame()
        
        return self.neuron_data[self.neuron_data['cell_type'] == cell_type]
    
    def get_hemilineage_neurons(self, hemilineage: str) -> pd.DataFrame:
        """Get all neurons of a specific hemilineage."""
        if self.neuron_data.empty:
            return pd.DataFrame()
        
        return self.neuron_data[self.neuron_data['hemilineage'] == hemilineage]
    
    def get_network_statistics(self) -> Dict:
        """Get network statistics."""
        if self.neuron_data.empty or self.connectivity_data.empty:
            return {}
        
        stats = {
            'total_neurons': len(self.neuron_data),
            'total_connections': len(self.connectivity_data),
            'connection_density': len(self.connectivity_data) / (len(self.neuron_data) ** 2),
            'avg_synapses_per_connection': self.connectivity_data['synapse_count'].mean(),
            'avg_synapse_strength': self.connectivity_data['synapse_strength'].mean(),
            'cell_type_distribution': self.cell_types,
            'hemilineage_distribution': self.hemilineages
        }
        
        return stats
    
    def export_for_simulation(self, output_file: str = None) -> str:
        """
        Export data in a format suitable for neuroscience simulation.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            str: Path to exported file
        """
        if output_file is None:
            output_file = os.path.join(self.data_dir, 'flywire_simulation_data.h5')
        
        try:
            with pd.HDFStore(output_file, 'w') as store:
                store['neurons'] = self.neuron_data
                store['connectivity'] = self.connectivity_data
                store['metadata'] = pd.DataFrame([{
                    'cell_types': json.dumps(self.cell_types),
                    'hemilineages': json.dumps(self.hemilineages)
                }])
            
            logger.info(f"Data exported to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return ""

class FlyNeuronSimulator:
    """Simulates fly neurons using FlyWire data."""
    
    def __init__(self, data_manager: FlyWireDataManager):
        self.data_manager = data_manager
        self.simulation_time = 0
        self.neuron_states = {}
        self.synaptic_weights = {}
        
        logger.info("Fly Neuron Simulator initialized")
    
    def initialize_simulation(self):
        """Initialize the simulation with neuron states."""
        if self.data_manager.neuron_data.empty:
            logger.error("No neuron data available")
            return
        
        # Initialize neuron states
        for _, neuron in self.data_manager.neuron_data.iterrows():
            neuron_id = neuron['neuron_id']
            self.neuron_states[neuron_id] = {
                'membrane_potential': -70.0,  # mV, resting potential
                'spike_count': 0,
                'last_spike_time': 0,
                'refractory_period': 0,
                'external_input': 0.0
            }
        
        # Initialize synaptic weights
        for _, connection in self.data_manager.connectivity_data.iterrows():
            connection_key = (connection['pre_neuron_id'], connection['post_neuron_id'])
            self.synaptic_weights[connection_key] = connection['synapse_strength']
        
        logger.info(f"Initialized simulation with {len(self.neuron_states)} neurons")
    
    def step_simulation(self, dt: float = 0.1):
        """
        Advance simulation by one time step.
        
        Args:
            dt: Time step in milliseconds
        """
        self.simulation_time += dt
        
        # Update neuron states
        for neuron_id, state in self.neuron_states.items():
            # Simple leaky integrate-and-fire model
            if state['refractory_period'] > 0:
                state['refractory_period'] -= dt
                continue
            
            # Update membrane potential
            tau_m = 20.0  # membrane time constant (ms)
            v_rest = -70.0  # resting potential (mV)
            v_thresh = -55.0  # threshold potential (mV)
            
            # Leak current
            leak_current = (state['membrane_potential'] - v_rest) / tau_m
            
            # Synaptic input
            synaptic_input = self._calculate_synaptic_input(neuron_id)
            
            # Update membrane potential
            dv = (-leak_current + synaptic_input + state['external_input']) * dt
            state['membrane_potential'] += dv
            
            # Check for spike
            if state['membrane_potential'] >= v_thresh:
                state['membrane_potential'] = v_rest
                state['spike_count'] += 1
                state['last_spike_time'] = self.simulation_time
                state['refractory_period'] = 2.0  # 2ms refractory period
    
    def _calculate_synaptic_input(self, neuron_id: str) -> float:
        """Calculate synaptic input to a neuron."""
        total_input = 0.0
        
        # Find presynaptic connections
        presynaptic = self.data_manager.get_connections(neuron_id, 'post')
        
        for _, connection in presynaptic.iterrows():
            pre_neuron_id = connection['pre_neuron_id']
            if pre_neuron_id in self.neuron_states:
                # Check if presynaptic neuron recently spiked
                pre_state = self.neuron_states[pre_neuron_id]
                time_since_spike = self.simulation_time - pre_state['last_spike_time']
                
                if 0 <= time_since_spike <= 5.0:  # 5ms synaptic delay
                    connection_key = (pre_neuron_id, neuron_id)
                    if connection_key in self.synaptic_weights:
                        weight = self.synaptic_weights[connection_key]
                        synapse_count = connection['synapse_count']
                        
                        # Excitatory or inhibitory
                        if connection['connection_type'] == 'excitatory':
                            total_input += weight * synapse_count * 0.1
                        else:
                            total_input -= weight * synapse_count * 0.1
        
        return total_input
    
    def get_simulation_state(self) -> Dict:
        """Get current simulation state."""
        return {
            'simulation_time': self.simulation_time,
            'active_neurons': sum(1 for state in self.neuron_states.values() 
                                if state['refractory_period'] <= 0),
            'total_spikes': sum(state['spike_count'] for state in self.neuron_states.values()),
            'avg_membrane_potential': np.mean([state['membrane_potential'] 
                                             for state in self.neuron_states.values()])
        }
    
    def run_simulation(self, duration: float = 100.0, dt: float = 0.1):
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration in milliseconds
            dt: Time step in milliseconds
        """
        logger.info(f"Starting simulation for {duration}ms with dt={dt}ms")
        
        self.initialize_simulation()
        
        steps = int(duration / dt)
        for step in range(steps):
            self.step_simulation(dt)
            
            if step % 100 == 0:  # Log every 100 steps
                state = self.get_simulation_state()
                logger.info(f"Step {step}: Time={state['simulation_time']:.1f}ms, "
                          f"Active={state['active_neurons']}, Spikes={state['total_spikes']}")
        
        logger.info("Simulation completed")

def main():
    """Main function to demonstrate FlyWire integration."""
    print("FlyWire Integration for Fly Neuron Modeling")
    print("=" * 50)
    
    # Initialize data manager
    data_manager = FlyWireDataManager()
    
    # Download/create sample data
    print("\n1. Setting up FlyWire data...")
    if not data_manager.download_sample_data():
        print("Failed to create sample data")
        return
    
    # Load data
    print("\n2. Loading data...")
    if not data_manager.load_data():
        print("Failed to load data")
        return
    
    # Display data summary
    print("\n3. Data Summary:")
    stats = data_manager.get_network_statistics()
    for key, value in stats.items():
        if key not in ['cell_type_distribution', 'hemilineage_distribution']:
            print(f"  {key}: {value}")
    
    print("\n  Cell Types:")
    for cell_type, count in stats['cell_type_distribution'].items():
        print(f"    {cell_type}: {count}")
    
    print("\n  Hemilineages:")
    for hemilineage, count in stats['hemilineage_distribution'].items():
        print(f"    {hemilineage}: {count}")
    
    # Example queries
    print("\n4. Example Queries:")
    
    # Get a specific neuron
    sample_neuron = data_manager.neuron_data.iloc[0]
    print(f"  Sample neuron {sample_neuron['neuron_id']}:")
    print(f"    Type: {sample_neuron['cell_type']}")
    print(f"    Hemilineage: {sample_neuron['hemilineage']}")
    print(f"    Position: ({sample_neuron['x_coord']:.1f}, {sample_neuron['y_coord']:.1f}, {sample_neuron['z_coord']:.1f})")
    
    # Get connections
    connections = data_manager.get_connections(sample_neuron['neuron_id'])
    print(f"    Connections: {len(connections)}")
    
    # Run simulation
    print("\n5. Running neuron simulation...")
    simulator = FlyNeuronSimulator(data_manager)
    simulator.run_simulation(duration=50.0, dt=0.1)
    
    # Export data
    print("\n6. Exporting data for simulation...")
    output_file = data_manager.export_for_simulation()
    if output_file:
        print(f"  Data exported to: {output_file}")
    
    print("\nFlyWire integration completed successfully!")
    print("\nNext steps:")
    print("1. Use the exported HDF5 file in your neuroscience simulations")
    print("2. Integrate with your existing NEST, MuJoCo, or other simulation frameworks")
    print("3. Extend the simulation with more sophisticated neuron models")
    print("4. Add visualization capabilities for the fly brain network")

if __name__ == "__main__":
    main()
