"""
Brain Physics Simulator using NEST

Simulates brain development and neural network dynamics including:
- Neural population growth
- Synaptic connectivity development
- Network activity patterns
- Developmental plasticity
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import sys
import os

# Add conda base path to Python path to access NEST
conda_base = "/opt/homebrew/Caskroom/miniforge/base"
if conda_base not in sys.path:
    sys.path.insert(0, conda_base)

try:
    import nest
    NEST_AVAILABLE = True
except ImportError:
    NEST_AVAILABLE = False
    print("Warning: NEST not available. Install with: conda install -c conda-forge nest-simulator")

# Add VisIt integration
try:
    from .....................................................visit_interface import VisItInterface, VISIT_AVAILABLE
except ImportError:
    VISIT_AVAILABLE = False
    print("Warning: VisIt interface not available")

logger = logging.getLogger(__name__)


class BrainPhysicsSimulator:
    """Simulates brain development and neural network dynamics using NEST"""
    
    def __init__(self, simulation_time: float = 1000.0, time_step: float = 0.1):
        """
        Initialize brain physics simulator
        
        Args:
            simulation_time: Total simulation time in milliseconds
            time_step: Simulation time step in milliseconds
        """
        if not NEST_AVAILABLE:
            raise ImportError("NEST simulator not available")
        
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.brain_regions = {}
        self.cell_populations = {}
        self.synaptic_connections = {}
        self.development_stage = 0.0
        self.network_activity = []
        
        # Initialize NEST
        self._setup_nest()
        logger.info("NEST-based brain physics simulator initialized")
        
        # Initialize VisIt interface if available
        self.visit_interface = None
        if VISIT_AVAILABLE:
            try:
                self.visit_interface = VisItInterface()
                logger.info("VisIt interface initialized for visualization")
            except Exception as e:
                logger.warning(f"Failed to initialize VisIt interface: {e}")
                self.visit_interface = None
    
    def _setup_nest(self):
        """Setup NEST simulation environment"""
        try:
            # Reset NEST kernel
            nest.ResetKernel()
            
            # Set simulation parameters
            nest.SetKernelStatus({
                'resolution': self.time_step,
                'print_time': True,
                'overwrite_files': True
            })
            
            logger.info("NEST kernel configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup NEST: {e}")
            raise
    
    def setup_brain_development_model(self, 
                                    brain_regions: List[str],
                                    cell_types: List[str],
                                    region_sizes: Optional[Dict[str, int]] = None) -> bool:
        """
        Setup brain development simulation model
        
        Args:
            brain_regions: List of brain region names
            cell_types: List of cell type names
            region_sizes: Dictionary mapping regions to neuron counts
        """
        try:
            if region_sizes is None:
                region_sizes = {region: 1000 for region in brain_regions}
            
            # Create neural populations for each region
            for region in brain_regions:
                self.brain_regions[region] = {
                    'neurons': {},
                    'size': region_sizes[region]
                }
                
                # Create different neuron types for each region
                for cell_type in cell_types:
                    if cell_type == 'excitatory':
                        neuron_model = 'iaf_cond_alpha'
                        neuron_params = {
                            'V_th': -55.0,
                            'V_reset': -70.0,
                            't_ref': 2.0,
                            'tau_syn_ex': 0.5,
                            'tau_syn_in': 0.5
                        }
                    elif cell_type == 'inhibitory':
                        neuron_model = 'iaf_cond_alpha'
                        neuron_params = {
                            'V_th': -55.0,
                            'V_reset': -70.0,
                            't_ref': 2.0,
                            'tau_syn_ex': 0.5,
                            'tau_syn_in': 0.5
                        }
                    else:
                        # Default neuron model
                        neuron_model = 'iaf_cond_alpha'
                        neuron_params = {
                            'V_th': -55.0,
                            'V_reset': -70.0,
                            't_ref': 2.0
                        }
                    
                    # Create neuron population
                    neurons = nest.Create(neuron_model, region_sizes[region], neuron_params)
                    self.brain_regions[region]['neurons'][cell_type] = neurons
                    
                    # Store population info
                    self.cell_populations[f"{region}_{cell_type}"] = {
                        'count': region_sizes[region],
                        'neurons': neurons,
                        'type': cell_type,
                        'region': region
                    }
            
            logger.info(f"Brain development model setup complete: {len(brain_regions)} regions, {len(cell_types)} cell types")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup brain development model: {e}")
            return False
    
    def setup_developmental_connectivity(self, connectivity_pattern: str = 'random'):
        """
        Setup developmental connectivity patterns
        
        Args:
            connectivity_pattern: Type of connectivity ('random', 'lateral', 'hierarchical')
        """
        try:
            if connectivity_pattern == 'random':
                self._setup_random_connectivity()
            elif connectivity_pattern == 'lateral':
                self._setup_lateral_connectivity()
            elif connectivity_pattern == 'hierarchical':
                self._setup_hierarchical_connectivity()
            else:
                logger.warning(f"Unknown connectivity pattern: {connectivity_pattern}, using random")
                self._setup_random_connectivity()
                
            logger.info(f"Developmental connectivity setup complete: {connectivity_pattern}")
            
        except Exception as e:
            logger.error(f"Failed to setup connectivity: {e}")
    
    def _setup_random_connectivity(self):
        """Setup random synaptic connections"""
        for region_name, region_data in self.brain_regions.items():
            for cell_type, neurons in region_data['neurons'].items():
                # Connect neurons within the same population
                if len(neurons) > 1:
                    nest.Connect(neurons, neurons, 
                               {'rule': 'pairwise_bernoulli', 'p': 0.1})
                
                # Connect to other regions
                for other_region, other_data in self.brain_regions.items():
                    if other_region != region_name:
                        for other_cell_type, other_neurons in other_data['neurons'].items():
                            nest.Connect(neurons, other_neurons,
                                       {'rule': 'pairwise_bernoulli', 'p': 0.05})
    
    def _setup_lateral_connectivity(self):
        """Setup lateral inhibitory connectivity"""
        for region_name, region_data in self.brain_regions.items():
            if 'inhibitory' in region_data['neurons'] and 'excitatory' in region_data['neurons']:
                excitatory = region_data['neurons']['excitatory']
                inhibitory = region_data['neurons']['inhibitory']
                
                # Lateral inhibition: inhibitory neurons connect to excitatory
                nest.Connect(inhibitory, excitatory,
                           {'rule': 'pairwise_bernoulli', 'p': 0.2})
    
    def _setup_hierarchical_connectivity(self):
        """Setup hierarchical feedforward connectivity"""
        regions = list(self.brain_regions.keys())
        for i in range(len(regions) - 1):
            current_region = regions[i]
            next_region = regions[i + 1]
            
            # Connect excitatory neurons from current to next region
            if 'excitatory' in self.brain_regions[current_region]['neurons']:
                current_excitatory = self.brain_regions[current_region]['neurons']['excitatory']
                if 'excitatory' in self.brain_regions[next_region]['neurons']:
                    next_excitatory = self.brain_regions[next_region]['neurons']['excitatory']
                    nest.Connect(current_excitatory, next_excitatory,
                               {'rule': 'pairwise_bernoulli', 'p': 0.15})
    
    def simulate_brain_growth(self, duration: float) -> Dict:
        """
        Simulate brain growth and development over specified duration
        
        Args:
            duration: Simulation duration in milliseconds
            
        Returns:
            Dictionary containing simulation results
        """
        if not self.brain_regions:
            raise RuntimeError("No brain model loaded")
        
        try:
            # Add spike recorders to monitor activity
            spike_detectors = {}
            for region_name, region_data in self.brain_regions.items():
                for cell_type, neurons in region_data['neurons'].items():
                    detector = nest.Create('spike_recorder')
                    nest.Connect(neurons, detector)
                    spike_detectors[f"{region_name}_{cell_type}"] = detector
            
            # Run simulation
            nest.Simulate(duration)
            
            # Collect results
            results = self._collect_simulation_results(spike_detectors, duration)
            
            logger.info(f"Brain growth simulation completed: {duration}ms")
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {}
    
    def _collect_simulation_results(self, spike_detectors: Dict, duration: float) -> Dict:
        """Collect and process simulation results"""
        results = {
            'time_points': np.arange(0, duration, self.time_step),
            'region_activity': {},
            'network_activity': [],
            'development_stage': min(1.0, duration / 1000.0)
        }
        
        # Define time bins for firing rate calculation
        time_bins = np.arange(0, duration, 100)  # 100ms bins
        
        # Process spike data for each region
        for detector_name, detector in spike_detectors.items():
            spike_times = detector.get('events', 'times')
            spike_senders = detector.get('events', 'senders')
            
            if len(spike_times) > 0:
                # Calculate firing rates over time
                spike_counts = np.zeros(len(time_bins) - 1)
                
                for i in range(len(time_bins) - 1):
                    mask = (spike_times >= time_bins[i]) & (spike_times < time_bins[i + 1])
                    spike_counts[i] = np.sum(mask)
                
                results['region_activity'][detector_name] = {
                    'spike_times': spike_times,
                    'spike_senders': spike_senders,
                    'firing_rates': spike_counts / 0.1,  # Convert to Hz
                    'total_spikes': len(spike_times)
                }
            else:
                results['region_activity'][detector_name] = {
                    'spike_times': [],
                    'spike_senders': [],
                    'firing_rates': np.zeros(len(time_bins) - 1),
                    'total_spikes': 0
                }
        
        # Calculate overall network activity
        total_spikes = sum(data['total_spikes'] for data in results['region_activity'].values())
        results['total_network_spikes'] = total_spikes
        results['average_firing_rate'] = total_spikes / (duration / 1000.0) / sum(
            pop['count'] for pop in self.cell_populations.values()
        )
        
        return results
    
    def get_development_metrics(self) -> Dict:
        """Get current brain development metrics"""
        if not self.brain_regions:
            return {}
        
        metrics = {
            'development_stage': self.development_stage,
            'region_count': len(self.brain_regions),
            'total_neurons': sum(
                sum(len(neurons) for neurons in region['neurons'].values())
                for region in self.brain_regions.values()
            ),
            'cell_population_types': len(self.cell_populations),
            'simulation_time': self.simulation_time,
            'time_step': self.time_step
        }
        
        return metrics
    
    def add_external_input(self, region: str, cell_type: str, input_strength: float = 1.0):
        """
        Add external input to specific neural population
        
        Args:
            region: Target brain region
            cell_type: Target cell type
            input_strength: Strength of external input
        """
        try:
            if region in self.brain_regions and cell_type in self.brain_regions[region]['neurons']:
                neurons = self.brain_regions[region]['neurons'][cell_type]
                
                # Create DC input device
                dc_input = nest.Create('dc_generator', params={'amplitude': input_strength})
                nest.Connect(dc_input, neurons)
                
                logger.info(f"Added external input to {region}_{cell_type}: strength={input_strength}")
            else:
                logger.warning(f"Target population {region}_{cell_type} not found")
                
        except Exception as e:
            logger.error(f"Failed to add external input: {e}")
    
    def visualize_brain_development(self, 
                                  visualization_type: str = "3D",
                                  output_file: str = "brain_development.png") -> bool:
        """
        Visualize brain development using VisIt
        
        Args:
            visualization_type: Type of visualization (2D, 3D, time_series)
            output_file: Output image file
        """
        if not self.visit_interface:
            logger.warning("VisIt interface not available")
            return False
        
        try:
            # Prepare brain data for visualization
            brain_data = self._prepare_visualization_data()
            
            # Create visualization
            success = self.visit_interface.create_brain_visualization(
                brain_data, visualization_type
            )
            
            if success:
                # Export visualization
                self.visit_interface.export_visualization(output_file)
                logger.info(f"Brain development visualization exported to {output_file}")
                return True
            else:
                logger.error("Failed to create brain visualization")
                return False
                
        except Exception as e:
            logger.error(f"Failed to visualize brain development: {e}")
            return False
    
    def _prepare_visualization_data(self) -> Dict[str, Any]:
        """Prepare brain data for VisIt visualization"""
        try:
            brain_data = {
                "regions": {},
                "neurons": [],
                "connections": [],
                "time_series": {
                    "time": [],
                    "activity": []
                }
            }
            
            # Add region information
            for region_name, region_data in self.brain_regions.items():
                brain_data["regions"][region_name] = {
                    "position": [len(brain_data["regions"]), 0, 0],
                    "size": region_data.get("size", 0),
                    "neuron_count": sum(len(neurons) for neurons in region_data["neurons"].values())
                }
            
            # Add neuron information
            neuron_id = 0
            for region_name, region_data in self.brain_regions.items():
                for cell_type, neurons in region_data["neurons"].items():
                    for neuron in neurons:
                        brain_data["neurons"].append({
                            "id": neuron_id,
                            "type": cell_type,
                            "region": region_name,
                            "position": [neuron_id * 0.1, 0, 0],
                            "activity": 0.0  # Placeholder for activity
                        })
                        neuron_id += 1
            
            # Add time series data if available
            if hasattr(self, 'network_activity') and self.network_activity:
                brain_data["time_series"]["time"] = list(range(len(self.network_activity)))
                brain_data["time_series"]["activity"] = self.network_activity
            
            return brain_data
            
        except Exception as e:
            logger.error(f"Failed to prepare visualization data: {e}")
            return {}
    
    def analyze_brain_data_with_visit(self, 
                                    analysis_type: str = "statistics") -> Dict[str, Any]:
        """
        Analyze brain data using VisIt
        
        Args:
            analysis_type: Type of analysis (statistics, spatial, temporal)
        """
        if not self.visit_interface:
            logger.warning("VisIt interface not available")
            return {}
        
        try:
            # Prepare brain data
            brain_data = self._prepare_visualization_data()
            
            # Perform analysis
            results = self.visit_interface.analyze_brain_data(brain_data, analysis_type)
            
            logger.info(f"Brain data analysis completed: {analysis_type}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze brain data with VisIt: {e}")
            return {}
    
    def create_interactive_visualization(self, 
                                       brain_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create interactive visualization session
        
        Args:
            brain_data: Optional brain data, uses current simulation data if None
        """
        if not self.visit_interface:
            logger.warning("VisIt interface not available")
            return False
        
        try:
            if brain_data is None:
                brain_data = self._prepare_visualization_data()
            
            # Create 3D visualization
            success = self.visit_interface.create_brain_visualization(brain_data, "3D")
            
            if success:
                logger.info("Interactive visualization session created")
                logger.info("Use VisIt interface methods to interact with visualization")
                return True
            else:
                logger.error("Failed to create interactive visualization")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create interactive visualization: {e}")
            return False
    
    def cleanup(self):
        """Cleanup NEST resources and VisIt interface"""
        try:
            # Cleanup NEST
            nest.ResetKernel()
            logger.info("NEST resources cleaned up")
            
            # Cleanup VisIt interface
            if self.visit_interface:
                self.visit_interface.close()
                self.visit_interface = None
                logger.info("VisIt interface cleaned up")
                
        except Exception as e:
            logger.error(f"Failed to cleanup resources: {e}")
