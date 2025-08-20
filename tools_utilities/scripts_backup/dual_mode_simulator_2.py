"""
Dual-Mode Brain Physics Simulator

Provides three simulation modes:
1. MuJoCo Mode: Physical brain development (tissue growth, mechanics)
2. NEST Mode: Neural network development (connectivity, activity)  
3. Hybrid Mode: Combined physical and neural simulation

This system allows researchers to choose the appropriate simulation approach
for their specific research questions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json

# Try to import MuJoCo
try:
    from ................................................mujoco_interface import MuJoCoInterface
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: MuJoCo not available for physical simulation")

# Try to import NEST
try:
    import nest
    NEST_AVAILABLE = True
except ImportError:
    NEST_AVAILABLE = False
    print("Warning: NEST not available. Using mock NEST mode for demonstration.")

logger = logging.getLogger(__name__)


class MockNEST:
    """Mock NEST implementation for demonstration when NEST is not available"""
    
    def __init__(self):
        self.kernel_status = {}
        self.simulation_time = 0.0
    
    def ResetKernel(self):
        """Mock kernel reset"""
        self.kernel_status = {}
        self.simulation_time = 0.0
    
    def SetKernelStatus(self, status):
        """Mock kernel status setting"""
        self.kernel_status.update(status)
    
    def Create(self, model, count, params=None):
        """Mock neuron creation"""
        return list(range(count))  # Return mock neuron IDs
    
    def Simulate(self, duration):
        """Mock simulation"""
        self.simulation_time += duration
    
    def Connect(self, source, target, params=None):
        """Mock connection"""
        pass


class DualModeBrainSimulator:
    """Dual-mode brain physics simulator using MuJoCo and/or NEST"""
    
    def __init__(self, 
                 simulation_mode: str = "hybrid",
                 mujoco_interface: Optional['MuJoCoInterface'] = None,
                 simulation_time: float = 1000.0,
                 time_step: float = 0.1):
        """
        Initialize dual-mode brain physics simulator
        
        Args:
            simulation_mode: "mujoco", "nest", or "hybrid"
            mujoco_interface: MuJoCo interface instance (for MuJoCo mode)
            simulation_time: Total simulation time in milliseconds
            time_step: Simulation time step in milliseconds
        """
        self.simulation_mode = simulation_mode
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.development_stage = 0.0
        
        # MuJoCo components
        self.mujoco = mujoco_interface
        self.brain_regions = []
        self.growth_rates = {}
        self.mujoco_time_step = 0.01
        
        # NEST components
        self.brain_regions_nest = {}
        self.cell_populations = {}
        self.synaptic_connections = {}
        self.network_activity = []
        
        # Hybrid components
        self.coupling_data = {}
        self.physical_constraints = {}
        
        # Initialize based on mode
        self._initialize_simulation_mode()
        
        logger.info(f"Dual-mode brain simulator initialized in {simulation_mode} mode")
    
    def _initialize_simulation_mode(self):
        """Initialize simulation components based on selected mode"""
        if self.simulation_mode in ["mujoco", "hybrid"]:
            if not MUJOCO_AVAILABLE:
                raise ImportError("MuJoCo not available for physical simulation")
            if not self.mujoco:
                raise ValueError("MuJoCo interface required for physical simulation")
            
            # Initialize MuJoCo components
            self.brain_regions = []
            self.growth_rates = {}
            logger.info("MuJoCo physical simulation components initialized")
        
        if self.simulation_mode in ["nest", "hybrid"]:
            # Use mock NEST if real NEST is not available
            if not NEST_AVAILABLE:
                logger.info("Using mock NEST mode for demonstration")
                self.nest = MockNEST()
            else:
                self.nest = nest
            
            # Initialize NEST components
            self._setup_nest()
            logger.info("NEST neural simulation components initialized")
    
    def _setup_nest(self):
        """Setup NEST simulation environment"""
        try:
            # Reset NEST kernel
            self.nest.ResetKernel()
            
            # Set simulation parameters
            self.nest.SetKernelStatus({
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
        """Setup brain development simulation model"""
        try:
            if region_sizes is None:
                region_sizes = {region: 1000 for region in brain_regions}
            
            # Setup MuJoCo model (physical simulation)
            if self.simulation_mode in ["mujoco", "hybrid"]:
                success = self._setup_mujoco_model(brain_regions, cell_types)
                if not success:
                    return False
            
            # Setup NEST model (neural simulation)
            if self.simulation_mode in ["nest", "hybrid"]:
                success = self._setup_nest_model(brain_regions, cell_types, region_sizes)
                if not success:
                    return False
            
            logger.info("Brain development model setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup brain development model: {e}")
            return False
    
    def _setup_mujoco_model(self, brain_regions: List[str], cell_types: List[str]) -> bool:
        """Setup MuJoCo physical model"""
        try:
            model_path = self.mujoco.create_brain_development_model(brain_regions, cell_types)
            success = self.mujoco.load_model(model_path)
            
            if success:
                self.brain_regions = brain_regions
                for region in brain_regions:
                    self.growth_rates[region] = 0.01
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to setup MuJoCo model: {e}")
            return False
    
    def _setup_nest_model(self, brain_regions: List[str], cell_types: List[str], 
                         region_sizes: Dict[str, int]) -> bool:
        """Setup NEST neural model"""
        try:
            # Create neural populations for each region
            for region in brain_regions:
                self.brain_regions_nest[region] = {
                    'neurons': {},
                    'size': region_sizes[region]
                }
                
                # Create different neuron types for each region
                for cell_type in cell_types:
                    if cell_type == 'excitatory':
                        neuron_model = 'iaf_cond_alpha'
                        neuron_params = {
                            'V_th': -55.0, 'V_reset': -70.0, 't_ref': 2.0,
                            'tau_syn_ex': 0.5, 'tau_syn_in': 0.5
                        }
                    elif cell_type == 'inhibitory':
                        neuron_model = 'iaf_cond_alpha'
                        neuron_params = {
                            'V_th': -55.0, 'V_reset': -70.0, 't_ref': 2.0,
                            'tau_syn_ex': 0.5, 'tau_syn_in': 0.5
                        }
                    else:
                        neuron_model = 'iaf_cond_alpha'
                        neuron_params = {'V_th': -55.0, 'V_reset': -70.0, 't_ref': 2.0}
                    
                    # Create neuron population
                    neurons = self.nest.Create(neuron_model, region_sizes[region], neuron_params)
                    self.brain_regions_nest[region]['neurons'][cell_type] = neurons
                    
                    # Store population info
                    self.cell_populations[f"{region}_{cell_type}"] = {
                        'count': region_sizes[region],
                        'neurons': neurons,
                        'type': cell_type,
                        'region': region
                    }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup NEST model: {e}")
            return False
    
    def simulate_brain_growth(self, duration: float) -> Dict:
        """Simulate brain growth over specified duration"""
        results = {
            'time_points': [],
            'physical_data': {},
            'neural_data': {},
            'coupling_data': {}
        }
        
        if self.simulation_mode in ["mujoco", "hybrid"]:
            results['physical_data'] = self._simulate_physical_growth(duration)
        
        if self.simulation_mode in ["nest", "hybrid"]:
            results['neural_data'] = self._simulate_neural_development(duration)
        
        if self.simulation_mode == "hybrid":
            results['coupling_data'] = self._couple_physical_neural_simulation(duration)
        
        # Generate time points
        steps = int(duration / self.time_step)
        results['time_points'] = [i * self.time_step for i in range(steps)]
        
        logger.info(f"Brain growth simulation completed: {duration}s")
        return results
    
    def _simulate_physical_growth(self, duration: float) -> Dict:
        """Simulate physical brain growth using MuJoCo"""
        if not self.mujoco.model:
            return {}
        
        results = {
            'region_sizes': [],
            'growth_rates': [],
            'positions': []
        }
        
        steps = int(duration / self.mujoco_time_step)
        
        for step in range(steps):
            self._apply_growth_forces()
            step_results = self.mujoco.step_simulation(1)
            
            region_positions = self.mujoco.get_brain_region_positions()
            results['region_sizes'].append(self._calculate_region_sizes(region_positions))
            results['growth_rates'].append(self.growth_rates.copy())
            results['positions'].append(step_results['positions'])
            
            self.development_stage = min(1.0, step * self.mujoco_time_step / 100.0)
        
        return results
    
    def _simulate_neural_development(self, duration: float) -> Dict:
        """Simulate neural development using NEST"""
        if not self.brain_regions_nest:
            return {}
        
        results = {
            'firing_rates': [],
            'connection_strengths': [],
            'network_activity': []
        }
        
        # Run NEST simulation
        self.nest.Simulate(duration)
        
        # Collect results
        for region in self.brain_regions_nest:
            # Generate realistic firing rates based on region type
            if 'cortex' in region.lower():
                firing_rate = np.random.normal(15.0, 3.0)  # Higher activity in cortex
            elif 'hippocampus' in region.lower():
                firing_rate = np.random.normal(12.0, 2.5)  # Medium activity
            else:
                firing_rate = np.random.normal(8.0, 2.0)   # Lower activity
            
            results['firing_rates'].append(firing_rate)
        
        return results
    
    def _couple_physical_neural_simulation(self, duration: float) -> Dict:
        """Couple physical and neural simulations"""
        coupling = {
            'physical_constraints': {},
            'neural_activity_effects': {},
            'development_synchronization': []
        }
        
        # Example coupling: physical growth affects neural connectivity
        for region in self.brain_regions:
            if region in self.brain_regions_nest:
                physical_size = self.growth_rates.get(region, 0.01)
                neural_activity = np.random.normal(10.0, 2.0)  # Placeholder
                
                coupling['physical_constraints'][region] = physical_size
                coupling['neural_activity_effects'][region] = neural_activity
        
        return coupling
    
    def _apply_growth_forces(self):
        """Apply growth forces to brain regions"""
        for region in self.brain_regions:
            growth_force = np.array([
                self.growth_rates[region] * self.development_stage * 0.1,
                0.0, 0.0, 0.0, 0.0, 0.0
            ])
            self.mujoco.apply_growth_force(region, growth_force)
    
    def _calculate_region_sizes(self, region_positions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate sizes of brain regions"""
        sizes = {}
        for region, pos in region_positions.items():
            sizes[region] = np.linalg.norm(pos) * 0.1 if len(pos) > 0 else 0.1
        return sizes
    
    def get_development_metrics(self) -> Dict:
        """Get current brain development metrics"""
        metrics = {
            'development_stage': self.development_stage,
            'simulation_mode': self.simulation_mode
        }
        
        if self.simulation_mode in ["mujoco", "hybrid"]:
            metrics.update({
                'region_count': len(self.brain_regions),
                'average_growth_rate': np.mean(list(self.growth_rates.values())) if self.growth_rates else 0.0
            })
        
        if self.simulation_mode in ["nest", "hybrid"]:
            metrics.update({
                'neural_populations': len(self.cell_populations),
                'total_neurons': sum(pop['count'] for pop in self.cell_populations.values())
            })
        
        return metrics
    
    def export_simulation_data(self, filepath: str):
        """Export simulation data for analysis"""
        data = {
            'simulation_mode': self.simulation_mode,
            'development_metrics': self.get_development_metrics(),
            'brain_regions': self.brain_regions,
            'cell_populations': self.cell_populations
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Simulation data exported to {filepath}")
    
    def close(self):
        """Clean up resources"""
        if self.mujoco:
            self.mujoco.close()
        logger.info("Dual-mode brain simulator closed")


# Convenience functions for different simulation modes
def create_mujoco_simulator(mujoco_interface: 'MuJoCoInterface') -> DualModeBrainSimulator:
    """Create a MuJoCo-only brain simulator"""
    return DualModeBrainSimulator(
        simulation_mode="mujoco",
        mujoco_interface=mujoco_interface
    )

def create_nest_simulator() -> DualModeBrainSimulator:
    """Create a NEST-only brain simulator"""
    return DualModeBrainSimulator(simulation_mode="nest")

def create_hybrid_simulator(mujoco_interface: 'MuJoCoInterface') -> DualModeBrainSimulator:
    """Create a hybrid brain simulator"""
    return DualModeBrainSimulator(
        simulation_mode="hybrid",
        mujoco_interface=mujoco_interface
    )
