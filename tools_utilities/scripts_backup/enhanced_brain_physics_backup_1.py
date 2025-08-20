"""
Enhanced Brain Physics Simulator

Integrates MuJoCo physics simulation with neural network dynamics:
- Tissue growth and deformation
- Neural migration and axon guidance
- Real-time neural network simulation
- Synaptic plasticity and learning
- Comprehensive visualization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
import matplotlib.pyplot as plt
from ................................................mujoco_interface import MuJoCoInterface
from ................................................neural_simulator import NeuralSimulator, NeuronType
from ................................................neural_visualization import NeuralVisualizer

logger = logging.getLogger(__name__)


class EnhancedBrainPhysicsSimulator:
    """Enhanced brain physics simulator with neural integration"""
    
    def __init__(self, mujoco_interface: MuJoCoInterface):
        """
        Initialize enhanced brain physics simulator
        
        Args:
            mujoco_interface: MuJoCo interface instance
        """
        self.mujoco = mujoco_interface
        self.brain_regions = []
        self.cell_populations = {}
        self.growth_rates = {}
        self.development_stage = 0.0
        self.time_step = 0.01
        
        # Neural simulation components
        self.neural_simulator = None
        self.neural_visualizer = None
        self.neural_integration = False
        
        # Simulation state
        self.simulation_running = False
        self.current_time = 0.0
        
    def setup_brain_development_model(self, 
                                    brain_regions: List[str],
                                    cell_types: List[str],
                                    enable_neural_simulation: bool = True) -> bool:
        """Setup brain development simulation model with neural integration"""
        try:
            # Setup MuJoCo physics model
            model_path = self.mujoco.create_brain_development_model(
                brain_regions, cell_types
            )
            
            success = self.mujoco.load_model(model_path)
            if not success:
                return False
            
            self.brain_regions = brain_regions
            for region in brain_regions:
                self.growth_rates[region] = 0.01
            
            for cell_type in cell_types:
                self.cell_populations[cell_type] = {
                    'count': 100,
                    'migration_rate': 0.001
                }
            
            # Setup neural simulation if enabled
            if enable_neural_simulation:
                self._setup_neural_simulation()
            
            logger.info("Enhanced brain development model setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup brain development model: {e}")
            return False
    
    def _setup_neural_simulation(self):
        """Setup neural network simulation for brain regions"""
        try:
            self.neural_simulator = NeuralSimulator(self.brain_regions)
            self.neural_visualizer = NeuralVisualizer()
            self.neural_integration = True
            
            # Configure neural networks for each brain region
            region_configs = {}
            for region in self.brain_regions:
                region_configs[region] = {
                    'neuron_count': 200,  # Adjust based on region size
                    'excitatory_ratio': 0.8,
                    'connections': [r for r in self.brain_regions if r != region]
                }
            
            self.neural_simulator.setup_brain_regions(region_configs)
            logger.info("Neural simulation setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup neural simulation: {e}")
            self.neural_integration = False
    
    def simulate_brain_growth(self, duration: float, 
                             include_neural_activity: bool = True,
                             real_time_sync: bool = True) -> Dict:
        """Simulate brain growth with optional real-time neural activity"""
        if not self.mujoco.model:
            raise RuntimeError("No brain model loaded")
        
        results = {
            'time_points': [],
            'region_sizes': [],
            'growth_rates': [],
            'neural_activity': None,
            'development_stages': []
        }
        
        steps = int(duration / self.time_step)
        self.simulation_running = True
        
        # Run neural simulation in parallel if enabled
        neural_results = None
        if include_neural_activity and self.neural_integration:
            if real_time_sync:
                neural_results = self._run_synchronized_neural_simulation(duration, steps)
            else:
                neural_results = self._run_neural_simulation(duration)
            results['neural_activity'] = neural_results
        
        for step in range(steps):
            self._apply_growth_forces()
            step_results = self.mujoco.step_simulation(1)
            
            current_time = step * self.time_step
            self.current_time = current_time
            
            results['time_points'].append(current_time)
            
            region_positions = self.mujoco.get_brain_region_positions()
            results['region_sizes'].append(self._calculate_region_sizes(region_positions))
            results['growth_rates'].append(self.growth_rates.copy())
            
            self.development_stage = min(1.0, current_time / 100.0)
            results['development_stages'].append(self.development_stage)
            
            # Update neural simulation if running synchronized
            if real_time_sync and include_neural_activity and self.neural_integration:
                self._step_neural_simulation_synchronized(step, neural_results)
        
        self.simulation_running = False
        logger.info(f"Enhanced brain growth simulation completed: {duration}s")
        return results
    
    def _run_neural_simulation(self, duration: float) -> Dict:
        """Run neural network simulation independently"""
        if not self.neural_simulator:
            return {}
        
        # Add external stimuli to create interesting activity patterns
        self._add_external_stimuli()
        
        # Run neural simulation
        neural_results = self.neural_simulator.run_simulation(duration * 1000)  # Convert to ms
        return neural_results
    
    def _run_synchronized_neural_simulation(self, duration: float, steps: int) -> Dict:
        """Setup neural simulation for synchronized stepping"""
        if not self.neural_simulator:
            return {}
        
        # Add external stimuli
        self._add_external_stimuli()
        
        # Initialize results structure
        neural_results = {
            'network_activity': [],
            'region_activities': [],
            'spike_data': [],
            'final_weights': []
        }
        
        return neural_results
    
    def _step_neural_simulation_synchronized(self, step: int, neural_results: Dict):
        """Step neural simulation in sync with physics simulation"""
        if not self.neural_simulator:
            return
        
        # Step neural network
        external_inputs = self._calculate_external_inputs(step)
        self.neural_simulator.network.step(external_inputs)
        
        # Record activity
        if step % 10 == 0:  # Record every 10 steps
            neural_results['network_activity'].append(
                self.neural_simulator.network.get_network_activity()
            )
            
            region_activities = {}
            for region in self.brain_regions:
                region_activities[region] = self.neural_simulator.network.get_region_activity(region)
            neural_results['region_activities'].append(region_activities)
    
    def _add_external_stimuli(self):
        """Add external stimuli to create realistic neural activity patterns"""
        if not self.neural_simulator:
            return
        
        # Add sensory input to sensory neurons
        for region in self.brain_regions:
            if 'sensory' in region.lower() or 'visual' in region.lower():
                region_neurons = self.neural_simulator.network.region_neurons.get(region, [])
                if region_neurons:
                    # Add periodic stimulus to first few neurons
                    for i, neuron_id in enumerate(region_neurons[:5]):
                        stimulus_func = lambda t, freq=1.0 + i*0.5: 5.0 * np.sin(2 * np.pi * freq * t / 1000)
                        self.neural_simulator.add_external_stimulus(neuron_id, stimulus_func)
        
        # Add motor cortex stimulation
        for region in self.brain_regions:
            if 'motor' in region.lower():
                region_neurons = self.neural_simulator.network.region_neurons.get(region, [])
                if region_neurons:
                    # Add random bursts of activity
                    for i, neuron_id in enumerate(region_neurons[:3]):
                        stimulus_func = lambda t, base_freq=0.5: 3.0 * np.random.poisson(base_freq)
                        self.neural_simulator.add_external_stimulus(neuron_id, stimulus_func)
    
    def _calculate_external_inputs(self, step: int) -> Dict[int, float]:
        """Calculate external inputs for neural simulation based on physics state"""
        external_inputs = {}
        
        if not self.neural_simulator:
            return external_inputs
        
        # Add physics-based inputs (e.g., mechanical stress affecting neural activity)
        for region in self.brain_regions:
            region_neurons = self.neural_simulator.network.region_neurons.get(region, [])
            
            # Calculate mechanical stress for this region
            stress = self._calculate_mechanical_stress(region)
            
            # Apply stress-based modulation to neurons
            for neuron_id in region_neurons:
                # Stress can modulate neural excitability
                stress_modulation = 1.0 + stress * 0.1  # 10% modulation per unit stress
                external_inputs[neuron_id] = stress_modulation
        
        return external_inputs
    
    def _calculate_mechanical_stress(self, region: str) -> float:
        """Calculate mechanical stress for a brain region"""
        # Simplified stress calculation based on growth rate and development stage
        growth_rate = self.growth_rates.get(region, 0.01)
        stress = growth_rate * self.development_stage * 10.0
        return stress
    
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
        """Get current brain development metrics including neural activity"""
        if not self.mujoco.model:
            return {}
        
        region_positions = self.mujoco.get_brain_region_positions()
        
        metrics = {
            'development_stage': self.development_stage,
            'region_count': len(self.brain_regions),
            'cell_population': sum(pop['count'] for pop in self.cell_populations.values()),
            'average_growth_rate': np.mean(list(self.growth_rates.values())),
            'neural_integration': self.neural_integration,
            'current_time': self.current_time
        }
        
        # Add neural activity metrics if available
        if self.neural_integration and self.neural_simulator:
            neural_metrics = self.neural_simulator.network.get_network_activity()
            metrics['neural_activity'] = neural_metrics
        
        return metrics
    
    def create_comprehensive_visualization(self, results: Dict, save_dir: Optional[str] = None):
        """Create comprehensive visualization of brain development and neural activity"""
        if not results:
            logger.warning("No results to visualize")
            return
        
        # Create brain development plots
        self._plot_brain_development(results, save_dir)
        
        # Create neural activity visualizations if available
        if results.get('neural_activity') and self.neural_visualizer:
            self._plot_neural_activity(results['neural_activity'], save_dir)
    
    def _plot_brain_development(self, results: Dict, save_dir: Optional[str] = None):
        """Plot brain development metrics over time"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        time_points = results['time_points']
        
        # Plot 1: Region sizes over time
        region_sizes = results['region_sizes']
        for i, region in enumerate(self.brain_regions):
            sizes = [step[i] for step in region_sizes]
            ax1.plot(time_points, sizes, label=region, linewidth=2)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Region Size')
        ax1.set_title('Brain Region Growth Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Growth rates over time
        growth_rates = results['growth_rates']
        for i, region in enumerate(self.brain_regions):
            rates = [step[region] for step in growth_rates]
            ax2.plot(time_points, rates, label=region, linewidth=2)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Growth Rate')
        ax2.set_title('Growth Rates Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Development stage
        development_stages = results.get('development_stages', [])
        if development_stages:
            ax3.plot(time_points, development_stages, 'b-', linewidth=3)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Development Stage')
        ax3.set_title('Overall Development Progress')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # Plot 4: Cell population changes
        cell_populations = results.get('cell_populations', {})
        if cell_populations:
            for cell_type, population_data in cell_populations.items():
                if isinstance(population_data, dict) and 'count' in population_data:
                    ax4.bar(cell_type, population_data['count'], alpha=0.7)
        
        ax4.set_xlabel('Cell Type')
        ax4.set_ylabel('Population Count')
        ax4.set_title('Cell Population Distribution')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/enhanced_brain_development.png", dpi=300, bbox_inches='tight')
            logger.info(f"Brain development plots saved to {save_dir}")
        
        plt.show()
    
    def _plot_neural_activity(self, neural_results: Dict, save_dir: Optional[str] = None):
        """Plot neural activity visualizations"""
        if not self.neural_visualizer or not self.neural_simulator:
            return
        
        network = self.neural_simulator.network
        
        # Create comprehensive neural report
        self.neural_visualizer.create_comprehensive_report(network, neural_results, save_dir)
        
        # Create additional specialized plots
        if neural_results.get('spike_data'):
            spike_data = neural_results['spike_data']
            duration = max(spike[0] for spike in spike_data) if spike_data else 1000
            
            # Spike raster plot
            self.neural_visualizer.plot_spike_raster(
                spike_data, duration, 
                f"{save_dir}/enhanced_spike_raster.png" if save_dir else None
            )
            
            # Activity heatmap
            self.neural_visualizer.plot_activity_heatmap(
                network, 100.0,
                f"{save_dir}/enhanced_activity_heatmap.png" if save_dir else None
            )
    
    def run_comprehensive_simulation(self, duration: float, 
                                   save_results: bool = True,
                                   save_dir: Optional[str] = None,
                                   real_time_sync: bool = True) -> Dict:
        """Run comprehensive brain development simulation with visualization"""
        logger.info(f"Starting comprehensive enhanced brain development simulation: {duration}s")
        
        # Run simulation
        results = self.simulate_brain_growth(
            duration, 
            include_neural_activity=True,
            real_time_sync=real_time_sync
        )
        
        # Create visualizations
        self.create_comprehensive_visualization(results, save_dir)
        
        # Save results if requested
        if save_results and save_dir:
            import json
            import os
            
            os.makedirs(save_dir, exist_ok=True)
            
            # Save simulation data
            with open(f"{save_dir}/enhanced_simulation_results.json", 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = self._prepare_results_for_json(results)
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Enhanced simulation results saved to {save_dir}")
        
        return results
    
    def _prepare_results_for_json(self, results: Dict) -> Dict:
        """Prepare results dictionary for JSON serialization"""
        json_results = {}
        
        for key, value in results.items():
            if key == 'neural_activity':
                # Handle neural activity data specially
                if value:
                    json_results[key] = {
                        'network_activity': value.get('network_activity', {}),
                        'region_activities': value.get('region_activities', {}),
                        'spike_count': len(value.get('spike_data', [])),
                        'synapse_count': len(value.get('final_weights', []))
                    }
            elif isinstance(value, list) and len(value) > 0:
                # Convert numpy arrays to lists
                if isinstance(value[0], np.ndarray):
                    json_results[key] = [v.tolist() for v in value]
                elif isinstance(value[0], dict):
                    json_results[key] = value
                else:
                    json_results[key] = value
            else:
                json_results[key] = value
        
        return json_results
