#!/usr/bin/env python3
"""
🚀 Google Colab Experiment Runner for Quark Brain Simulation

This module provides Colab-optimized classes and utilities for running
brain simulation experiments with GPU acceleration and interactive visualization.

Usage:
    from colab_experiment_runner import ColabBrainExperiment
    
    experiment = ColabBrainExperiment(
        population_size=200,
        use_gpu=True,
        experiment_type="parameter_sweep"
    )
    
    results = experiment.run()
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, field
from datetime import datetime
import os, sys

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    import torch
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Plotting libraries not available. Install with: pip install torch plotly")

# Import Quark components
try:
    from core.neural_components import SpikingNeuron, HebbianSynapse, STDP, NeuralPopulation
    from core.neural_parameters import NeuralParameters, NeuromodulatorSystem
    from core.biological_validator import BiologicalValidator
    from core.performance_optimizer import PerformanceOptimizer
except ImportError as e:
    print(f"⚠️ Could not import Quark components: {e}")
    print("Make sure you're running from the Quark root directory")


@dataclass
class ExperimentConfig:
    """Configuration for Colab experiments"""
    population_size: int = 200
    num_epochs: int = 100
    learning_rate: float = 0.01
    use_gpu: bool = True
    save_results: bool = True
    save_to_drive: bool = False
    experiment_name: str = "neural_training"
    validation_enabled: bool = True


@dataclass
class ExperimentResults:
    """Results from a Colab experiment"""
    config: ExperimentConfig
    training_results: Dict[str, Any]
    validation_results: Optional[Dict[str, Any]] = None
    parameter_sweep_results: Optional[List[Dict]] = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ColabBrainTrainer:
    """Colab-optimized brain training system"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = self._get_device()
        
        # Initialize components
        self.neural_params = NeuralParameters()
        self.neuromod_system = NeuromodulatorSystem()
        self.validator = BiologicalValidator() if config.validation_enabled else None
        self.optimizer = PerformanceOptimizer()
        
        # Create neural population
        self.population = self._create_population()
        
        # Training metrics
        self.training_history = {
            'loss': [],
            'firing_rates': [],
            'synchrony': [],
            'plasticity': []
        }
        
        print(f"🧠 Initialized brain trainer:")
        print(f"   - Population size: {self.config.population_size}")
        print(f"   - Device: {self.device}")
        print(f"   - GPU enabled: {self.config.use_gpu}")
        
    def _get_device(self):
        """Get optimal device for computation"""
        if self.config.use_gpu and 'torch' in sys.modules:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"🚀 GPU acceleration enabled: {gpu_name} ({memory_gb:.1f} GB)")
                return device
            else:
                print("⚠️ CUDA not available, using CPU")
        return torch.device('cpu')
        
    def _create_population(self) -> NeuralPopulation:
        """Create neural population with optimized parameters"""
        neurons = []
        for i in range(self.config.population_size):
            neuron = SpikingNeuron(
                threshold=self.neural_params.threshold,
                decay_rate=self.neural_params.decay_rate,
                refractory_period=self.neural_params.refractory_period
            )
            neurons.append(neuron)
        
        population = NeuralPopulation(
            neurons=neurons,
            connection_probability=0.2,
            connection_strength=0.1
        )
        
        return population
    
    def train_stdp_learning(self) -> Dict[str, Any]:
        """Train STDP learning with progress tracking"""
        print(f"🎯 Training STDP learning for {self.config.num_epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # Generate biologically realistic input patterns
            input_current = self._generate_input_pattern(epoch)
            
            # Step population
            result = self.population.step(dt=0.001, input_current=input_current)
            
            # Calculate biological metrics
            metrics = self._calculate_biological_metrics()
            
            # Store metrics
            for key, value in metrics.items():
                self.training_history[key].append(value)
            
            # Calculate loss (biological objective)
            loss = self._calculate_biological_loss(metrics)
            self.training_history['loss'].append(loss)
            
            # Progress reporting
            if epoch % 20 == 0 or epoch == self.config.num_epochs - 1:
                self._print_training_progress(epoch, loss, metrics)
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'final_loss': self.training_history['loss'][-1],
            'final_metrics': {k: v[-1] for k, v in self.training_history.items() if v},
            'convergence_epoch': self._find_convergence_epoch(),
            'stability_score': self._calculate_stability_score()
        }
    
    def _generate_input_pattern(self, epoch: int) -> np.ndarray:
        """Generate biologically realistic input patterns"""
        # Simulate developmental input patterns
        base_current = 0.5
        
        # Add structured patterns that encourage learning
        pattern_frequency = 0.1  # Hz
        pattern_phase = 2 * np.pi * pattern_frequency * epoch
        
        # Create spatial input pattern
        spatial_pattern = np.sin(np.linspace(0, 4*np.pi, self.config.population_size))
        temporal_modulation = 0.3 * np.sin(pattern_phase)
        
        input_current = base_current + temporal_modulation * spatial_pattern
        
        # Add noise for robustness
        noise = np.random.normal(0, 0.1, self.config.population_size)
        input_current += noise
        
        # Ensure positive currents
        input_current = np.maximum(input_current, 0.1)
        
        return input_current
    
    def _calculate_biological_metrics(self) -> Dict[str, float]:
        """Calculate biologically relevant metrics"""
        # Firing rate
        firing_rate = np.mean([getattr(n, 'spike_count', 0) for n in self.population.neurons])
        
        # Neural synchrony
        synchrony = self._calculate_synchrony()
        
        # Plasticity strength
        plasticity = self._calculate_plasticity_strength()
        
        # Network efficiency
        efficiency = self._calculate_network_efficiency()
        
        return {
            'firing_rates': firing_rate,
            'synchrony': synchrony,
            'plasticity': plasticity,
            'efficiency': efficiency
        }
    
    def _calculate_synchrony(self) -> float:
        """Calculate neural synchrony using spike timing"""
        spike_times = []
        for neuron in self.population.neurons:
            if hasattr(neuron, 'last_spike_time') and neuron.last_spike_time > 0:
                spike_times.append(neuron.last_spike_time)
        
        if len(spike_times) < 2:
            return 0.0
            
        # Synchrony as inverse of coefficient of variation
        if np.mean(spike_times) > 0:
            cv = np.std(spike_times) / np.mean(spike_times)
            return 1.0 / (1.0 + cv)
        return 0.0
    
    def _calculate_plasticity_strength(self) -> float:
        """Calculate average synaptic plasticity strength"""
        strengths = []
        for connection in self.population.connections:
            if hasattr(connection, 'weight'):
                strengths.append(abs(connection.weight))
        
        return np.mean(strengths) if strengths else 0.0
    
    def _calculate_network_efficiency(self) -> float:
        """Calculate network efficiency metric"""
        # Simple efficiency: ratio of active connections to total possible
        active_connections = len([c for c in self.population.connections 
                                if hasattr(c, 'weight') and abs(c.weight) > 0.01])
        total_possible = self.config.population_size * (self.config.population_size - 1)
        
        return active_connections / total_possible if total_possible > 0 else 0.0
    
    def _calculate_biological_loss(self, metrics: Dict[str, float]) -> float:
        """Calculate biologically motivated loss function"""
        # Target biological values
        target_firing_rate = 15.0  # Hz (typical cortical)
        target_synchrony = 0.3     # Moderate synchrony
        target_efficiency = 0.15   # Small-world efficiency
        
        # Calculate deviations
        rate_loss = abs(metrics['firing_rates'] - target_firing_rate) / target_firing_rate
        sync_loss = abs(metrics['synchrony'] - target_synchrony) / target_synchrony
        efficiency_loss = abs(metrics['efficiency'] - target_efficiency) / target_efficiency
        
        # Weighted combination
        total_loss = 0.5 * rate_loss + 0.3 * sync_loss + 0.2 * efficiency_loss
        
        return total_loss
    
    def _print_training_progress(self, epoch: int, loss: float, metrics: Dict[str, float]):
        """Print training progress"""
        print(f"Epoch {epoch:3d}: Loss={loss:.4f}, "
              f"Rate={metrics['firing_rates']:.2f}Hz, "
              f"Sync={metrics['synchrony']:.3f}, "
              f"Eff={metrics['efficiency']:.3f}")
    
    def _find_convergence_epoch(self) -> int:
        """Find epoch where training converged"""
        if len(self.training_history['loss']) < 10:
            return -1
        
        # Find where loss stabilizes (derivative approaches zero)
        losses = self.training_history['loss']
        window_size = 10
        
        for i in range(window_size, len(losses)):
            recent_losses = losses[i-window_size:i]
            if np.std(recent_losses) < 0.01:  # Converged
                return i - window_size
        
        return -1  # Not converged
    
    def _calculate_stability_score(self) -> float:
        """Calculate training stability score"""
        if len(self.training_history['loss']) < 10:
            return 0.0
        
        # Stability = inverse of loss variance in final 20% of training
        final_portion = int(0.8 * len(self.training_history['loss']))
        final_losses = self.training_history['loss'][final_portion:]
        
        variance = np.var(final_losses)
        return 1.0 / (1.0 + variance)
    
    def biological_validation(self) -> Optional[Dict[str, Any]]:
        """Run biological validation if enabled"""
        if not self.validator:
            return None
            
        print("🔬 Running biological validation...")
        
        # Extract final metrics for validation
        if not self.training_history['firing_rates']:
            return None
            
        metrics = {
            'firing_rate': self.training_history['firing_rates'][-1],
            'synchrony': self.training_history['synchrony'][-1],
            'population_size': self.config.population_size,
            'connection_density': len(self.population.connections) / (self.config.population_size ** 2)
        }
        
        # Run validation
        validation_result = self.validator.validate_neural_population(metrics)
        
        return {
            'validation_score': validation_result.overall_score,
            'firing_rate_valid': validation_result.firing_rate_valid,
            'connectivity_valid': validation_result.connectivity_valid,
            'biological_plausibility': validation_result.overall_score > 0.7,
            'detailed_results': validation_result
        }


class ColabBrainExperiment:
    """Main experiment runner for Colab"""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.trainer = ColabBrainTrainer(self.config)
        self.results = None
        
    def run_training_experiment(self) -> ExperimentResults:
        """Run complete training experiment"""
        print("🚀 Starting Colab brain training experiment...")
        start_time = time.time()
        
        # Train the network
        training_results = self.trainer.train_stdp_learning()
        
        # Run biological validation
        validation_results = self.trainer.biological_validation()
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        
        # Create results object
        self.results = ExperimentResults(
            config=self.config,
            training_results=training_results,
            validation_results=validation_results,
            execution_time=execution_time
        )
        
        self._print_experiment_summary()
        
        return self.results
    
    def run_parameter_sweep(self, 
                          population_sizes: List[int] = [50, 100, 200],
                          learning_rates: List[float] = [0.001, 0.01, 0.1]) -> ExperimentResults:
        """Run parameter sweep experiment"""
        print("🔍 Starting parameter sweep experiment...")
        start_time = time.time()
        
        sweep_results = []
        total_combinations = len(population_sizes) * len(learning_rates)
        combination_count = 0
        
        for pop_size in population_sizes:
            for lr in learning_rates:
                combination_count += 1
                print(f"\n📊 Testing combination {combination_count}/{total_combinations}: "
                      f"population={pop_size}, learning_rate={lr}")
                
                # Create configuration for this combination
                sweep_config = ExperimentConfig(
                    population_size=pop_size,
                    learning_rate=lr,
                    num_epochs=50,  # Shorter for sweep
                    use_gpu=self.config.use_gpu,
                    validation_enabled=self.config.validation_enabled
                )
                
                # Create trainer and run experiment
                sweep_trainer = ColabBrainTrainer(sweep_config)
                training_results = sweep_trainer.train_stdp_learning()
                validation_results = sweep_trainer.biological_validation()
                
                # Store results
                sweep_results.append({
                    'population_size': pop_size,
                    'learning_rate': lr,
                    'final_loss': training_results['final_loss'],
                    'stability_score': training_results['stability_score'],
                    'validation_score': validation_results['validation_score'] if validation_results else 0.0,
                    'training_time': training_results['training_time'],
                    'convergence_epoch': training_results['convergence_epoch']
                })
        
        # Find best parameters
        best_result = max(sweep_results, 
                         key=lambda x: x['validation_score'] if x['validation_score'] > 0 else -x['final_loss'])
        
        execution_time = time.time() - start_time
        
        # Create results object
        self.results = ExperimentResults(
            config=self.config,
            training_results={'parameter_sweep_summary': best_result},
            parameter_sweep_results=sweep_results,
            execution_time=execution_time
        )
        
        self._print_parameter_sweep_summary(best_result, len(sweep_results))
        
        return self.results
    
    def _print_experiment_summary(self):
        """Print experiment summary"""
        if not self.results:
            return
            
        print("\n" + "="*60)
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print(f"\n📊 Training Results:")
        training = self.results.training_results
        print(f"   - Execution time: {self.results.execution_time:.2f} seconds")
        print(f"   - Final loss: {training['final_loss']:.4f}")
        print(f"   - Stability score: {training['stability_score']:.3f}")
        print(f"   - Convergence epoch: {training['convergence_epoch']}")
        
        if self.results.validation_results:
            validation = self.results.validation_results
            print(f"\n🔬 Biological Validation:")
            print(f"   - Validation score: {validation['validation_score']:.3f}")
            print(f"   - Biologically plausible: {validation['biological_plausibility']}")
            print(f"   - Firing rate valid: {validation['firing_rate_valid']}")
            print(f"   - Connectivity valid: {validation['connectivity_valid']}")
        
        print(f"\n⚙️ Configuration:")
        print(f"   - Population size: {self.config.population_size}")
        print(f"   - Learning rate: {self.config.learning_rate}")
        print(f"   - Training epochs: {self.config.num_epochs}")
        print(f"   - GPU acceleration: {self.config.use_gpu}")
    
    def _print_parameter_sweep_summary(self, best_result: Dict, total_tested: int):
        """Print parameter sweep summary"""
        print("\n" + "="*60)
        print("🎯 PARAMETER SWEEP COMPLETED")
        print("="*60)
        
        print(f"\n📈 Sweep Results:")
        print(f"   - Total combinations tested: {total_tested}")
        print(f"   - Total execution time: {self.results.execution_time:.2f} seconds")
        
        print(f"\n🏆 Best Parameters Found:")
        print(f"   - Population size: {best_result['population_size']}")
        print(f"   - Learning rate: {best_result['learning_rate']}")
        print(f"   - Validation score: {best_result['validation_score']:.3f}")
        print(f"   - Final loss: {best_result['final_loss']:.4f}")
        print(f"   - Training time: {best_result['training_time']:.2f}s")
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save experiment results to file"""
        if not self.results:
            raise ValueError("No results to save. Run an experiment first.")
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"colab_experiment_{self.config.experiment_name}_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        results_dict = {
            'config': self.config.__dict__,
            'training_results': self.results.training_results,
            'validation_results': self.results.validation_results,
            'parameter_sweep_results': self.results.parameter_sweep_results,
            'execution_time': self.results.execution_time,
            'timestamp': self.results.timestamp,
            'training_history': self.trainer.training_history
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")
        
        # Optionally save to Google Drive
        if self.config.save_to_drive:
            try:
                drive_path = f'/content/drive/MyDrive/quark_experiments/{filename}'
                os.makedirs(os.path.dirname(drive_path), exist_ok=True)
                with open(drive_path, 'w') as f:
                    json.dump(results_dict, f, indent=2, default=str)
                print(f"☁️ Results also saved to Google Drive: {drive_path}")
            except Exception as e:
                print(f"⚠️ Could not save to Google Drive: {e}")
        
        return filename
    
    def create_visualization(self):
        """Create interactive visualization (requires plotly)"""
        if not PLOTTING_AVAILABLE:
            print("⚠️ Plotting libraries not available. Install with: pip install plotly")
            return None
            
        if not self.results or not hasattr(self.trainer, 'training_history'):
            print("⚠️ No training data available for visualization")
            return None
        
        return self._create_plotly_dashboard()
    
    def _create_plotly_dashboard(self):
        """Create Plotly dashboard"""
        history = self.trainer.training_history
        epochs = list(range(len(history['loss'])))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Firing Rate', 'Neural Synchrony', 'Network Efficiency'),
            specs=[[{}, {}], [{}, {}]]
        )
        
        # Training loss
        fig.add_trace(
            go.Scatter(x=epochs, y=history['loss'], name='Loss', 
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Firing rate with target line
        fig.add_trace(
            go.Scatter(x=epochs, y=history['firing_rates'], name='Firing Rate',
                      line=dict(color='blue', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=15.0, line_dash="dash", line_color="green", 
                      annotation_text="Target (15 Hz)", row=1, col=2)
        
        # Neural synchrony
        fig.add_trace(
            go.Scatter(x=epochs, y=history['synchrony'], name='Synchrony',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        # Network efficiency
        if 'efficiency' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['efficiency'], name='Efficiency',
                          line=dict(color='orange', width=2)),
                row=2, col=2
            )
        
        fig.update_layout(
            title="🧠 Neural Training Dashboard",
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Hz", row=1, col=2)
        fig.update_yaxes(title_text="Synchrony", row=2, col=1)
        fig.update_yaxes(title_text="Efficiency", row=2, col=2)
        
        return fig


# Convenience functions for easy Colab usage
def quick_training_experiment(population_size: int = 200, 
                            num_epochs: int = 100,
                            use_gpu: bool = True) -> ExperimentResults:
    """Quick training experiment with default settings"""
    config = ExperimentConfig(
        population_size=population_size,
        num_epochs=num_epochs,
        use_gpu=use_gpu,
        experiment_name="quick_training"
    )
    
    experiment = ColabBrainExperiment(config)
    return experiment.run_training_experiment()


def quick_parameter_sweep(use_gpu: bool = True) -> ExperimentResults:
    """Quick parameter sweep with default ranges"""
    config = ExperimentConfig(
        use_gpu=use_gpu,
        experiment_name="quick_parameter_sweep"
    )
    
    experiment = ColabBrainExperiment(config)
    return experiment.run_parameter_sweep()


if __name__ == "__main__":
    # Example usage
    print("🧠 Quark Brain Simulation - Colab Experiment Runner")
    print("This module provides tools for running brain simulation experiments in Google Colab")
    print("\nExample usage:")
    print("  from colab_experiment_runner import quick_training_experiment")
    print("  results = quick_training_experiment(population_size=200)")
    print("  results.save_results()")
