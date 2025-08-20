"""
Neural Network Visualization Module

Provides visualization capabilities for:
- Network topology and connectivity
- Spike raster plots
- Activity heatmaps
- 3D neuron positions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NeuralVisualizer:
    """Visualization tools for neural networks"""
    
    def __init__(self):
        self.colors = {
            'excitatory': 'red',
            'inhibitory': 'blue',
            'interneuron': 'green',
            'sensory': 'orange',
            'motor': 'purple'
        }
    
    def plot_network_topology(self, network, save_path: Optional[str] = None):
        """Plot network topology showing neurons and connections"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Neuron positions by type
        for neuron in network.neurons.values():
            color = self.colors.get(neuron.neuron_type.value, 'gray')
            ax1.scatter(neuron.position[0], neuron.position[1], 
                       c=color, alpha=0.7, s=20, label=neuron.neuron_type.value)
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Neuron Positions by Type')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Connection matrix
        if network.synapses:
            max_id = max(network.neurons.keys())
            connection_matrix = np.zeros((max_id + 1, max_id + 1))
            
            for synapse in network.synapses:
                connection_matrix[synapse.pre_neuron_id][synapse.post_neuron_id] = synapse.weight
            
            im = ax2.imshow(connection_matrix, cmap='viridis', aspect='auto')
            ax2.set_xlabel('Post-synaptic Neuron ID')
            ax2.set_ylabel('Pre-synaptic Neuron ID')
            ax2.set_title('Synaptic Connection Matrix')
            plt.colorbar(im, ax=ax2, label='Synaptic Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Network topology saved to {save_path}")
        
        plt.show()
    
    def plot_spike_raster(self, spike_times: List[Tuple[float, int]], 
                          duration: float, save_path: Optional[str] = None):
        """Plot spike raster plot"""
        if not spike_times:
            logger.warning("No spike data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract spike data
        times, neuron_ids = zip(*spike_times)
        
        # Create raster plot
        ax.scatter(times, neuron_ids, alpha=0.6, s=1, c='black')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron ID')
        ax.set_title('Spike Raster Plot')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spike raster plot saved to {save_path}")
        
        plt.show()
    
    def plot_activity_heatmap(self, network, time_window: float = 100.0, 
                             save_path: Optional[str] = None):
        """Plot activity heatmap for different brain regions"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        regions = list(network.brain_regions)
        activity_data = []
        
        for region in regions:
            region_activity = network.get_region_activity(region)
            firing_rate = region_activity.get('firing_rate', 0.0)
            activity_data.append(firing_rate)
        
        # Create heatmap
        im = ax.imshow([activity_data], cmap='hot', aspect='auto')
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=45)
        ax.set_yticks([])
        ax.set_title(f'Brain Region Activity Heatmap (Last {time_window}ms)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Firing Rate (Hz)')
        
        # Add text annotations
        for i, (region, rate) in enumerate(zip(regions, activity_data)):
            ax.text(i, 0, f'{rate:.2f}', ha='center', va='center', 
                   color='white' if rate > 0.5 else 'black', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Activity heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_neuron_dynamics(self, network, neuron_ids: List[int], 
                           duration: float, save_path: Optional[str] = None):
        """Plot membrane potential dynamics for specific neurons"""
        if not neuron_ids:
            logger.warning("No neuron IDs provided for dynamics plot")
            return
        
        fig, axes = plt.subplots(len(neuron_ids), 1, figsize=(12, 3*len(neuron_ids)))
        if len(neuron_ids) == 1:
            axes = [axes]
        
        for i, neuron_id in enumerate(neuron_ids):
            if neuron_id in network.neurons:
                neuron = network.neurons[neuron_id]
                
                # Simulate membrane potential over time
                time_points = np.linspace(0, duration, int(duration/0.1))
                membrane_potentials = []
                
                # Reset neuron state for simulation
                original_potential = neuron.membrane_potential
                original_refractory = neuron.refractory_counter
                
                for t in time_points:
                    membrane_potentials.append(neuron.membrane_potential)
                    neuron.update(0.1, 0.0)  # No external input
                
                # Restore original state
                neuron.membrane_potential = original_potential
                neuron.refractory_counter = original_refractory
                
                axes[i].plot(time_points, membrane_potentials, linewidth=2)
                axes[i].axhline(y=neuron.threshold, color='red', linestyle='--', 
                               alpha=0.7, label='Threshold')
                axes[i].set_ylabel('Membrane Potential (mV)')
                axes[i].set_title(f'Neuron {neuron_id} ({neuron.neuron_type.value})')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
                
                if i == len(neuron_ids) - 1:
                    axes[i].set_xlabel('Time (ms)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Neuron dynamics plot saved to {save_path}")
        
        plt.show()
    
    def create_animated_network(self, network, spike_times: List[Tuple[float, int]], 
                               duration: float, save_path: Optional[str] = None):
        """Create animated visualization of network activity"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Setup plot
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Network Activity Animation')
        ax.grid(True, alpha=0.3)
        
        # Plot neurons
        neuron_scatters = {}
        for neuron in network.neurons.values():
            color = self.colors.get(neuron.neuron_type.value, 'gray')
            scatter = ax.scatter(neuron.position[0], neuron.position[1], 
                               c=color, alpha=0.7, s=50)
            neuron_scatters[neuron.id] = scatter
        
        # Animation function
        def animate(frame):
            current_time = frame * 0.1  # 0.1ms per frame
            
            # Reset all neurons to normal state
            for scatter in neuron_scatters.values():
                scatter.set_sizes([50])
                scatter.set_alpha(0.7)
            
            # Highlight neurons that spiked recently
            for spike_time, neuron_id in spike_times:
                if abs(spike_time - current_time) < 2.0:  # 2ms window
                    if neuron_id in neuron_scatters:
                        neuron_scatters[neuron_id].set_sizes([100])
                        neuron_scatters[neuron_id].set_alpha(1.0)
            
            ax.set_title(f'Network Activity at {current_time:.1f}ms')
            return list(neuron_scatters.values())
        
        # Create animation
        frames = int(duration / 0.1)
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                     interval=50, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            logger.info(f"Network animation saved to {save_path}")
        
        plt.show()
        return anim
    
    def plot_learning_curves(self, network, save_path: Optional[str] = None):
        """Plot synaptic weight changes over time"""
        if not network.synapses:
            logger.warning("No synapses to plot learning curves for")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group synapses by weight ranges
        weights = [synapse.weight for synapse in network.synapses]
        
        ax.hist(weights, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Synaptic Weight')
        ax.set_ylabel('Number of Synapses')
        ax.set_title('Distribution of Synaptic Weights')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        ax.axvline(mean_weight, color='red', linestyle='--', 
                   label=f'Mean: {mean_weight:.2f}')
        ax.axvline(mean_weight + std_weight, color='orange', linestyle=':', 
                   label=f'+1σ: {mean_weight + std_weight:.2f}')
        ax.axvline(mean_weight - std_weight, color='orange', linestyle=':', 
                   label=f'-1σ: {mean_weight - std_weight:.2f}')
        
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves plot saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_report(self, network, simulation_results: Dict, 
                                  save_dir: Optional[str] = None):
        """Create a comprehensive visualization report"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Network topology (top left)
        ax1 = plt.subplot(3, 3, 1)
        for neuron in network.neurons.values():
            color = self.colors.get(neuron.neuron_type.value, 'gray')
            ax1.scatter(neuron.position[0], neuron.position[1], 
                       c=color, alpha=0.7, s=20)
        ax1.set_title('Network Topology')
        ax1.grid(True, alpha=0.3)
        
        # 2. Activity summary (top center)
        ax2 = plt.subplot(3, 3, 2)
        activity = simulation_results.get('network_activity', {})
        metrics = ['Total Neurons', 'Active Neurons', 'Recent Spikes']
        values = [activity.get('total_neurons', 0), 
                 activity.get('active_neurons', 0),
                 activity.get('recent_spikes', 0)]
        ax2.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax2.set_title('Network Activity Summary')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Region activity (top right)
        ax3 = plt.subplot(3, 3, 3)
        region_activities = simulation_results.get('region_activities', {})
        if region_activities:
            regions = list(region_activities.keys())
            firing_rates = [region_activities[r].get('firing_rate', 0) for r in regions]
            ax3.bar(regions, firing_rates, color='lightblue')
            ax3.set_title('Regional Firing Rates')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Spike raster (middle row, full width)
        ax4 = plt.subplot(3, 3, (4, 6))
        spike_data = simulation_results.get('spike_data', [])
        if spike_data:
            times, neuron_ids = zip(*spike_data)
            ax4.scatter(times, neuron_ids, alpha=0.6, s=1, c='black')
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('Neuron ID')
            ax4.set_title('Spike Raster Plot')
            ax4.grid(True, alpha=0.3)
        
        # 5. Synaptic weight distribution (bottom left)
        ax5 = plt.subplot(3, 3, 7)
        if network.synapses:
            weights = [s.weight for s in network.synapses]
            ax5.hist(weights, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax5.set_xlabel('Synaptic Weight')
            ax5.set_ylabel('Count')
            ax5.set_title('Synaptic Weight Distribution')
            ax5.grid(True, alpha=0.3)
        
        # 6. Connection matrix (bottom center)
        ax6 = plt.subplot(3, 3, 8)
        if network.synapses and network.neurons:
            max_id = max(network.neurons.keys())
            connection_matrix = np.zeros((max_id + 1, max_id + 1))
            
            for synapse in network.synapses:
                connection_matrix[synapse.pre_neuron_id][synapse.post_neuron_id] = synapse.weight
            
            im = ax6.imshow(connection_matrix, cmap='viridis', aspect='auto')
            ax6.set_title('Connection Matrix')
            plt.colorbar(im, ax=ax6)
        
        # 7. Time evolution (bottom right)
        ax7 = plt.subplot(3, 3, 9)
        if spike_data:
            # Count spikes per time bin
            time_bins = np.linspace(0, max(times) if times else 100, 20)
            spike_counts = np.zeros(len(time_bins)-1)
            
            for time in times:
                bin_idx = np.digitize(time, time_bins) - 1
                if 0 <= bin_idx < len(spike_counts):
                    spike_counts[bin_idx] += 1
            
            ax7.plot(time_bins[:-1], spike_counts, 'b-', linewidth=2)
            ax7.set_xlabel('Time (ms)')
            ax7.set_ylabel('Spike Count')
            ax7.set_title('Spike Rate Over Time')
            ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/comprehensive_report.png", dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive report saved to {save_dir}")
        
        plt.show()
