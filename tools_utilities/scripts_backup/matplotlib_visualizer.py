#!/usr/bin/env python3
"""
Matplotlib-based Brain Development Visualizer
Opens in a native window - no web server needed
"""

import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import threading
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class BrainDevelopmentVisualizer:
    """Matplotlib-based brain development visualizer"""
    
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.suptitle('🧠 Brain Development Simulation', fontsize=20, fontweight='bold')
        
        # Simulation state
        self.current_step = 0
        self.max_steps = 100
        self.is_running = False
        
        # Data storage
        self.neural_tube_data = []
        self.neuron_counts = []
        self.synapse_counts = []
        self.region_data = []
        
        # Generate sample data
        self._generate_data()
        
        # Setup the plot
        self._setup_plot()
        
        # Animation
        self.ani = None
        
    def _generate_data(self):
        """Generate sample brain development data"""
        for step in range(self.max_steps):
            time_factor = step / self.max_steps
            
            # Neural tube dimensions
            length = 10 + 20 * time_factor
            width = 2 + 3 * time_factor
            height = 1 + 2 * time_factor
            
            # Neuron count (exponential growth)
            neuron_count = int(100 * (2 ** (3 * time_factor)))
            
            # Synapse count
            synapse_count = int(neuron_count * 10 * time_factor)
            
            # Regional development
            regions = {
                'forebrain': int(neuron_count * 0.4),
                'midbrain': int(neuron_count * 0.2),
                'hindbrain': int(neuron_count * 0.3),
                'spinal_cord': int(neuron_count * 0.1)
            }
            
            self.neural_tube_data.append((length, width, height))
            self.neuron_counts.append(neuron_count)
            self.synapse_counts.append(synapse_count)
            self.region_data.append(regions)
    
    def _setup_plot(self):
        """Setup the matplotlib plot"""
        self.ax.clear()
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main neural tube visualization
        ax_main = self.fig.add_subplot(gs[0:2, 0:2])
        ax_main.set_title('Neural Tube Development', fontweight='bold')
        ax_main.set_xlim(0, 30)
        ax_main.set_ylim(0, 10)
        ax_main.set_aspect('equal')
        
        # Progress bar
        ax_progress = self.fig.add_subplot(gs[2, 0:2])
        ax_progress.set_title('Development Progress', fontweight='bold')
        ax_progress.set_xlim(0, 100)
        ax_progress.set_ylim(0, 1)
        
        # Neuron count over time
        ax_neurons = self.fig.add_subplot(gs[0, 2])
        ax_neurons.set_title('Neuron Growth', fontweight='bold')
        ax_neurons.set_xlabel('Step')
        ax_neurons.set_ylabel('Count')
        
        # Synapse count over time
        ax_synapses = self.fig.add_subplot(gs[1, 2])
        ax_synapses.set_title('Synapse Formation', fontweight='bold')
        ax_synapses.set_xlabel('Step')
        ax_synapses.set_ylabel('Count')
        
        # Regional breakdown
        ax_regions = self.fig.add_subplot(gs[2, 2])
        ax_regions.set_title('Regional Development', fontweight='bold')
        
        # Store axes for later use
        self.axes = {
            'main': ax_main,
            'progress': ax_progress,
            'neurons': ax_neurons,
            'synapses': ax_synapses,
            'regions': ax_regions
        }
        
        # Initial plots
        self._update_plots()
        
    def _update_plots(self):
        """Update all plot elements"""
        if self.current_step >= len(self.neural_tube_data):
            return
            
        # Get current data
        length, width, height = self.neural_tube_data[self.current_step]
        neuron_count = self.neuron_counts[self.current_step]
        synapse_count = self.synapse_counts[self.current_step]
        regions = self.region_data[self.current_step]
        
        # Update main neural tube visualization
        ax_main = self.axes['main']
        ax_main.clear()
        ax_main.set_title(f'Neural Tube Development (Step {self.current_step})', fontweight='bold')
        ax_main.set_xlim(0, 30)
        ax_main.set_ylim(0, 10)
        ax_main.set_aspect('equal')
        
        # Draw neural tube
        tube = FancyBboxPatch(
            (0, 5 - height/2), length, height,
            boxstyle="round,pad=0.1",
            facecolor='lightblue',
            edgecolor='navy',
            linewidth=2
        )
        ax_main.add_patch(tube)
        
        # Add neurons as dots
        if neuron_count > 0:
            # Distribute neurons along the tube
            x_positions = np.linspace(1, length-1, min(neuron_count, 100))
            y_positions = 5 + (np.random.random(len(x_positions)) - 0.5) * height * 0.8
            
            for x, y in zip(x_positions, y_positions):
                neuron = Circle((x, y), 0.1, color='red', alpha=0.7)
                ax_main.add_patch(neuron)
        
        # Add synapses as small lines
        if synapse_count > 0:
            # Draw some representative synapses
            num_synapses_to_show = min(synapse_count // 100, 50)
            for _ in range(num_synapses_to_show):
                x1 = np.random.uniform(1, length-1)
                y1 = 5 + np.random.uniform(-height/2, height/2)
                x2 = x1 + np.random.uniform(-0.5, 0.5)
                y2 = y1 + np.random.uniform(-0.5, 0.5)
                ax_main.plot([x1, x2], [y1, y2], 'orange', linewidth=1, alpha=0.6)
        
        # Update progress bar
        ax_progress = self.axes['progress']
        ax_progress.clear()
        ax_progress.set_title(f'Development Progress: {self.current_step}/{self.max_steps}', fontweight='bold')
        ax_progress.set_xlim(0, 100)
        ax_progress.set_ylim(0, 1)
        
        progress = (self.current_step / self.max_steps) * 100
        progress_bar = Rectangle((0, 0), progress, 1, facecolor='green', alpha=0.7)
        ax_progress.add_patch(progress_bar)
        ax_progress.set_xticks([0, 25, 50, 75, 100])
        ax_progress.set_yticks([])
        
        # Update neuron count plot
        ax_neurons = self.axes['neurons']
        ax_neurons.clear()
        ax_neurons.set_title('Neuron Growth', fontweight='bold')
        ax_neurons.plot(range(self.current_step + 1), self.neuron_counts[:self.current_step + 1], 
                       'b-', linewidth=2, label=f'Current: {neuron_count:,}')
        ax_neurons.set_yscale('log')
        ax_neurons.legend()
        ax_neurons.grid(True, alpha=0.3)
        
        # Update synapse count plot
        ax_synapses = self.axes['synapses']
        ax_synapses.clear()
        ax_synapses.set_title('Synapse Formation', fontweight='bold')
        ax_synapses.plot(range(self.current_step + 1), self.synapse_counts[:self.current_step + 1], 
                        'r-', linewidth=2, label=f'Current: {synapse_count:,}')
        ax_synapses.set_yscale('log')
        ax_synapses.legend()
        ax_synapses.grid(True, alpha=0.3)
        
        # Update regional breakdown
        ax_regions = self.axes['regions']
        ax_regions.clear()
        ax_regions.set_title('Regional Development', fontweight='bold')
        
        if regions:
            region_names = list(regions.keys())
            region_counts = list(regions.values())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            bars = ax_regions.bar(region_names, region_counts, color=colors, alpha=0.7)
            ax_regions.set_ylabel('Neuron Count')
            
            # Add value labels on bars
            for bar, count in zip(bars, region_counts):
                height = bar.get_height()
                ax_regions.text(bar.get_x() + bar.get_width()/2., height,
                              f'{count:,}', ha='center', va='bottom')
        
        # Add development stage text
        stage = self._get_development_stage(self.current_step / self.max_steps)
        self.fig.suptitle(f'🧠 Brain Development Simulation - {stage}', fontsize=20, fontweight='bold')
        
        plt.tight_layout()
    
    def _get_development_stage(self, time_factor: float) -> str:
        """Get human-readable development stage"""
        if time_factor < 0.1:
            return "Neural Plate Formation"
        elif time_factor < 0.2:
            return "Neural Tube Closure"
        elif time_factor < 0.4:
            return "Primary Vesicle Formation"
        elif time_factor < 0.6:
            return "Secondary Vesicle Formation"
        elif time_factor < 0.8:
            return "Cortical Layering"
        else:
            return "Synaptogenesis & Circuit Formation"
    
    def start_simulation(self):
        """Start the brain development simulation"""
        self.is_running = True
        self.current_step = 0
        
        def run_simulation():
            while self.is_running and self.current_step < self.max_steps:
                time.sleep(0.2)  # 200ms per step
                self.current_step += 1
        
        thread = threading.Thread(target=run_simulation)
        thread.daemon = True
        thread.start()
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, 
            lambda frame: self._update_plots(), 
            interval=200,  # Update every 200ms
            blit=False
        )
        
        plt.show()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        if self.ani:
            self.ani.event_source.stop()
    
    def run_interactive(self):
        """Run the simulation interactively"""
        print("🧠 Starting Brain Development Simulation...")
        print("📊 Watch neurons grow and connections form in real-time!")
        print("⏹️  Close the window to stop the simulation")
        
        self.start_simulation()

def main():
    """Main function to run the visualizer"""
    try:
        # Check if matplotlib is available
        import matplotlib
        print("✅ Matplotlib visualization ready!")
        
        # Create and run visualizer
        visualizer = BrainDevelopmentVisualizer()
        visualizer.run_interactive()
        
    except ImportError as e:
        print(f"❌ Matplotlib not available: {e}")
        print("Install with: pip install matplotlib")
    except Exception as e:
        print(f"❌ Error running visualizer: {e}")

if __name__ == "__main__":
    main()
