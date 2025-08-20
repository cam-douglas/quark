#!/usr/bin/env python3
"""
Terminal-based ASCII Brain Development Visualizer
Works in any terminal - no GUI or web server needed
"""

import os, sys
import time
import math
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class TerminalBrainVisualizer:
    """Terminal-based brain development visualizer"""
    
    def __init__(self):
        self.current_step = 0
        self.max_steps = 100
        self.is_running = False
        
        # Terminal dimensions
        self.terminal_width = 80
        self.terminal_height = 24
        
        # Data storage
        self.neural_tube_data = []
        self.neuron_counts = []
        self.synapse_counts = []
        self.region_data = []
        
        # Generate sample data
        self._generate_data()
        
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
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def draw_neural_tube(self, length, width, height):
        """Draw ASCII representation of neural tube"""
        tube_width = min(int(width * 2), 20)  # Scale for terminal
        
        # Draw top of tube
        top_line = " " * 10 + "╭" + "─" * tube_width + "╮"
        print(top_line)
        
        # Draw tube body
        for i in range(int(height * 2)):
            body_line = " " * 10 + "│" + " " * tube_width + "│"
            print(body_line)
        
        # Draw bottom of tube
        bottom_line = " " * 10 + "╰" + "─" * tube_width + "╯"
        print(bottom_line)
        
        # Add neurons as dots
        if self.current_step > 0:
            neuron_count = self.neuron_counts[self.current_step]
            num_neurons_to_show = min(neuron_count // 100, 10)
            
            for _ in range(num_neurons_to_show):
                x = 10 + 1 + int(tube_width * (0.2 + 0.6 * (_ / num_neurons_to_show)))
                y = 12 + int(height)
                if y < self.terminal_height - 5:
                    print(f"\033[{y};{x}H●")
    
    def draw_progress_bar(self):
        """Draw ASCII progress bar"""
        progress = (self.current_step / self.max_steps) * 50  # 50 chars wide
        filled = int(progress)
        empty = 50 - filled
        
        bar = "█" * filled + "░" * empty
        print(f"\n📊 Progress: [{bar}] {self.current_step}/{self.max_steps}")
        
        # Show development stage
        stage = self._get_development_stage(self.current_step / self.max_steps)
        print(f"🧬 Stage: {stage}")
    
    def draw_stats(self):
        """Draw statistics"""
        if self.current_step >= len(self.neuron_counts):
            return
            
        neuron_count = self.neuron_counts[self.current_step]
        synapse_count = self.synapse_counts[self.current_step]
        regions = self.region_data[self.current_step]
        
        print(f"\n🧠 Neurons: {neuron_count:,}")
        print(f"🔗 Synapses: {synapse_count:,}")
        
        # Regional breakdown
        print("\n📍 Regional Development:")
        for region, count in regions.items():
            bar_length = int((count / neuron_count) * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {region:12} [{bar}] {count:,}")
    
    def draw_ascii_art(self):
        """Draw ASCII art brain"""
        brain_art = [
            "    🧠 BRAIN DEVELOPMENT SIMULATION 🧠    ",
            "                                           ",
            "        Watch neurons grow and            ",
            "        connections form in real-time!    ",
            "                                           ",
            "    ╭─────────────────────────────────╮    ",
            "    │  Neural Tube Development        │    ",
            "    │  • Neurulation                  │    ",
            "    │  • Vesiculation                 │    ",
            "    │  • Cortical Layering            │    ",
            "    │  • Synaptogenesis               │    ",
            "    ╰─────────────────────────────────╯    "
        ]
        
        for line in brain_art:
            print(line)
    
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
    
    def update_display(self):
        """Update the entire display"""
        self.clear_screen()
        
        # Draw header
        self.draw_ascii_art()
        
        # Draw neural tube
        if self.current_step < len(self.neural_tube_data):
            length, width, height = self.neural_tube_data[self.current_step]
            print(f"\n🧬 Neural Tube (Step {self.current_step}):")
            self.draw_neural_tube(length, width, height)
        
        # Draw progress and stats
        self.draw_progress_bar()
        self.draw_stats()
        
        # Draw controls
        print(f"\n🎮 Controls: Press Ctrl+C to stop")
        print(f"⏱️  Speed: {200}ms per step")
    
    def start_simulation(self):
        """Start the brain development simulation"""
        self.is_running = True
        self.current_step = 0
        
        print("🧠 Starting Brain Development Simulation...")
        print("📊 Watch neurons grow and connections form in real-time!")
        print("⏹️  Press Ctrl+C to stop the simulation")
        time.sleep(2)
        
        try:
            while self.is_running and self.current_step < self.max_steps:
                self.update_display()
                time.sleep(0.2)  # 200ms per step
                self.current_step += 1
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Simulation stopped by user")
            self.is_running = False
    
    def run(self):
        """Run the terminal visualizer"""
        try:
            self.start_simulation()
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            print("\n👋 Thanks for watching the brain develop!")

def main():
    """Main function to run the terminal visualizer"""
    print("🧠 Terminal Brain Development Visualizer")
    print("=" * 50)
    
    # Check terminal size
    try:
        import shutil
        cols, rows = shutil.get_terminal_size()
        print(f"📱 Terminal size: {cols}x{rows}")
        if cols < 80 or rows < 24:
            print("⚠️  Terminal might be too small for optimal display")
    except:
        print("📱 Terminal size: Unknown")
    
    print("\n🚀 Starting in 3 seconds...")
    time.sleep(3)
    
    # Create and run visualizer
    visualizer = TerminalBrainVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
