#!/usr/bin/env python3
"""
Neural Network Evolution Game

An interactive game where players evolve and train neural networks in real-time.
Players can create networks, train them on tasks, and watch them evolve through generations.
"""

import random
import time
import math
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider, TextBox
import tkinter as tk
from tkinter import ttk, messagebox

class NeuralNetwork:
    """Simple neural network for the game."""
    
    def __init__(self, layers: List[int], name: str = None):
        self.layers = layers
        self.name = name or f"Network_{random.randint(1000, 9999)}"
        self.weights = []
        self.biases = []
        self.fitness = 0.0
        self.generation = 0
        self.mutations = 0
        self.training_history = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i + 1], layers[i]) * 0.1
            b = np.random.randn(layers[i + 1], 1) * 0.1
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        current = inputs.reshape(-1, 1)
        
        for w, b in zip(self.weights, self.biases):
            current = np.tanh(w @ current + b)
        
        return current.flatten()
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2):
        """Mutate the network weights and biases."""
        for i in range(len(self.weights)):
            # Mutate weights
            mask = np.random.random(self.weights[i].shape) < mutation_rate
            self.weights[i] += mask * np.random.randn(*self.weights[i].shape) * mutation_strength
            
            # Mutate biases
            mask = np.random.random(self.biases[i].shape) < mutation_rate
            self.biases[i] += mask * np.random.randn(*self.biases[i].shape) * mutation_strength
        
        self.mutations += 1
    
    def clone(self) -> 'NeuralNetwork':
        """Create a clone of this network."""
        clone = NeuralNetwork(self.layers, f"{self.name}_clone")
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        clone.fitness = self.fitness
        clone.generation = self.generation + 1
        clone.mutations = 0
        return clone

class Task:
    """Base class for training tasks."""
    
    def __init__(self, name: str, input_size: int, output_size: int):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
    
    def generate_training_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for the task."""
        raise NotImplementedError
    
    def evaluate(self, network: NeuralNetwork, num_samples: int = 100) -> float:
        """Evaluate network performance on the task."""
        raise NotImplementedError

class XORTask(Task):
    """XOR task for testing network learning."""
    
    def __init__(self):
        super().__init__("XOR", 2, 1)
    
    def generate_training_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate XOR training data."""
        inputs = np.random.randint(0, 2, (num_samples, 2))
        outputs = np.logical_xor(inputs[:, 0], inputs[:, 1]).astype(float)
        return inputs, outputs
    
    def evaluate(self, network: NeuralNetwork, num_samples: int = 100) -> float:
        """Evaluate XOR performance."""
        inputs, targets = self.generate_training_data(num_samples)
        predictions = np.array([network.forward(inp) for inp in inputs])
        
        # Calculate accuracy
        correct = np.sum(np.abs(predictions - targets) < 0.5)
        accuracy = correct / num_samples
        
        return accuracy

class CircleTask(Task):
    """Circle classification task."""
    
    def __init__(self):
        super().__init__("Circle Classification", 2, 1)
    
    def generate_training_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate circle classification data."""
        inputs = np.random.rand(num_samples, 2) * 4 - 2  # Points in [-2, 2] x [-2, 2]
        distances = np.sqrt(inputs[:, 0]**2 + inputs[:, 1]**2)
        outputs = (distances < 1.0).astype(float)  # Inside unit circle = 1, outside = 0
        return inputs, outputs
    
    def evaluate(self, network: NeuralNetwork, num_samples: int = 100) -> float:
        """Evaluate circle classification performance."""
        inputs, targets = self.generate_training_data(num_samples)
        predictions = np.array([network.forward(inp) for inp in inputs])
        
        # Calculate accuracy
        correct = np.sum(np.abs(predictions - targets) < 0.5)
        accuracy = correct / num_samples
        
        return accuracy

class NeuralEvolutionGame:
    """Main game class."""
    
    def __init__(self):
        self.population_size = 20
        self.population: List[NeuralNetwork] = []
        self.generation = 0
        self.best_fitness = 0.0
        self.best_network = None
        self.tasks = [
            XORTask(),
            CircleTask()
        ]
        self.current_task_idx = 0
        self.current_task = self.tasks[0]
        
        # Game state
        self.is_running = False
        self.auto_evolve = False
        self.evolution_speed = 1.0
        
        # Statistics
        self.generation_history = []
        self.fitness_history = []
        self.mutation_history = []
        
        # Setup GUI
        self.setup_gui()
        self.initialize_population()
    
    def initialize_population(self):
        """Initialize the initial population."""
        self.population = []
        for i in range(self.population_size):
            network = NeuralNetwork([self.current_task.input_size, 8, 6, self.current_task.output_size])
            network.name = f"Network_{i+1:03d}"
            self.population.append(network)
        
        self.generation = 0
        self.best_fitness = 0.0
        self.best_network = None
    
    def evaluate_population(self):
        """Evaluate all networks in the population."""
        for network in self.population:
            network.fitness = self.current_task.evaluate(network)
            network.training_history.append(network.fitness)
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update best network
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_network = self.population[0].clone()
        
        # Record statistics
        self.generation_history.append(self.generation)
        self.fitness_history.append(self.best_fitness)
        self.mutation_history.append(np.mean([n.mutations for n in self.population]))
    
    def evolve_population(self):
        """Evolve the population to the next generation."""
        # Keep top performers
        elite_size = max(1, self.population_size // 4)
        new_population = self.population[:elite_size].copy()
        
        # Generate offspring through mutation and crossover
        while len(new_population) < self.population_size:
            parent = random.choice(self.population[:elite_size * 2])
            offspring = parent.clone()
            offspring.mutate(mutation_rate=0.15, mutation_strength=0.3)
            new_population.append(offspring)
        
        self.population = new_population
        self.generation += 1
        
        # Update generation numbers
        for network in self.population:
            network.generation = self.generation
    
    def run_generation(self):
        """Run one generation of evolution."""
        if not self.is_running:
            return
        
        self.evaluate_population()
        self.evolve_population()
        self.update_display()
        
        # Check win condition
        if self.best_fitness >= 0.95:
            self.game_won()
    
    def game_won(self):
        """Handle game win condition."""
        self.is_running = False
        messagebox.showinfo("Congratulations!", 
                          f"You've evolved a successful network!\n"
                          f"Best Fitness: {self.best_fitness:.3f}\n"
                          f"Generations: {self.generation}")
    
    def setup_gui(self):
        """Setup the game GUI."""
        self.root = tk.Tk()
        self.root.title("Neural Network Evolution Game")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Game Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Evolution", command=self.toggle_evolution)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        reset_button = ttk.Button(control_frame, text="Reset", command=self.reset_game)
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # Task selector
        task_label = ttk.Label(control_frame, text="Task:")
        task_label.pack(side=tk.LEFT, padx=(20, 5))
        
        self.task_var = tk.StringVar(value=self.current_task.name)
        task_combo = ttk.Combobox(control_frame, textvariable=self.task_var, 
                                 values=[task.name for task in self.tasks], state="readonly")
        task_combo.pack(side=tk.LEFT, padx=5)
        task_combo.bind('<<ComboboxSelected>>', self.change_task)
        
        # Auto-evolve checkbox
        self.auto_var = tk.BooleanVar()
        auto_check = ttk.Checkbutton(control_frame, text="Auto-evolve", variable=self.auto_var)
        auto_check.pack(side=tk.LEFT, padx=20)
        
        # Speed slider
        speed_label = ttk.Label(control_frame, text="Speed:")
        speed_label.pack(side=tk.LEFT, padx=(20, 5))
        
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_slider = ttk.Scale(control_frame, from_=0.1, to=5.0, variable=self.speed_var, 
                                orient=tk.HORIZONTAL, length=100)
        speed_slider.pack(side=tk.LEFT, padx=5)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Stats labels
        self.gen_label = ttk.Label(stats_frame, text="Generation: 0")
        self.gen_label.pack(side=tk.LEFT, padx=10)
        
        self.fitness_label = ttk.Label(stats_frame, text="Best Fitness: 0.000")
        self.fitness_label.pack(side=tk.LEFT, padx=20)
        
        self.pop_label = ttk.Label(stats_frame, text="Population: 20")
        self.pop_label.pack(side=tk.LEFT, padx=20)
        
        self.task_label = ttk.Label(stats_frame, text=f"Task: {self.current_task.name}")
        self.task_label.pack(side=tk.LEFT, padx=20)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        # Embed matplotlib in tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Setup plots
        self.setup_plots()
        
        # Network visualization
        self.setup_network_viz()
    
    def setup_plots(self):
        """Setup the matplotlib plots."""
        # Fitness evolution plot
        self.ax1.set_title("Fitness Evolution")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Best Fitness")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(0, 1)
        
        # Population diversity plot
        self.ax2.set_title("Population Diversity")
        self.ax2.set_xlabel("Generation")
        self.ax2.set_ylabel("Average Mutations")
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
    
    def setup_network_viz(self):
        """Setup network visualization."""
        # This will be updated dynamically
        pass
    
    def update_display(self):
        """Update the display with current game state."""
        # Update labels
        self.gen_label.config(text=f"Generation: {self.generation}")
        self.fitness_label.config(text=f"Best Fitness: {self.best_fitness:.3f}")
        self.pop_label.config(text=f"Population: {len(self.population)}")
        
        # Update plots
        self.update_plots()
        
        # Update network visualization
        self.update_network_viz()
        
        # Schedule next update if auto-evolving
        if self.auto_var.get() and self.is_running:
            delay = int(1000 / self.speed_var.get())  # Convert to milliseconds
            self.root.after(delay, self.run_generation)
    
    def update_plots(self):
        """Update the matplotlib plots."""
        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Fitness evolution
        if len(self.generation_history) > 0:
            self.ax1.plot(self.generation_history, self.fitness_history, 'b-', linewidth=2)
            self.ax1.scatter(self.generation_history, self.fitness_history, c='blue', s=30)
        
        self.ax1.set_title("Fitness Evolution")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Best Fitness")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(0, 1)
        
        # Population diversity
        if len(self.generation_history) > 0:
            self.ax2.plot(self.generation_history, self.mutation_history, 'r-', linewidth=2)
            self.ax2.scatter(self.generation_history, self.mutation_history, c='red', s=30)
        
        self.ax2.set_title("Population Diversity")
        self.ax2.set_xlabel("Generation")
        self.ax2.set_ylabel("Average Mutations")
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_network_viz(self):
        """Update the network visualization."""
        if self.best_network is None:
            return
        
        # This would show the best network's structure
        # For now, we'll just update the task label
        self.task_label.config(text=f"Task: {self.current_task.name} (Best: {self.best_network.name})")
    
    def toggle_evolution(self):
        """Toggle evolution on/off."""
        if self.is_running:
            self.is_running = False
            self.start_button.config(text="Start Evolution")
        else:
            self.is_running = True
            self.start_button.config(text="Stop Evolution")
            if self.auto_var.get():
                self.run_generation()
            else:
                self.run_generation()
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.is_running = False
        self.start_button.config(text="Start Evolution")
        self.initialize_population()
        self.generation_history = []
        self.fitness_history = []
        self.mutation_history = []
        self.update_display()
    
    def change_task(self, event):
        """Change the current task."""
        task_name = self.task_var.get()
        for task in self.tasks:
            if task.name == task_name:
                self.current_task = task
                self.task_label.config(text=f"Task: {task.name}")
                self.reset_game()
                break
    
    def run(self):
        """Run the game."""
        self.root.mainloop()

def main():
    """Main function to run the game."""
    print("🧠 Starting Neural Network Evolution Game...")
    print("🎮 This is an interactive game where you evolve neural networks!")
    print("📊 Watch as your networks learn and improve through generations.")
    print("🎯 Try to achieve 95% accuracy to win!")
    
    game = NeuralEvolutionGame()
    game.run()

if __name__ == "__main__":
    main()
