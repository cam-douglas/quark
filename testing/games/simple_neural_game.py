#!/usr/bin/env python3
"""
Simple Neural Network Evolution Game

A web-based interactive game where players evolve neural networks in real-time.
Uses Flask and simple HTML/JavaScript for visualization.
"""

import random
import time
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from flask import Flask, render_template_string, jsonify, request
import threading

class SimpleNeuralNetwork:
    """Simple neural network for the game."""
    
    def __init__(self, layers: List[int], name: str = None):
        self.layers = layers
        self.name = name or f"Network_{random.randint(1000, 9999)}"
        self.weights = []
        self.biases = []
        self.fitness = 0.0
        self.generation = 0
        self.mutations = 0
        
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
    
    def clone(self) -> 'SimpleNeuralNetwork':
        """Create a clone of this network."""
        clone = SimpleNeuralNetwork(self.layers, f"{self.name}_clone")
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        clone.fitness = self.fitness
        clone.generation = self.generation + 1
        clone.mutations = 0
        return clone

class XORTask:
    """XOR task for testing network learning."""
    
    def __init__(self):
        self.name = "XOR"
        self.input_size = 2
        self.output_size = 1
    
    def evaluate(self, network: SimpleNeuralNetwork, num_samples: int = 100) -> float:
        """Evaluate XOR performance."""
        inputs = np.random.randint(0, 2, (num_samples, 2))
        outputs = np.logical_xor(inputs[:, 0], inputs[:, 1]).astype(float)
        
        predictions = np.array([network.forward(inp) for inp in inputs])
        
        # Calculate accuracy
        correct = np.sum(np.abs(predictions - outputs) < 0.5)
        accuracy = correct / num_samples
        
        return accuracy

class CircleTask:
    """Circle classification task."""
    
    def __init__(self):
        self.name = "Circle Classification"
        self.input_size = 2
        self.output_size = 1
    
    def evaluate(self, network: SimpleNeuralNetwork, num_samples: int = 100) -> float:
        """Evaluate circle classification performance."""
        inputs = np.random.rand(num_samples, 2) * 4 - 2  # Points in [-2, 2] x [-2, 2]
        distances = np.sqrt(inputs[:, 0]**2 + inputs[:, 1]**2)
        outputs = (distances < 1.0).astype(float)  # Inside unit circle = 1, outside = 0
        
        predictions = np.array([network.forward(inp) for inp in inputs])
        
        # Calculate accuracy
        correct = np.sum(np.abs(predictions - outputs) < 0.5)
        accuracy = correct / num_samples
        
        return accuracy

class SimpleNeuralGame:
    """Main game class."""
    
    def __init__(self):
        self.population_size = 20
        self.population: List[SimpleNeuralNetwork] = []
        self.generation = 0
        self.best_fitness = 0.0
        self.best_network = None
        self.tasks = [XORTask(), CircleTask()]
        self.current_task_idx = 0
        self.current_task = self.tasks[0]
        
        # Game state
        self.is_running = False
        self.auto_evolve = False
        
        # Statistics
        self.generation_history = []
        self.fitness_history = []
        self.mutation_history = []
        
        # Initialize population
        self.initialize_population()
        
        # Setup Flask app
        self.app = Flask(__name__)
        self.setup_routes()
    
    def initialize_population(self):
        """Initialize the initial population."""
        self.population = []
        for i in range(self.population_size):
            network = SimpleNeuralNetwork([self.current_task.input_size, 8, 6, self.current_task.output_size])
            network.name = f"Network_{i+1:03d}"
            self.population.append(network)
        
        self.generation = 0
        self.best_fitness = 0.0
        self.best_network = None
    
    def evaluate_population(self):
        """Evaluate all networks in the population."""
        for network in self.population:
            network.fitness = self.current_task.evaluate(network)
        
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
        
        # Check win condition
        if self.best_fitness >= 0.95:
            self.is_running = False
            return {"won": True, "fitness": self.best_fitness, "generations": self.generation}
        
        return {"won": False, "fitness": self.best_fitness, "generations": self.generation}
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return self.get_html_template()
        
        @self.app.route('/api/start')
        def start_evolution():
            self.is_running = True
            return jsonify({"status": "started"})
        
        @self.app.route('/api/stop')
        def stop_evolution():
            self.is_running = False
            return jsonify({"status": "stopped"})
        
        @self.app.route('/api/reset')
        def reset_game():
            self.is_running = False
            self.initialize_population()
            self.generation_history = []
            self.fitness_history = []
            self.mutation_history = []
            return jsonify({"status": "reset"})
        
        @self.app.route('/api/evolve')
        def evolve():
            result = self.run_generation()
            return jsonify({
                "generation": self.generation,
                "best_fitness": self.best_fitness,
                "population_size": len(self.population),
                "current_task": self.current_task.name,
                "won": result.get("won", False),
                "generation_history": self.generation_history,
                "fitness_history": self.fitness_history,
                "mutation_history": self.mutation_history
            })
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                "is_running": self.is_running,
                "generation": self.generation,
                "best_fitness": self.best_fitness,
                "population_size": len(self.population),
                "current_task": self.current_task.name
            })
        
        @self.app.route('/api/change_task/<task_name>')
        def change_task(task_name):
            for task in self.tasks:
                if task.name == task_name:
                    self.current_task = task
                    self.reset_game()
                    return jsonify({"status": "changed", "task": task.name})
            return jsonify({"status": "error", "message": "Task not found"})
    
    def get_html_template(self):
        """Get the HTML template for the game."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Evolution Game</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .controls {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .control-group {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }
        button {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        select {
            padding: 10px 15px;
            border-radius: 10px;
            border: none;
            background: rgba(255,255,255,0.9);
            font-size: 16px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .visualization {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .chart-container {
            height: 400px;
            margin-top: 20px;
        }
        .auto-evolve {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .auto-evolve input[type="checkbox"] {
            transform: scale(1.5);
        }
        .win-message {
            background: linear-gradient(45deg, #00b894, #00cec9);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin: 20px 0;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Neural Network Evolution Game</h1>
            <p>Evolve neural networks and watch them learn in real-time!</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <button id="startBtn">Start Evolution</button>
                <button id="resetBtn">Reset</button>
                <button id="evolveBtn">Evolve Once</button>
            </div>
            
            <div class="control-group">
                <label>Task:</label>
                <select id="taskSelect">
                    <option value="XOR">XOR Problem</option>
                    <option value="Circle Classification">Circle Classification</option>
                </select>
                
                <div class="auto-evolve">
                    <input type="checkbox" id="autoEvolve">
                    <label for="autoEvolve">Auto-evolve</label>
                </div>
            </div>
        </div>
        
        <div class="win-message" id="winMessage">
            <h2>🎉 Congratulations! 🎉</h2>
            <p>You've evolved a successful neural network!</p>
            <p id="winDetails"></p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Generation</h3>
                <div class="stat-value" id="generation">0</div>
            </div>
            <div class="stat-card">
                <h3>Best Fitness</h3>
                <div class="stat-value" id="bestFitness">0.000</div>
            </div>
            <div class="stat-card">
                <h3>Population</h3>
                <div class="stat-value" id="population">20</div>
            </div>
            <div class="stat-card">
                <h3>Current Task</h3>
                <div class="stat-value" id="currentTask">XOR</div>
            </div>
        </div>
        
        <div class="visualization">
            <h3>Evolution Progress</h3>
            <div class="chart-container">
                <canvas id="fitnessChart"></canvas>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        let gameState = {
            isRunning: false,
            generation: 0,
            bestFitness: 0,
            population: 20,
            currentTask: 'XOR'
        };
        
        let fitnessChart;
        let autoEvolveInterval;
        
        // Initialize chart
        function initChart() {
            const ctx = document.getElementById('fitnessChart').getContext('2d');
            fitnessChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Best Fitness',
                        data: [],
                        borderColor: '#00b894',
                        backgroundColor: 'rgba(0, 184, 148, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            },
                            ticks: {
                                color: 'white'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            },
                            ticks: {
                                color: 'white'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: 'white'
                            }
                        }
                    }
                }
            });
        }
        
        // Update display
        function updateDisplay(data) {
            document.getElementById('generation').textContent = data.generation;
            document.getElementById('bestFitness').textContent = data.best_fitness.toFixed(3);
            document.getElementById('population').textContent = data.population_size;
            document.getElementById('currentTask').textContent = data.current_task;
            
            // Update chart
            if (data.generation_history && data.fitness_history) {
                fitnessChart.data.labels = data.generation_history;
                fitnessChart.data.datasets[0].data = data.fitness_history;
                fitnessChart.update();
            }
            
            // Check for win
            if (data.won) {
                showWinMessage(data);
            }
        }
        
        // Show win message
        function showWinMessage(data) {
            document.getElementById('winDetails').innerHTML = 
                `Best Fitness: ${data.fitness.toFixed(3)}<br>Generations: ${data.generations}`;
            document.getElementById('winMessage').style.display = 'block';
        }
        
        // API calls
        async function apiCall(endpoint) {
            try {
                const response = await fetch(`/api/${endpoint}`);
                return await response.json();
            } catch (error) {
                console.error('API call failed:', error);
                return null;
            }
        }
        
        // Start evolution
        document.getElementById('startBtn').addEventListener('click', async () => {
            if (!gameState.isRunning) {
                await apiCall('start');
                gameState.isRunning = true;
                document.getElementById('startBtn').textContent = 'Stop Evolution';
                document.getElementById('evolveBtn').disabled = true;
                
                if (document.getElementById('autoEvolve').checked) {
                    startAutoEvolve();
                }
            } else {
                await apiCall('stop');
                gameState.isRunning = false;
                document.getElementById('startBtn').textContent = 'Start Evolution';
                document.getElementById('evolveBtn').disabled = false;
                stopAutoEvolve();
            }
        });
        
        // Reset game
        document.getElementById('resetBtn').addEventListener('click', async () => {
            await apiCall('reset');
            gameState.isRunning = false;
            document.getElementById('startBtn').textContent = 'Start Evolution';
            document.getElementById('evolveBtn').disabled = false;
            stopAutoEvolve();
            
            // Reset chart
            fitnessChart.data.labels = [];
            fitnessChart.data.datasets[0].data = [];
            fitnessChart.update();
            
            // Hide win message
            document.getElementById('winMessage').style.display = 'none';
            
            // Update display
            const status = await apiCall('status');
            if (status) updateDisplay(status);
        });
        
        // Evolve once
        document.getElementById('evolveBtn').addEventListener('click', async () => {
            const result = await apiCall('evolve');
            if (result) updateDisplay(result);
        });
        
        // Change task
        document.getElementById('taskSelect').addEventListener('change', async (e) => {
            const taskName = e.target.value;
            await apiCall(`change_task/${taskName}`);
            const status = await apiCall('status');
            if (status) updateDisplay(status);
        });
        
        // Auto-evolve
        document.getElementById('autoEvolve').addEventListener('change', (e) => {
            if (e.target.checked && gameState.isRunning) {
                startAutoEvolve();
            } else {
                stopAutoEvolve();
            }
        });
        
        function startAutoEvolve() {
            autoEvolveInterval = setInterval(async () => {
                if (gameState.isRunning) {
                    const result = await apiCall('evolve');
                    if (result) updateDisplay(result);
                }
            }, 1000);
        }
        
        function stopAutoEvolve() {
            if (autoEvolveInterval) {
                clearInterval(autoEvolveInterval);
                autoEvolveInterval = null;
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initChart();
            apiCall('status').then(status => {
                if (status) updateDisplay(status);
            });
        });
    </script>
</body>
</html>
        """
    
    def run(self, host='0.0.0.0', port=8001):
        """Run the game server."""
        print("🧠 Starting Simple Neural Network Evolution Game...")
        print("🎮 This is an interactive game where you evolve neural networks!")
        print("📊 Watch as your networks learn and improve through generations.")
        print("🎯 Try to achieve 95% accuracy to win!")
        print(f"🌐 Open your browser and go to: http://localhost:{port}")
        
        self.app.run(host=host, port=port, debug=False)

def main():
    """Main function to run the game."""
    game = SimpleNeuralGame()
    game.run()

if __name__ == "__main__":
    main()
