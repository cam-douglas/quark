#!/usr/bin/env python3
"""
Simple Web Server for Brain Development Visualization
Provides a web interface to view brain development simulation results
"""

import os, sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, render_template, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available. Install with: pip install numpy")

class BrainDevelopmentVisualizer:
    """Simple brain development visualizer"""
    
    def __init__(self):
        self.simulation_data = {}
        self.current_step = 0
        self.max_steps = 100
        self.is_running = False
        
    def generate_sample_data(self):
        """Generate sample brain development data"""
        if not NUMPY_AVAILABLE:
            return
            
        # Generate sample neural tube data
        for step in range(self.max_steps):
            # Simulate neural tube growth
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
            
            self.simulation_data[step] = {
                'step': step,
                'time': time_factor,
                'neural_tube': {
                    'length': length,
                    'width': width,
                    'height': height
                },
                'neurons': {
                    'total': neuron_count,
                    'regions': regions
                },
                'synapses': synapse_count,
                'development_stage': self._get_development_stage(time_factor)
            }
    
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
                time.sleep(0.1)  # 100ms per step
                self.current_step += 1
        
        thread = threading.Thread(target=run_simulation)
        thread.daemon = True
        thread.start()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get current simulation data"""
        if self.current_step in self.simulation_data:
            return self.simulation_data[self.current_step]
        return {}
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all simulation data"""
        return {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'is_running': self.is_running,
            'data': self.simulation_data
        }

# Create Flask app
if FLASK_AVAILABLE:
    app = Flask(__name__)
    
    # Initialize visualizer
    visualizer = BrainDevelopmentVisualizer()
    visualizer.generate_sample_data()
    
    @app.route('/')
    def index():
        """Main visualization page"""
        return render_template('brain_development.html')
    
    @app.route('/api/status')
    def get_status():
        """Get current simulation status"""
        return jsonify(visualizer.get_all_data())
    
    @app.route('/api/start', methods=['POST'])
    def start_simulation():
        """Start the simulation"""
        visualizer.start_simulation()
        return jsonify({'status': 'started'})
    
    @app.route('/api/stop', methods=['POST'])
    def stop_simulation():
        """Stop the simulation"""
        visualizer.stop_simulation()
        return jsonify({'status': 'stopped'})
    
    @app.route('/api/step/<int:step>')
    def get_step_data(step):
        """Get data for a specific step"""
        if step in visualizer.simulation_data:
            return jsonify(visualizer.simulation_data[step])
        return jsonify({'error': 'Step not found'}), 404

def create_html_template():
    """Create the HTML template for the visualization"""
    template_dir = Path(__file__).parent.parent.parent / 'templates'
    template_dir.mkdir(exist_ok=True)
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain Development Simulation</title>
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
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 30px;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .start-btn {
            background: #4CAF50;
            color: white;
        }
        .stop-btn {
            background: #f44336;
            color: white;
        }
        .reset-btn {
            background: #2196F3;
            color: white;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .simulation-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .info-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
        }
        .chart-container {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            margin-top: 20px;
        }
        .neuron-count {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
            margin: 20px 0;
        }
        .region-breakdown {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .region-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Brain Development Simulation</h1>
            <p>Watch neurons grow and connections form in real-time</p>
        </div>
        
        <div class="controls">
            <button class="start-btn" onclick="startSimulation()">▶️ Start Simulation</button>
            <button class="stop-btn" onclick="stopSimulation()">⏹️ Stop</button>
            <button class="reset-btn" onclick="resetSimulation()">🔄 Reset</button>
        </div>
        
        <div class="simulation-info">
            <div class="info-card">
                <h3>📊 Development Progress</h3>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressBar"></div>
                </div>
                <p id="progressText">Step 0 / 100</p>
                <p id="stageText">Neural Plate Formation</p>
            </div>
            
            <div class="info-card">
                <h3>🧬 Neural Tube</h3>
                <div id="neuralTubeInfo">
                    <p>Length: <span id="tubeLength">10</span> units</p>
                    <p>Width: <span id="tubeWidth">2</span> units</p>
                    <p>Height: <span id="tubeHeight">1</span> units</p>
                </div>
            </div>
            
            <div class="info-card">
                <h3>🔗 Synapses</h3>
                <div class="neuron-count" id="synapseCount">0</div>
                <p>Total connections forming</p>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>🧠 Regional Development</h3>
            <div class="region-breakdown" id="regionBreakdown">
                <!-- Region data will be populated here -->
            </div>
        </div>
    </div>
    
    <script>
        let simulationInterval;
        let currentData = {};
        
        async function updateDisplay() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                if (data.data && data.current_step in data.data) {
                    currentData = data.data[data.current_step];
                    updateUI(currentData);
                }
            } catch (error) {
                console.error('Error updating display:', error);
            }
        }
        
        function updateUI(data) {
            // Update progress
            const progress = (data.step / 100) * 100;
            document.getElementById('progressBar').style.width = progress + '%';
            document.getElementById('progressText').textContent = `Step ${data.step} / 100`;
            document.getElementById('stageText').textContent = data.development_stage;
            
            // Update neural tube
            document.getElementById('tubeLength').textContent = data.neural_tube.length.toFixed(1);
            document.getElementById('tubeWidth').textContent = data.neural_tube.width.toFixed(1);
            document.getElementById('tubeHeight').textContent = data.neural_tube.height.toFixed(1);
            
            // Update synapse count
            document.getElementById('synapseCount').textContent = data.synapses.toLocaleString();
            
            // Update regional breakdown
            const regionBreakdown = document.getElementById('regionBreakdown');
            regionBreakdown.innerHTML = '';
            
            Object.entries(data.neurons.regions).forEach(([region, count]) => {
                const regionItem = document.createElement('div');
                regionItem.className = 'region-item';
                regionItem.innerHTML = `
                    <span>${region.charAt(0).toUpperCase() + region.slice(1)}</span>
                    <span>${count.toLocaleString()}</span>
                `;
                regionBreakdown.appendChild(regionItem);
            });
        }
        
        async function startSimulation() {
            try {
                await fetch('/api/start', { method: 'POST' });
                simulationInterval = setInterval(updateDisplay, 100);
            } catch (error) {
                console.error('Error starting simulation:', error);
            }
        }
        
        async function stopSimulation() {
            try {
                await fetch('/api/stop', { method: 'POST' });
                if (simulationInterval) {
                    clearInterval(simulationInterval);
                }
            } catch (error) {
                console.error('Error stopping simulation:', error);
            }
        }
        
        function resetSimulation() {
            stopSimulation();
            location.reload();
        }
        
        // Initial load
        updateDisplay();
    </script>
</body>
</html>"""
    
    template_file = template_dir / 'brain_development.html'
    template_file.write_text(html_content)
    print(f"HTML template created at: {template_file}")

def main():
    """Main function to run the visualization server"""
    if not FLASK_AVAILABLE:
        print("Flask not available. Installing...")
        os.system("pip install flask")
        try:
            from flask import Flask, render_template, jsonify, request
            FLASK_AVAILABLE = True
        except ImportError:
            print("Failed to install Flask. Please install manually: pip install flask")
            return
    
    # Create HTML template
    create_html_template()
    
    print("🧠 Starting Brain Development Visualization Server...")
    print("🌐 Open your browser to: http://localhost:5000")
    print("📊 Watch neurons grow and connections form in real-time!")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()
