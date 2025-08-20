#!/usr/bin/env python3
"""
Brain Simulation Visual Server

Provides a real-time web interface for monitoring and visualizing brain simulation data.
Features:
- Real-time data streaming
- Interactive 3D visualizations
- Network topology displays
- Activity heatmaps
- Performance metrics
"""

import os, sys
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    from io import BytesIO
    import base64
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install flask flask-socketio matplotlib")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'brain_simulation_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global simulation state
simulation_state = {
    'running': False,
    'mode': 'hybrid',
    'current_time': 0.0,
    'total_time': 1000.0,
    'brain_regions': [],
    'neural_data': {},
    'physical_data': {},
    'metrics': {},
    'last_update': datetime.now()
}

# Mock data generator for demonstration
class MockDataGenerator:
    def __init__(self):
        self.time = 0.0
        self.brain_regions = [
            'visual_cortex', 'motor_cortex', 'sensory_cortex', 
            'prefrontal_cortex', 'hippocampus'
        ]
        self.neuron_counts = {
            'visual_cortex': 300,
            'motor_cortex': 250,
            'sensory_cortex': 280,
            'prefrontal_cortex': 400,
            'hippocampus': 200
        }
    
    def generate_neural_data(self) -> Dict:
        """Generate realistic neural simulation data"""
        data = {
            'timestamp': self.time,
            'regions': {},
            'network_activity': {
                'total_spikes': int(np.random.poisson(500)),
                'active_neurons': int(np.random.normal(800, 100)),
                'average_firing_rate': np.random.normal(15.0, 3.0)
            }
        }
        
        for region in self.brain_regions:
            # Generate region-specific activity
            base_rate = np.random.normal(10.0, 2.0)
            time_modulation = 1.0 + 0.3 * np.sin(self.time * 0.01)
            
            data['regions'][region] = {
                'firing_rate': base_rate * time_modulation,
                'active_neurons': int(np.random.normal(0.7, 0.1) * self.neuron_counts[region]),
                'spike_count': int(np.random.poisson(base_rate * 0.1)),
                'synaptic_activity': np.random.normal(0.5, 0.1),
                'plasticity_index': np.random.normal(0.3, 0.05)
            }
        
        return data
    
    def generate_physical_data(self) -> Dict:
        """Generate physical simulation data"""
        data = {
            'timestamp': self.time,
            'regions': {},
            'growth_metrics': {
                'overall_growth_rate': np.random.normal(0.01, 0.002),
                'mechanical_stress': np.random.normal(0.5, 0.1),
                'tissue_deformation': np.random.normal(0.3, 0.05)
            }
        }
        
        for region in self.brain_regions:
            # Generate physical properties
            growth_rate = np.random.normal(0.01, 0.003)
            stress = np.random.normal(0.5, 0.1)
            
            data['regions'][region] = {
                'size': 1.0 + growth_rate * self.time,
                'growth_rate': growth_rate,
                'mechanical_stress': stress,
                'position': [
                    np.random.normal(0, 10),
                    np.random.normal(0, 10),
                    np.random.normal(0, 5)
                ]
            }
        
        return data
    
    def update(self, dt: float):
        """Update simulation time and generate new data"""
        self.time += dt
        return {
            'neural': self.generate_neural_data(),
            'physical': self.generate_physical_data()
        }

# Initialize data generator
data_generator = MockDataGenerator()

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current simulation status"""
    return jsonify(simulation_state)

@app.route('/api/start_simulation')
def start_simulation():
    """Start the simulation"""
    global simulation_state
    simulation_state['running'] = True
    simulation_state['current_time'] = 0.0
    simulation_state['last_update'] = datetime.now()
    
    # Start simulation thread
    threading.Thread(target=run_simulation, daemon=True).start()
    
    return jsonify({'status': 'started'})

@app.route('/api/stop_simulation')
def stop_simulation():
    """Stop the simulation"""
    global simulation_state
    simulation_state['running'] = False
    return jsonify({'status': 'stopped'})

@app.route('/api/reset_simulation')
def reset_simulation():
    """Reset the simulation"""
    global simulation_state
    simulation_state['running'] = False
    simulation_state['current_time'] = 0.0
    simulation_state['neural_data'] = {}
    simulation_state['physical_data'] = {}
    simulation_state['metrics'] = {}
    return jsonify({'status': 'reset'})

@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    """Update simulation settings"""
    data = request.get_json()
    global simulation_state
    
    if 'mode' in data:
        simulation_state['mode'] = data['mode']
    if 'total_time' in data:
        simulation_state['total_time'] = float(data['total_time'])
    
    return jsonify({'status': 'updated'})

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('status_update', simulation_state)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_data')
def handle_data_request():
    """Handle data request from client"""
    emit('data_update', {
        'neural': simulation_state.get('neural_data', {}),
        'physical': simulation_state.get('physical_data', {}),
        'metrics': simulation_state.get('metrics', {})
    })

# Simulation functions
def run_simulation():
    """Run the simulation loop"""
    global simulation_state, data_generator
    
    dt = 0.1  # Time step
    update_interval = 0.1  # Update interval for clients
    
    while simulation_state['running']:
        try:
            # Generate new data
            new_data = data_generator.update(dt)
            
            # Update simulation state
            simulation_state['neural_data'] = new_data['neural']
            simulation_state['physical_data'] = new_data['physical']
            simulation_state['current_time'] = data_generator.time
            simulation_state['last_update'] = datetime.now()
            
            # Calculate metrics
            simulation_state['metrics'] = calculate_metrics(new_data)
            
            # Emit updates to all connected clients
            socketio.emit('simulation_update', {
                'timestamp': simulation_state['current_time'],
                'neural_data': new_data['neural'],
                'physical_data': new_data['physical'],
                'metrics': simulation_state['metrics']
            })
            
            # Check if simulation is complete
            if simulation_state['current_time'] >= simulation_state['total_time']:
                simulation_state['running'] = False
                socketio.emit('simulation_complete', {
                    'final_time': simulation_state['current_time'],
                    'total_duration': simulation_state['total_time']
                })
                break
            
            time.sleep(update_interval)
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            break
    
    logger.info("Simulation loop ended")

def calculate_metrics(data: Dict) -> Dict:
    """Calculate performance and analysis metrics"""
    neural_data = data['neural']
    physical_data = data['physical']
    
    metrics = {
        'performance': {
            'total_neurons': sum(neural_data['regions'][r]['active_neurons'] 
                               for r in neural_data['regions']),
            'overall_firing_rate': neural_data['network_activity']['average_firing_rate'],
            'spike_efficiency': neural_data['network_activity']['total_spikes'] / 
                              max(neural_data['network_activity']['active_neurons'], 1)
        },
        'development': {
            'average_growth_rate': np.mean([physical_data['regions'][r]['growth_rate'] 
                                          for r in physical_data['regions']]),
            'stress_distribution': np.std([physical_data['regions'][r]['mechanical_stress'] 
                                        for r in physical_data['regions']]),
            'development_progress': (data_generator.time / simulation_state['total_time']) * 100
        },
        'network_health': {
            'connectivity_index': np.mean([neural_data['regions'][r]['synaptic_activity'] 
                                        for r in neural_data['regions']]),
            'plasticity_level': np.mean([neural_data['regions'][r]['plasticity_index'] 
                                      for r in neural_data['regions']]),
            'activity_balance': np.std([neural_data['regions'][r]['firing_rate'] 
                                     for r in neural_data['regions']])
        }
    }
    
    return metrics

# Visualization routes
@app.route('/api/plot/network_topology')
def plot_network_topology():
    """Generate network topology plot"""
    try:
        # Create network topology visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get current neural data
        neural_data = simulation_state.get('neural_data', {})
        if not neural_data:
            return jsonify({'error': 'No neural data available'})
        
        regions = list(neural_data.get('regions', {}).keys())
        if not regions:
            return jsonify({'error': 'No region data available'})
        
        # Create network graph
        positions = {}
        for i, region in enumerate(regions):
            angle = 2 * np.pi * i / len(regions)
            positions[region] = [np.cos(angle) * 3, np.sin(angle) * 3]
        
        # Plot regions
        for region, pos in positions.items():
            region_data = neural_data['regions'].get(region, {})
            size = region_data.get('active_neurons', 100) / 100
            color = plt.cm.viridis(region_data.get('firing_rate', 0) / 20.0)
            
            ax.scatter(pos[0], pos[1], s=size*500, c=[color], alpha=0.7, 
                      label=f"{region}\n({region_data.get('active_neurons', 0)} neurons)")
        
        # Add connections (simplified)
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions[i+1:], i+1):
                pos1 = positions[region1]
                pos2 = positions[region2]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                       'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_title('Brain Network Topology')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({'plot': plot_data})
        
    except Exception as e:
        logger.error(f"Error generating network topology plot: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/plot/activity_heatmap')
def plot_activity_heatmap():
    """Generate activity heatmap"""
    try:
        neural_data = simulation_state.get('neural_data', {})
        if not neural_data:
            return jsonify({'error': 'No neural data available'})
        
        regions = list(neural_data.get('regions', {}).keys())
        if not regions:
            return jsonify({'error': 'No region data available'})
        
        # Extract metrics for heatmap
        metrics = ['firing_rate', 'active_neurons', 'spike_count', 'synaptic_activity']
        data_matrix = []
        
        for region in regions:
            region_data = neural_data['regions'].get(region, {})
            row = [
                region_data.get('firing_rate', 0),
                region_data.get('active_neurons', 0),
                region_data.get('spike_count', 0),
                region_data.get('synaptic_activity', 0)
            ]
            data_matrix.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(data_matrix, cmap='viridis', aspect='auto')
        
        # Customize heatmap
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions)
        ax.set_title('Brain Region Activity Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activity Level')
        
        # Add text annotations
        for i in range(len(regions)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i][j]:.1f}',
                             ha="center", va="center", color="white" if data_matrix[i][j] > 0.5 else "black")
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({'plot': plot_data})
        
    except Exception as e:
        logger.error(f"Error generating activity heatmap: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/plot/development_timeline')
def plot_development_timeline():
    """Generate development timeline plot"""
    try:
        # This would normally use historical data
        # For now, generate sample timeline
        time_points = np.linspace(0, simulation_state['current_time'], 50)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Neural development
        neural_activity = [np.random.normal(10, 2) * (1 + 0.3 * np.sin(t * 0.01)) for t in time_points]
        ax1.plot(time_points, neural_activity, 'b-', linewidth=2, label='Neural Activity')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Activity Level')
        ax1.set_title('Neural Development Timeline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Physical development
        physical_growth = [1.0 + 0.01 * t for t in time_points]
        ax2.plot(time_points, physical_growth, 'r-', linewidth=2, label='Physical Growth')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Growth Factor')
        ax2.set_title('Physical Development Timeline')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({'plot': plot_data})
        
    except Exception as e:
        logger.error(f"Error generating development timeline: {e}")
        return jsonify({'error': str(e)})

# Static file serving
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Brain Simulation Visual Server...")
    logger.info("Open http://localhost:5000 in your browser")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
