#!/usr/bin/env python3
"""
FlyWire Visual Server

Real-time web interface for visualizing FlyWire connectome data and simulations.
Features:
- Interactive 3D fly brain visualization
- Real-time neuron activity monitoring
- Network connectivity analysis
- Cell type and hemilineage exploration
- Simulation control and monitoring
"""

import os, sys
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flywire_integration import FlyWireDataManager, FlyNeuronSimulator

try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    from io import BytesIO
    import base64
    import plotly.graph_objects as go
    import plotly.utils
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install flask flask-socketio matplotlib plotly")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'flywire_visualization_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global FlyWire state
flywire_state = {
    'data_loaded': False,
    'simulation_running': False,
    'current_time': 0.0,
    'simulation_duration': 100.0,
    'neuron_count': 0,
    'connection_count': 0,
    'active_neurons': 0,
    'total_spikes': 0,
    'last_update': datetime.now()
}

# FlyWire data manager and simulator
data_manager = None
simulator = None

class FlyWireVisualizer:
    """Handles FlyWire data visualization and analysis."""
    
    def __init__(self, data_manager: FlyWireDataManager):
        self.data_manager = data_manager
        self.neuron_data = data_manager.neuron_data
        self.connectivity_data = data_manager.connectivity_data
        
        # Color schemes for visualization
        self.cell_type_colors = {
            'Kenyon cell': '#FF6B6B',
            'Mushroom body output neuron': '#4ECDC4',
            'Central complex neuron': '#45B7D1',
            'Optic lobe neuron': '#96CEB4',
            'Antennal lobe neuron': '#FFEAA7',
            'Lateral horn neuron': '#DDA0DD',
            'Subesophageal ganglion neuron': '#98D8C8',
            'Ventral nerve cord neuron': '#F7DC6F'
        }
        
        # Hemilineage colors
        self.hemilineage_colors = {
            'ALad1': '#FF6B6B', 'ALad2': '#FF8E8E',
            'ALl1': '#4ECDC4', 'ALl2': '#6EDDD4',
            'ALv1': '#45B7D1', 'ALv2': '#65C7E1',
            'MBad1': '#96CEB4', 'MBad2': '#B6DEB4',
            'MBl1': '#FFEAA7', 'MBl2': '#FFEAC7',
            'MBv1': '#DDA0DD', 'MBv2': '#FDB0ED',
            'CCad1': '#98D8C8', 'CCad2': '#B8E8C8',
            'CCl1': '#F7DC6F', 'CCl2': '#F7EC8F',
            'CCv1': '#FFB6C1', 'CCv2': '#FFC6D1'
        }
    
    def create_3d_brain_visualization(self) -> Dict:
        """Create 3D visualization of the fly brain network."""
        try:
            # Sample neurons for visualization (limit to 500 for performance)
            sample_size = min(500, len(self.neuron_data))
            sample_neurons = self.neuron_data.sample(sample_size)
            
            # Create 3D scatter plot
            fig = go.Figure()
            
            # Add neurons by cell type
            for cell_type in sample_neurons['cell_type'].unique():
                neurons = sample_neurons[sample_neurons['cell_type'] == cell_type]
                color = self.cell_type_colors.get(cell_type, '#CCCCCC')
                
                fig.add_trace(go.Scatter3d(
                    x=neurons['x_coord'],
                    y=neurons['y_coord'],
                    z=neurons['z_coord'],
                    mode='markers',
                    name=cell_type,
                    marker=dict(
                        size=neurons['soma_size'] / 10,  # Scale soma size
                        color=color,
                        opacity=0.8
                    ),
                    text=neurons['neuron_id'],
                    hovertemplate='<b>%{text}</b><br>' +
                                'Cell Type: ' + neurons['cell_type'] + '<br>' +
                                'Hemilineage: ' + neurons['hemilineage'] + '<br>' +
                                'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f}) μm<br>' +
                                '<extra></extra>'
                ))
            
            # Add some connections for visualization
            sample_connections = self.connectivity_data.sample(min(100, len(self.connectivity_data)))
            
            for _, connection in sample_connections.iterrows():
                pre_neuron = self.data_manager.get_neuron_by_id(connection['pre_neuron_id'])
                post_neuron = self.data_manager.get_neuron_by_id(connection['post_neuron_id'])
                
                if pre_neuron is not None and post_neuron is not None:
                    # Connection line
                    color = 'red' if connection['connection_type'] == 'excitatory' else 'blue'
                    width = connection['synapse_count'] * 2  # Scale by synapse count
                    
                    fig.add_trace(go.Scatter3d(
                        x=[pre_neuron['x_coord'], post_neuron['x_coord']],
                        y=[pre_neuron['y_coord'], post_neuron['y_coord']],
                        z=[pre_neuron['z_coord'], post_neuron['z_coord']],
                        mode='lines',
                        line=dict(color=color, width=width),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            # Update layout
            fig.update_layout(
                title='Fly Brain Network - 3D Visualization',
                scene=dict(
                    xaxis_title='X (μm)',
                    yaxis_title='Y (μm)',
                    zaxis_title='Z (μm)',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                width=800,
                height=600
            )
            
            # Convert to JSON for web display
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            return {
                'success': True,
                'graph_json': graph_json,
                'neuron_count': len(sample_neurons),
                'connection_count': len(sample_connections)
            }
            
        except Exception as e:
            logger.error(f"Failed to create 3D visualization: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_network_topology(self) -> Dict:
        """Create network topology visualization."""
        try:
            # Create network graph using plotly
            fig = go.Figure()
            
            # Sample neurons for network view
            sample_size = min(200, len(self.neuron_data))
            sample_neurons = self.neuron_data.sample(sample_size)
            
            # Create node positions (2D layout)
            x_pos = sample_neurons['x_coord'] / 10  # Scale down
            y_pos = sample_neurons['y_coord'] / 10
            
            # Add nodes
            for i, (_, neuron) in enumerate(sample_neurons.iterrows()):
                color = self.cell_type_colors.get(neuron['cell_type'], '#CCCCCC')
                
                fig.add_trace(go.Scatter(
                    x=[x_pos.iloc[i]],
                    y=[y_pos.iloc[i]],
                    mode='markers',
                    name=neuron['cell_type'],
                    marker=dict(
                        size=neuron['soma_size'] / 5,
                        color=color,
                        opacity=0.8
                    ),
                    text=neuron['neuron_id'],
                    hovertemplate='<b>%{text}</b><br>' +
                                'Type: ' + neuron['cell_type'] + '<br>' +
                                '<extra></extra>',
                    showlegend=False
                ))
            
            # Add connections
            sample_connections = self.connectivity_data.sample(min(50, len(self.connectivity_data)))
            
            for _, connection in sample_connections.iterrows():
                pre_neuron = self.data_manager.get_neuron_by_id(connection['pre_neuron_id'])
                post_neuron = self.data_manager.get_neuron_by_id(connection['post_neuron_id'])
                
                if pre_neuron is not None and post_neuron is not None:
                    # Find indices in sample
                    pre_idx = sample_neurons[sample_neurons['neuron_id'] == pre_neuron['neuron_id']].index
                    post_idx = sample_neurons[sample_neurons['neuron_id'] == post_neuron['neuron_id']].index
                    
                    if len(pre_idx) > 0 and len(post_idx) > 0:
                        pre_idx = pre_idx[0]
                        post_idx = post_idx[0]
                        
                        color = 'red' if connection['connection_type'] == 'excitatory' else 'blue'
                        width = connection['synapse_count']
                        
                        fig.add_trace(go.Scatter(
                            x=[x_pos.iloc[pre_idx], x_pos.iloc[post_idx]],
                            y=[y_pos.iloc[pre_idx], y_pos.iloc[post_idx]],
                            mode='lines',
                            line=dict(color=color, width=width),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
            
            # Update layout
            fig.update_layout(
                title='Fly Brain Network Topology',
                xaxis_title='X (scaled)',
                yaxis_title='Y (scaled)',
                width=800,
                height=600,
                showlegend=True
            )
            
            # Convert to JSON
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            return {
                'success': True,
                'graph_json': graph_json,
                'neuron_count': len(sample_neurons),
                'connection_count': len(sample_connections)
            }
            
        except Exception as e:
            logger.error(f"Failed to create network topology: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_activity_heatmap(self, simulation_data: Dict) -> Dict:
        """Create activity heatmap from simulation data."""
        try:
            # Create time series data
            time_points = np.linspace(0, simulation_data.get('duration', 100), 100)
            
            # Generate activity data for different brain regions
            regions = ['optic_lobe', 'antennal_lobe', 'mushroom_body', 'central_complex']
            activity_data = {}
            
            for region in regions:
                # Simulate region-specific activity
                base_activity = np.random.normal(0.5, 0.2, len(time_points))
                time_modulation = 1.0 + 0.3 * np.sin(time_points * 0.1)
                activity_data[region] = base_activity * time_modulation
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=[activity_data[region] for region in regions],
                x=time_points,
                y=regions,
                colorscale='Viridis',
                colorbar=dict(title='Activity Level')
            ))
            
            fig.update_layout(
                title='Brain Region Activity Over Time',
                xaxis_title='Time (ms)',
                yaxis_title='Brain Region',
                width=800,
                height=400
            )
            
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            return {
                'success': True,
                'graph_json': graph_json
            }
            
        except Exception as e:
            logger.error(f"Failed to create activity heatmap: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_cell_type_distribution(self) -> Dict:
        """Create cell type distribution chart."""
        try:
            cell_type_counts = self.neuron_data['cell_type'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=cell_type_counts.index,
                values=cell_type_counts.values,
                hole=0.3,
                marker_colors=[self.cell_type_colors.get(ct, '#CCCCCC') for ct in cell_type_counts.index]
            )])
            
            fig.update_layout(
                title='Neuron Distribution by Cell Type',
                width=600,
                height=400
            )
            
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            return {
                'success': True,
                'graph_json': graph_json,
                'cell_type_counts': cell_type_counts.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to create cell type distribution: {e}")
            return {'success': False, 'error': str(e)}

# Flask routes
@app.route('/')
def index():
    """Main page."""
    return render_template('flywire_visualization.html')

@app.route('/api/status')
def get_status():
    """Get current FlyWire status."""
    return jsonify(flywire_state)

@app.route('/api/load_data')
def load_flywire_data():
    """Load FlyWire data."""
    global data_manager, simulator
    
    try:
        # Initialize data manager
        data_manager = FlyWireDataManager()
        data_manager.download_sample_data()
        data_manager.load_data()
        
        # Update state
        stats = data_manager.get_network_statistics()
        flywire_state.update({
            'data_loaded': True,
            'neuron_count': stats['total_neurons'],
            'connection_count': stats['total_connections']
        })
        
        return jsonify({
            'success': True,
            'message': 'FlyWire data loaded successfully',
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Failed to load FlyWire data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/visualization/3d')
def get_3d_visualization():
    """Get 3D brain visualization."""
    if not data_manager:
        return jsonify({'success': False, 'error': 'Data not loaded'})
    
    visualizer = FlyWireVisualizer(data_manager)
    result = visualizer.create_3d_brain_visualization()
    return jsonify(result)

@app.route('/api/visualization/topology')
def get_network_topology():
    """Get network topology visualization."""
    if not data_manager:
        return jsonify({'success': False, 'error': 'Data not loaded'})
    
    visualizer = FlyWireVisualizer(data_manager)
    result = visualizer.create_network_topology()
    return jsonify(result)

@app.route('/api/visualization/activity')
def get_activity_heatmap():
    """Get activity heatmap."""
    if not data_manager:
        return jsonify({'success': False, 'error': 'Data not loaded'})
    
    visualizer = FlyWireVisualizer(data_manager)
    simulation_data = {'duration': flywire_state['simulation_duration']}
    result = visualizer.create_activity_heatmap(simulation_data)
    return jsonify(result)

@app.route('/api/visualization/cell_types')
def get_cell_type_distribution():
    """Get cell type distribution."""
    if not data_manager:
        return jsonify({'success': False, 'error': 'Data not loaded'})
    
    visualizer = FlyWireVisualizer(data_manager)
    result = visualizer.create_cell_type_distribution()
    return jsonify(result)

@app.route('/api/simulation/start')
def start_simulation():
    """Start FlyWire simulation."""
    global simulator
    
    if not data_manager:
        return jsonify({'success': False, 'error': 'Data not loaded'})
    
    try:
        simulator = FlyNeuronSimulator(data_manager)
        flywire_state['simulation_running'] = True
        
        # Start simulation in background thread
        def run_simulation():
            simulator.run_simulation(
                duration=flywire_state['simulation_duration'],
                dt=0.1
            )
            flywire_state['simulation_running'] = False
        
        thread = threading.Thread(target=run_simulation)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Simulation started'
        })
        
    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/simulation/stop')
def stop_simulation():
    """Stop FlyWire simulation."""
    global simulator
    
    flywire_state['simulation_running'] = False
    
    if simulator:
        # Get final state
        final_state = simulator.get_simulation_state()
        flywire_state.update({
            'current_time': final_state['simulation_time'],
            'active_neurons': final_state['active_neurons'],
            'total_spikes': final_state['total_spikes']
        })
    
    return jsonify({
        'success': True,
        'message': 'Simulation stopped',
        'final_state': flywire_state
    })

@app.route('/api/simulation/status')
def get_simulation_status():
    """Get simulation status."""
    if not simulator:
        return jsonify({'success': False, 'error': 'No simulation running'})
    
    try:
        state = simulator.get_simulation_state()
        return jsonify({
            'success': True,
            'state': state
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected')
    emit('status', flywire_state)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected')

@socketio.on('request_update')
def handle_update_request():
    """Handle update request from client."""
    if simulator and flywire_state['simulation_running']:
        try:
            state = simulator.get_simulation_state()
            flywire_state.update({
                'current_time': state['simulation_time'],
                'active_neurons': state['active_neurons'],
                'total_spikes': state['total_spikes'],
                'last_update': datetime.now()
            })
            
            emit('simulation_update', flywire_state)
            
        except Exception as e:
            logger.error(f"Failed to get simulation update: {e}")

def main():
    """Main function to run the FlyWire visual server."""
    logger.info("Starting FlyWire Visual Server...")
    
    try:
        # Create templates directory if it doesn't exist
        os.makedirs('templates', exist_ok=True)
        
        # Create HTML template
        create_html_template()
        
        # Start server
        logger.info("FlyWire Visual Server running on http://localhost:5000")
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")

def create_html_template():
    """Create the HTML template for the visualization interface."""
    template_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyWire Brain Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .controls { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .visualization { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        button:disabled { background: #6c757d; cursor: not-allowed; }
        .loading { color: #6c757d; font-style: italic; }
        .error { color: #dc3545; }
        .success { color: #28a745; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 FlyWire Brain Visualization</h1>
            <p>Real-time visualization of Drosophila melanogaster brain connectome data</p>
        </div>
        
        <div class="controls">
            <h3>Controls</h3>
            <button onclick="loadData()" id="loadBtn">Load FlyWire Data</button>
            <button onclick="startSimulation()" id="startBtn" disabled>Start Simulation</button>
            <button onclick="stopSimulation()" id="stopBtn" disabled>Stop Simulation</button>
            <button onclick="refreshVisualizations()" id="refreshBtn">Refresh Visualizations</button>
        </div>
        
        <div class="status">
            <h3>Status</h3>
            <div id="statusDisplay">Loading...</div>
        </div>
        
        <div class="visualization">
            <h3>3D Brain Network</h3>
            <div id="3dVisualization" class="loading">Click "Load FlyWire Data" to begin</div>
        </div>
        
        <div class="visualization">
            <h3>Network Topology</h3>
            <div id="topologyVisualization" class="loading">Loading...</div>
        </div>
        
        <div class="visualization">
            <h3>Cell Type Distribution</h3>
            <div id="cellTypeVisualization" class="loading">Loading...</div>
        </div>
        
        <div class="visualization">
            <h3>Activity Heatmap</h3>
            <div id="activityVisualization" class="loading">Loading...</div>
        </div>
    </div>

    <script>
        const socket = io();
        let dataLoaded = false;
        
        // Socket events
        socket.on('connect', function() {
            console.log('Connected to server');
            updateStatus('Connected to server');
        });
        
        socket.on('status', function(data) {
            updateStatus(data);
        });
        
        socket.on('simulation_update', function(data) {
            updateStatus(data);
        });
        
        // Functions
        async function loadData() {
            const btn = document.getElementById('loadBtn');
            btn.disabled = true;
            btn.textContent = 'Loading...';
            
            try {
                const response = await fetch('/api/load_data');
                const data = await response.json();
                
                if (data.success) {
                    dataLoaded = true;
                    document.getElementById('startBtn').disabled = false;
                    updateStatus('Data loaded successfully');
                    refreshVisualizations();
                } else {
                    updateStatus('Error: ' + data.error);
                }
            } catch (error) {
                updateStatus('Error: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Load FlyWire Data';
            }
        }
        
        async function startSimulation() {
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            try {
                const response = await fetch('/api/simulation/start');
                const data = await response.json();
                
                if (data.success) {
                    updateStatus('Simulation started');
                } else {
                    updateStatus('Error: ' + data.error);
                }
            } catch (error) {
                updateStatus('Error: ' + error.message);
            }
        }
        
        async function stopSimulation() {
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            
            startBtn.disabled = false;
            stopBtn.disabled = true;
            
            try {
                const response = await fetch('/api/simulation/stop');
                const data = await response.json();
                
                if (data.success) {
                    updateStatus('Simulation stopped');
                } else {
                    updateStatus('Error: ' + data.error);
                }
            } catch (error) {
                updateStatus('Error: ' + error.message);
            }
        }
        
        async function refreshVisualizations() {
            if (!dataLoaded) return;
            
            // Load 3D visualization
            try {
                const response = await fetch('/api/visualization/3d');
                const data = await response.json();
                
                if (data.success) {
                    const graphData = JSON.parse(data.graph_json);
                    Plotly.newPlot('3dVisualization', graphData.data, graphData.layout);
                } else {
                    document.getElementById('3dVisualization').innerHTML = '<span class="error">Error: ' + data.error + '</span>';
                }
            } catch (error) {
                document.getElementById('3dVisualization').innerHTML = '<span class="error">Error: ' + error.message + '</span>';
            }
            
            // Load topology visualization
            try {
                const response = await fetch('/api/visualization/topology');
                const data = await response.json();
                
                if (data.success) {
                    const graphData = JSON.parse(data.graph_json);
                    Plotly.newPlot('topologyVisualization', graphData.data, graphData.layout);
                } else {
                    document.getElementById('topologyVisualization').innerHTML = '<span class="error">Error: ' + data.error + '</span>';
                }
            } catch (error) {
                document.getElementById('topologyVisualization').innerHTML = '<span class="error">Error: ' + error.message + '</span>';
            }
            
            // Load cell type distribution
            try {
                const response = await fetch('/api/visualization/cell_types');
                const data = await response.json();
                
                if (data.success) {
                    const graphData = JSON.parse(data.graph_json);
                    Plotly.newPlot('cellTypeVisualization', graphData.data, graphData.layout);
                } else {
                    document.getElementById('cellTypeVisualization').innerHTML = '<span class="error">Error: ' + data.error + '</span>';
                }
            } catch (error) {
                document.getElementById('cellTypeVisualization').innerHTML = '<span class="error">Error: ' + error.message + '</span>';
            }
            
            // Load activity heatmap
            try {
                const response = await fetch('/api/visualization/activity');
                const data = await response.json();
                
                if (data.success) {
                    const graphData = JSON.parse(data.graph_json);
                    Plotly.newPlot('activityVisualization', graphData.data, graphData.layout);
                } else {
                    document.getElementById('activityVisualization').innerHTML = '<span class="error">Error: ' + data.error + '</span>';
                }
            } catch (error) {
                document.getElementById('activityVisualization').innerHTML = '<span class="error">Error: ' + error.message + '</span>';
            }
        }
        
        function updateStatus(data) {
            const statusDiv = document.getElementById('statusDisplay');
            
            if (typeof data === 'string') {
                statusDiv.innerHTML = '<p>' + data + '</p>';
            } else {
                let html = '<p><strong>Data Loaded:</strong> ' + (data.data_loaded ? 'Yes' : 'No') + '</p>';
                html += '<p><strong>Simulation Running:</strong> ' + (data.simulation_running ? 'Yes' : 'No') + '</p>';
                
                if (data.neuron_count) {
                    html += '<p><strong>Neurons:</strong> ' + data.neuron_count.toLocaleString() + '</p>';
                }
                
                if (data.connection_count) {
                    html += '<p><strong>Connections:</strong> ' + data.connection_count.toLocaleString() + '</p>';
                }
                
                if (data.current_time !== undefined) {
                    html += '<p><strong>Simulation Time:</strong> ' + data.current_time.toFixed(1) + ' ms</p>';
                }
                
                if (data.active_neurons !== undefined) {
                    html += '<p><strong>Active Neurons:</strong> ' + data.active_neurons + '</p>';
                }
                
                if (data.total_spikes !== undefined) {
                    html += '<p><strong>Total Spikes:</strong> ' + data.total_spikes + '</p>';
                }
                
                statusDiv.innerHTML = html;
            }
        }
        
        // Auto-refresh status
        setInterval(() => {
            if (dataLoaded) {
                socket.emit('request_update');
            }
        }, 1000);
    </script>
</body>
</html>
    '''
    
    with open('templates/flywire_visualization.html', 'w') as f:
        f.write(template_content)
    
    logger.info("HTML template created")

if __name__ == "__main__":
    main()
