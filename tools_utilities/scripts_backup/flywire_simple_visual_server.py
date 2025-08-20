#!/usr/bin/env python3
"""
FlyWire Simple Visual Server

A simplified web interface for visualizing FlyWire connectome data.
Provides basic 3D visualization and network analysis capabilities.
"""

import os, sys
import json
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flywire_integration import FlyWireDataManager, FlyNeuronSimulator

try:
    from flask import Flask, render_template, jsonify, request
    import plotly.graph_objects as go
    import plotly.utils
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install flask plotly")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'flywire_simple_visualization'

# Global FlyWire state
flywire_state = {
    'data_loaded': False,
    'neuron_count': 0,
    'connection_count': 0
}

# FlyWire data manager
data_manager = None

class FlyWireSimpleVisualizer:
    """Simplified FlyWire data visualizer."""
    
    def __init__(self, data_manager: FlyWireDataManager):
        self.data_manager = data_manager
        self.neuron_data = data_manager.neuron_data
        self.connectivity_data = data_manager.connectivity_data
        
        # Color schemes
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
    
    def create_3d_brain_visualization(self) -> Dict:
        """Create 3D visualization of the fly brain network."""
        try:
            # Sample neurons for visualization (limit to 300 for performance)
            sample_size = min(300, len(self.neuron_data))
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
                        size=neurons['soma_size'] / 10,
                        color=color,
                        opacity=0.8
                    ),
                    text=neurons['neuron_id'],
                    hovertemplate='<b>%{text}</b><br>' +
                                'Cell Type: ' + neurons['cell_type'] + '<br>' +
                                'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f}) μm<br>' +
                                '<extra></extra>'
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
                'neuron_count': len(sample_neurons)
            }
            
        except Exception as e:
            logger.error(f"Failed to create 3D visualization: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_network_topology(self) -> Dict:
        """Create network topology visualization."""
        try:
            # Create network graph
            fig = go.Figure()
            
            # Sample neurons for network view
            sample_size = min(150, len(self.neuron_data))
            sample_neurons = self.neuron_data.sample(sample_size)
            
            # Create node positions (2D layout)
            x_pos = sample_neurons['x_coord'] / 10
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
                'neuron_count': len(sample_neurons)
            }
            
        except Exception as e:
            logger.error(f"Failed to create network topology: {e}")
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
    return render_template('flywire_simple.html')

@app.route('/api/status')
def get_status():
    """Get current FlyWire status."""
    return jsonify(flywire_state)

@app.route('/api/load_data')
def load_flywire_data():
    """Load FlyWire data."""
    global data_manager
    
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
    
    visualizer = FlyWireSimpleVisualizer(data_manager)
    result = visualizer.create_3d_brain_visualization()
    return jsonify(result)

@app.route('/api/visualization/topology')
def get_network_topology():
    """Get network topology visualization."""
    if not data_manager:
        return jsonify({'success': False, 'error': 'Data not loaded'})
    
    visualizer = FlyWireSimpleVisualizer(data_manager)
    result = visualizer.create_network_topology()
    return jsonify(result)

@app.route('/api/visualization/cell_types')
def get_cell_type_distribution():
    """Get cell type distribution."""
    if not data_manager:
        return jsonify({'success': False, 'error': 'Data not loaded'})
    
    visualizer = FlyWireSimpleVisualizer(data_manager)
    result = visualizer.create_cell_type_distribution()
    return jsonify(result)

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
            <p>Interactive visualization of Drosophila melanogaster brain connectome data</p>
        </div>
        
        <div class="controls">
            <h3>Controls</h3>
            <button onclick="loadData()" id="loadBtn">Load FlyWire Data</button>
            <button onclick="refreshVisualizations()" id="refreshBtn" disabled>Refresh Visualizations</button>
        </div>
        
        <div class="status">
            <h3>Status</h3>
            <div id="statusDisplay">Click "Load FlyWire Data" to begin</div>
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
    </div>

    <script>
        let dataLoaded = false;
        
        async function loadData() {
            const btn = document.getElementById('loadBtn');
            const refreshBtn = document.getElementById('refreshBtn');
            
            btn.disabled = true;
            btn.textContent = 'Loading...';
            
            try {
                const response = await fetch('/api/load_data');
                const data = await response.json();
                
                if (data.success) {
                    dataLoaded = true;
                    refreshBtn.disabled = false;
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
        }
        
        function updateStatus(message) {
            const statusDiv = document.getElementById('statusDisplay');
            statusDiv.innerHTML = '<p class="success">' + message + '</p>';
        }
    </script>
</body>
</html>
    '''
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    with open('templates/flywire_simple.html', 'w') as f:
        f.write(template_content)
    
    logger.info("HTML template created")

def main():
    """Main function to run the FlyWire simple visual server."""
    logger.info("Starting FlyWire Simple Visual Server...")
    
    try:
        # Create HTML template
        create_html_template()
        
        # Start server
        logger.info("FlyWire Simple Visual Server running on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")

if __name__ == "__main__":
    main()
