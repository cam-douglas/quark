#!/usr/bin/env python3
"""
Live 3D Visualizer for Quark Live Streaming System.
Creates interactive 3D visualizations of test results, experiments, and performance data.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import os

from testing.visualizations.visual_utils import save_fig, live_series


class Live3DVisualizer:
    """Creates 3D visualizations for live streaming data."""
    
    def __init__(self):
        self.data_points = []
        self.test_results = []
        self.performance_metrics = []
        self.experiment_data = []
        
    def create_3d_test_landscape(self, test_results: List[Dict]) -> go.Figure:
        """Create a 3D landscape of test results."""
        if not test_results:
            return self._create_empty_3d_plot("No test data available")
        
        # Extract data for 3D plotting
        x_vals = []  # Test index
        y_vals = []  # Duration
        z_vals = []  # Success rate (1 for pass, 0 for fail)
        colors = []
        text_labels = []
        
        for i, test in enumerate(test_results):
            x_vals.append(i)
            y_vals.append(test.get('duration', 0))
            z_vals.append(1 if test.get('status') == 'PASSED' else 0)
            colors.append('green' if test.get('status') == 'PASSED' else 'red')
            text_labels.append(f"Test: {test.get('name', 'Unknown')}<br>Duration: {test.get('duration', 0):.4f}s<br>Status: {test.get('status', 'Unknown')}")
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers+text',
            marker=dict(
                size=12,
                color=colors,
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=[f"T{i+1}" for i in range(len(test_results))],
            textposition="middle center",
            hovertext=text_labels,
            hoverinfo='text',
            name='Test Results'
        )])
        
        # Add surface for success rate
        if len(x_vals) > 1:
            x_surface = np.linspace(0, len(x_vals)-1, 10)
            y_surface = np.linspace(min(y_vals), max(y_vals), 10)
            X, Y = np.meshgrid(x_surface, y_surface)
            Z = np.ones_like(X) * 0.5  # Success threshold plane
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                opacity=0.3,
                colorscale='Viridis',
                showscale=False,
                name='Success Threshold'
            ))
        
        fig.update_layout(
            title='🧪 3D Test Results Landscape',
            scene=dict(
                xaxis_title='Test Index',
                yaxis_title='Duration (seconds)',
                zaxis_title='Success Rate',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            template='plotly_dark'
        )
        
        return fig
    
    def create_3d_performance_cube(self, performance_data: List[Dict]) -> go.Figure:
        """Create a 3D cube visualization of performance metrics."""
        if not performance_data:
            return self._create_empty_3d_plot("No performance data available")
        
        # Extract performance metrics
        timestamps = []
        cpu_usage = []
        memory_usage = []
        message_count = []
        
        for data in performance_data:
            timestamps.append(data.get('timestamp', 0))
            cpu_usage.append(data.get('cpu_usage', 0))
            memory_usage.append(data.get('memory_usage', 0))
            message_count.append(data.get('message_count', 0))
        
        # Normalize timestamps to start from 0
        if timestamps:
            min_time = min(timestamps)
            timestamps = [(t - min_time) for t in timestamps]
        
        # Create 3D performance visualization
        fig = go.Figure()
        
        # CPU usage trajectory
        fig.add_trace(go.Scatter3d(
            x=timestamps,
            y=cpu_usage,
            z=[0] * len(timestamps),
            mode='lines+markers',
            line=dict(color='red', width=6),
            marker=dict(size=4),
            name='CPU Usage (%)',
            hovertemplate='Time: %{x}s<br>CPU: %{y}%<extra></extra>'
        ))
        
        # Memory usage trajectory
        fig.add_trace(go.Scatter3d(
            x=timestamps,
            y=[0] * len(timestamps),
            z=memory_usage,
            mode='lines+markers',
            line=dict(color='blue', width=6),
            marker=dict(size=4),
            name='Memory Usage (%)',
            hovertemplate='Time: %{x}s<br>Memory: %{z}%<extra></extra>'
        ))
        
        # Message count trajectory
        if message_count:
            max_messages = max(message_count) if message_count else 1
            normalized_messages = [m / max_messages * 100 for m in message_count]
            
            fig.add_trace(go.Scatter3d(
                x=[0] * len(timestamps),
                y=normalized_messages,
                z=timestamps,
                mode='lines+markers',
                line=dict(color='green', width=6),
                marker=dict(size=4),
                name='Message Count (normalized)',
                hovertemplate='Messages: %{y}<br>Time: %{z}s<extra></extra>'
            ))
        
        fig.update_layout(
            title='📊 3D Performance Metrics Cube',
            scene=dict(
                xaxis_title='Time (seconds)',
                yaxis_title='CPU Usage (%)',
                zaxis_title='Memory Usage (%)',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            template='plotly_dark'
        )
        
        return fig
    
    def create_3d_experiment_space(self, experiments: List[Dict]) -> go.Figure:
        """Create a 3D space visualization of experiments."""
        if not experiments:
            return self._create_empty_3d_plot("No experiment data available")
        
        # Extract experiment parameters for 3D plotting
        learning_rates = []
        batch_sizes = []
        durations = []
        success_rates = []
        names = []
        
        for exp in experiments:
            params = exp.get('params', {})
            learning_rates.append(params.get('learning_rate', 0.001))
            batch_sizes.append(params.get('batch_size', 32))
            durations.append(exp.get('duration', 0))
            success_rates.append(1 if exp.get('success', False) else 0)
            names.append(exp.get('name', 'Unknown'))
        
        # Create 3D experiment space
        fig = go.Figure(data=[go.Scatter3d(
            x=learning_rates,
            y=batch_sizes,
            z=durations,
            mode='markers+text',
            marker=dict(
                size=[sr * 20 + 5 for sr in success_rates],  # Size based on success
                color=success_rates,
                colorscale='RdYlGn',
                opacity=0.8,
                colorbar=dict(title="Success Rate"),
                line=dict(width=2, color='white')
            ),
            text=[f"E{i+1}" for i in range(len(experiments))],
            textposition="middle center",
            hovertext=[f"Experiment: {name}<br>LR: {lr}<br>Batch: {bs}<br>Duration: {dur:.4f}s" 
                      for name, lr, bs, dur in zip(names, learning_rates, batch_sizes, durations)],
            hoverinfo='text',
            name='Experiments'
        )])
        
        fig.update_layout(
            title='🧬 3D Experiment Parameter Space',
            scene=dict(
                xaxis_title='Learning Rate',
                yaxis_title='Batch Size',
                zaxis_title='Duration (seconds)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            template='plotly_dark'
        )
        
        return fig
    
    def create_3d_neural_network_topology(self, layers: List[int] = None) -> go.Figure:
        """Create a 3D visualization of neural network topology."""
        if layers is None:
            layers = [784, 128, 64, 10]  # Default network architecture
        
        fig = go.Figure()
        
        # Create nodes for each layer
        for layer_idx, num_nodes in enumerate(layers):
            # Arrange nodes in a circle for each layer
            angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
            radius = max(1, num_nodes / 10)  # Adjust radius based on layer size
            
            x_coords = radius * np.cos(angles)
            y_coords = radius * np.sin(angles)
            z_coords = [layer_idx * 2] * num_nodes  # Space layers apart
            
            # Add nodes
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=8,
                    color=layer_idx,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                name=f'Layer {layer_idx + 1} ({num_nodes} nodes)',
                hovertemplate=f'Layer {layer_idx + 1}<br>Node: %{{pointNumber}}<extra></extra>'
            ))
            
            # Add connections to next layer
            if layer_idx < len(layers) - 1:
                next_layer_nodes = layers[layer_idx + 1]
                next_angles = np.linspace(0, 2*np.pi, next_layer_nodes, endpoint=False)
                next_radius = max(1, next_layer_nodes / 10)
                
                next_x = next_radius * np.cos(next_angles)
                next_y = next_radius * np.sin(next_angles)
                next_z = [(layer_idx + 1) * 2] * next_layer_nodes
                
                # Add some sample connections (not all to avoid clutter)
                for i in range(min(5, num_nodes)):
                    for j in range(min(3, next_layer_nodes)):
                        fig.add_trace(go.Scatter3d(
                            x=[x_coords[i], next_x[j]],
                            y=[y_coords[i], next_y[j]],
                            z=[z_coords[i], next_z[j]],
                            mode='lines',
                            line=dict(color='rgba(255,255,255,0.3)', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        fig.update_layout(
            title='🧠 3D Neural Network Topology',
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Layer Depth',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            template='plotly_dark'
        )
        
        return fig
    
    def _create_empty_3d_plot(self, message: str) -> go.Figure:
        """Create an empty 3D plot with a message."""
        fig = go.Figure(data=[go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='text',
            text=[message],
            textposition="middle center",
            showlegend=False
        )])
        
        fig.update_layout(
            title='3D Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            template='plotly_dark'
        )
        
        return fig
    
    def stream_3d_visualization(self, viz_type: str, data: Any):
        """Stream 3D visualization data to live dashboard."""
        try:
            if viz_type == "test_landscape":
                fig = self.create_3d_test_landscape(data)
            elif viz_type == "performance_cube":
                fig = self.create_3d_performance_cube(data)
            elif viz_type == "experiment_space":
                fig = self.create_3d_experiment_space(data)
            elif viz_type == "neural_topology":
                fig = self.create_3d_neural_network_topology(data)
            else:
                fig = self._create_empty_3d_plot(f"Unknown visualization type: {viz_type}")
            
            # Save 3D visualization as HTML only (skip PNG to avoid Chrome dependency)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"3d_{viz_type}_{timestamp}"
            
            # Create output directory
            output_dir = "testing/visualizations/outputs/3d/"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as interactive HTML
            html_path = os.path.join(output_dir, f"{filename}.html")
            fig.write_html(html_path, include_plotlyjs='cdn')
            print(f"🌐 Saved 3D visualization: {html_path}")
            
            # Stream 3D visualization data live to dashboard
            live_series(f"3d_{viz_type}", {
                "type": viz_type,
                "timestamp": timestamp,
                "data_points": len(data) if isinstance(data, list) else 1,
                "plot_data": {
                    "figure": fig.to_dict(),  # Full Plotly figure data
                    "layout": fig.layout.to_dict() if hasattr(fig.layout, 'to_dict') else str(fig.layout),  # Convert layout
                    "data": [trace.to_dict() if hasattr(trace, 'to_dict') else str(trace) for trace in fig.data]  # Convert traces
                },
                "interactive": True,
                "dimensions": 3
            }, int(time.time()))
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating 3D visualization: {e}")
            return self._create_empty_3d_plot(f"Error: {e}")


# Global 3D visualizer instance
_3d_visualizer = None

def get_3d_visualizer():
    """Get the global 3D visualizer instance."""
    global _3d_visualizer
    if _3d_visualizer is None:
        _3d_visualizer = Live3DVisualizer()
    return _3d_visualizer

def create_3d_test_demo():
    """Create a 3D demonstration with sample data."""
    visualizer = get_3d_visualizer()
    
    # Sample test results
    test_results = [
        {"name": "test_thalamic_relay", "duration": 0.001, "status": "PASSED"},
        {"name": "test_stdp_weights", "duration": 0.015, "status": "PASSED"},
        {"name": "test_action_gate", "duration": 0.002, "status": "PASSED"},
        {"name": "test_layer_sheet", "duration": 0.012, "status": "FAILED"},
        {"name": "test_neural_dynamics", "duration": 0.008, "status": "PASSED"}
    ]
    
    # Sample performance data
    performance_data = []
    for i in range(20):
        performance_data.append({
            "timestamp": i,
            "cpu_usage": 20 + 10 * np.sin(i * 0.3) + np.random.normal(0, 2),
            "memory_usage": 45 + 5 * np.cos(i * 0.2) + np.random.normal(0, 1),
            "message_count": i * 2 + np.random.randint(0, 5)
        })
    
    # Sample experiments
    experiments = []
    for i in range(8):
        experiments.append({
            "name": f"experiment_{i}",
            "params": {
                "learning_rate": 0.001 * (i + 1),
                "batch_size": 16 * (i + 1),
                "epochs": 5 + i
            },
            "duration": 0.001 + i * 0.002,
            "success": np.random.choice([True, False], p=[0.8, 0.2])
        })
    
    print("🚀 Creating 3D Visualizations...")
    
    # Create all 3D visualizations
    fig1 = visualizer.stream_3d_visualization("test_landscape", test_results)
    fig2 = visualizer.stream_3d_visualization("performance_cube", performance_data)
    fig3 = visualizer.stream_3d_visualization("experiment_space", experiments)
    fig4 = visualizer.stream_3d_visualization("neural_topology", [784, 256, 128, 64, 10])
    
    print("✅ 3D visualizations created and streamed!")
    return [fig1, fig2, fig3, fig4]


if __name__ == "__main__":
    # Create 3D demonstration
    print("🌟 3D Live Visualization Demo")
    print("=" * 50)
    
    # Start live server
    from testing.visualizations.visual_utils import start_live_server
    start_live_server()
    
    # Create 3D visualizations
    figures = create_3d_test_demo()
    
    print(f"\n📊 Created {len(figures)} 3D visualizations")
    print("🌐 Check your browser for live 3D visualizations!")
    
    # Keep server running to show results
    time.sleep(10)
