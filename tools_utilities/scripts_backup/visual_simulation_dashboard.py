#!/usr/bin/env python3
"""
🧠 Visual Simulation Dashboard
=============================

Comprehensive visual simulation tools and real-time parameter monitoring dashboards
for brain training and consciousness development with 3D visualizations.

Features:
- Real-time 3D brain network visualization
- Interactive parameter monitoring
- Neural activity simulation
- Consciousness level visualization
- Training progress tracking
- Biological compliance monitoring
- Web-based interactive dashboard
- Real-time data streaming

Author: Quark Brain Simulation Team
Created: 2025-01-21
"""

import os, sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, asdict
import threading
import time
import queue
import asyncio
import websockets
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import psutil
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add quark root to path
QUARK_ROOT = Path(__file__).parent.parent
sys.path.append(str(QUARK_ROOT))

@dataclass
class SimulationState:
    """Current state of the brain simulation."""
    timestamp: datetime
    neural_activity: Dict[str, np.ndarray]
    connectivity_matrix: np.ndarray
    consciousness_level: float
    attention_focus: Dict[str, float]
    training_progress: Dict[str, float]
    biological_compliance: float
    energy_consumption: float
    memory_usage: float
    active_modules: List[str]

@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    update_interval: float = 1.0  # seconds
    max_history_points: int = 1000
    neural_activity_scale: float = 1.0
    consciousness_threshold: float = 0.5
    show_inactive_modules: bool = True
    animation_speed: float = 1.0
    color_scheme: str = "viridis"

class NeuralActivitySimulator:
    """Simulates realistic neural activity patterns."""
    
    def __init__(self, num_regions: int = 15, num_neurons_per_region: int = 100):
        self.num_regions = num_regions
        self.num_neurons_per_region = num_neurons_per_region
        self.total_neurons = num_regions * num_neurons_per_region
        
        # Brain region definitions
        self.brain_regions = [
            'Prefrontal Cortex', 'Motor Cortex', 'Somatosensory Cortex',
            'Visual Cortex', 'Auditory Cortex', 'Thalamus', 'Basal Ganglia',
            'Hippocampus', 'Amygdala', 'Cerebellum', 'Brainstem',
            'Working Memory', 'Default Mode', 'Salience Network', 'Conscious Agent'
        ]
        
        # Initialize neural activity
        self.neural_state = np.random.randn(self.total_neurons) * 0.1
        self.firing_rates = np.zeros(self.num_regions)
        self.connectivity = self._generate_connectivity_matrix()
        
        # Oscillatory patterns
        self.time_step = 0
        self.oscillation_frequencies = np.random.uniform(4, 40, self.num_regions)  # Hz
        
    def _generate_connectivity_matrix(self) -> np.ndarray:
        """Generate biologically plausible connectivity matrix."""
        # Start with distance-based connectivity
        connectivity = np.zeros((self.num_regions, self.num_regions))
        
        # Define anatomical connectivity patterns
        anatomical_connections = {
            0: [1, 2, 5, 6, 11, 14],  # PFC connections
            1: [0, 2, 9],             # Motor connections
            2: [0, 1, 3],             # Somatosensory connections
            3: [2, 5, 14],            # Visual connections
            4: [5, 14],               # Auditory connections
            5: [0, 1, 2, 3, 4],       # Thalamic connections
            6: [0, 1, 5],             # Basal ganglia connections
            7: [0, 12, 14],           # Hippocampal connections
            8: [0, 7, 10],            # Amygdala connections
            9: [1, 10],               # Cerebellar connections
            10: [8, 9],               # Brainstem connections
            11: [0, 5, 14],           # Working memory connections
            12: [0, 7, 14],           # Default mode connections
            13: [0, 5, 14],           # Salience network connections
            14: [0, 3, 4, 5, 7, 11, 12, 13]  # Conscious agent connections
        }
        
        # Set connections with realistic weights
        for source, targets in anatomical_connections.items():
            for target in targets:
                if source < self.num_regions and target < self.num_regions:
                    weight = np.random.exponential(0.3)  # Exponential distribution
                    connectivity[source, target] = weight
                    connectivity[target, source] = weight * 0.8  # Slightly asymmetric
        
        # Normalize
        max_weight = np.max(connectivity)
        if max_weight > 0:
            connectivity = connectivity / max_weight
            
        return connectivity
        
    def update_neural_activity(self, consciousness_level: float = 0.5, 
                             attention_focus: Dict[str, float] = None) -> Dict[str, np.ndarray]:
        """Update neural activity simulation."""
        self.time_step += 1
        dt = 0.01  # Time step in seconds
        
        # Base neural dynamics (Ornstein-Uhlenbeck process)
        decay = 0.95
        noise_strength = 0.1
        
        # Apply consciousness modulation
        consciousness_boost = 1.0 + consciousness_level * 0.5
        
        # Update each region
        for region_idx in range(self.num_regions):
            # Oscillatory component
            oscillation = np.sin(2 * np.pi * self.oscillation_frequencies[region_idx] * self.time_step * dt)
            
            # Attention modulation
            attention_weight = 1.0
            if attention_focus:
                region_name = self.brain_regions[region_idx]
                for focus_area, weight in attention_focus.items():
                    if focus_area.lower() in region_name.lower():
                        attention_weight = 1.0 + weight
                        break
            
            # Connectivity-driven activity
            connected_activity = 0
            for other_region in range(self.num_regions):
                if self.connectivity[region_idx, other_region] > 0:
                    connected_activity += (
                        self.connectivity[region_idx, other_region] * 
                        self.firing_rates[other_region]
                    )
            
            # Update firing rate
            new_rate = (
                decay * self.firing_rates[region_idx] +
                0.1 * oscillation * consciousness_boost * attention_weight +
                0.05 * connected_activity +
                noise_strength * np.random.randn()
            )
            
            # Apply activation function (sigmoid-like)
            self.firing_rates[region_idx] = np.tanh(new_rate)
        
        # Generate detailed neural activity for each region
        neural_activity = {}
        for region_idx, region_name in enumerate(self.brain_regions):
            # Generate population activity
            base_activity = self.firing_rates[region_idx]
            
            # Add spatial structure within region
            population_activity = np.random.normal(
                base_activity, 
                abs(base_activity) * 0.3 + 0.1, 
                self.num_neurons_per_region
            )
            
            # Apply spatial correlations
            population_activity = gaussian_filter(population_activity.reshape(10, 10), sigma=1.0).flatten()
            
            neural_activity[region_name] = population_activity
        
        return neural_activity
        
    def get_connectivity_matrix(self) -> np.ndarray:
        """Get current connectivity matrix."""
        return self.connectivity.copy()
        
    def get_region_names(self) -> List[str]:
        """Get brain region names."""
        return self.brain_regions.copy()

class Real3DBrainVisualizer:
    """Real-time 3D brain network visualizer."""
    
    def __init__(self):
        self.brain_coordinates = self._generate_brain_coordinates()
        self.connection_cache = {}
        
    def _generate_brain_coordinates(self) -> Dict[str, Tuple[float, float, float]]:
        """Generate 3D coordinates for brain regions."""
        # Approximate brain region coordinates (normalized)
        coordinates = {
            'Prefrontal Cortex': (0.0, 0.8, 0.3),
            'Motor Cortex': (-0.3, 0.3, 0.6),
            'Somatosensory Cortex': (-0.5, 0.0, 0.5),
            'Visual Cortex': (0.0, -0.8, 0.2),
            'Auditory Cortex': (-0.7, 0.0, 0.0),
            'Thalamus': (0.0, 0.0, -0.2),
            'Basal Ganglia': (-0.2, 0.2, -0.3),
            'Hippocampus': (-0.4, -0.3, -0.4),
            'Amygdala': (-0.3, -0.1, -0.5),
            'Cerebellum': (0.0, -0.5, -0.8),
            'Brainstem': (0.0, -0.2, -0.6),
            'Working Memory': (0.2, 0.6, 0.4),
            'Default Mode': (0.0, -0.4, 0.3),
            'Salience Network': (0.4, 0.4, 0.2),
            'Conscious Agent': (0.0, 0.0, 0.8)
        }
        return coordinates
        
    def create_3d_brain_network(self, neural_activity: Dict[str, np.ndarray],
                               connectivity_matrix: np.ndarray,
                               consciousness_level: float,
                               attention_focus: Dict[str, float] = None) -> go.Figure:
        """Create 3D brain network visualization."""
        
        # Prepare node data
        node_names = list(neural_activity.keys())
        node_positions = [self.brain_coordinates.get(name, (0, 0, 0)) for name in node_names]
        
        # Calculate node activities (mean firing rate)
        node_activities = [np.mean(np.abs(activity)) for activity in neural_activity.values()]
        max_activity = max(node_activities) if node_activities else 1.0
        normalized_activities = [act / max_activity for act in node_activities]
        
        # Node colors based on activity and consciousness
        node_colors = []
        for i, activity in enumerate(normalized_activities):
            # Base color from activity
            if activity > 0.7:
                color = 'red'  # High activity
            elif activity > 0.4:
                color = 'orange'  # Medium activity
            elif activity > 0.2:
                color = 'yellow'  # Low activity
            else:
                color = 'blue'  # Very low activity
                
            # Special coloring for conscious agent
            if 'conscious' in node_names[i].lower():
                if consciousness_level > 0.7:
                    color = 'magenta'  # High consciousness
                elif consciousness_level > 0.4:
                    color = 'purple'  # Medium consciousness
                else:
                    color = 'darkblue'  # Low consciousness
                    
            node_colors.append(color)
        
        # Node sizes based on activity and attention
        node_sizes = []
        for i, activity in enumerate(normalized_activities):
            base_size = 10 + activity * 20  # Base size 10-30
            
            # Attention boost
            if attention_focus:
                for focus_area, weight in attention_focus.items():
                    if focus_area.lower() in node_names[i].lower():
                        base_size *= (1 + weight)
                        break
                        
            node_sizes.append(min(50, base_size))  # Cap at 50
        
        # Create 3D scatter plot for nodes
        x_coords = [pos[0] for pos in node_positions]
        y_coords = [pos[1] for pos in node_positions]
        z_coords = [pos[2] for pos in node_positions]
        
        node_trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=[name[:12] for name in node_names],  # Truncate names
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>' +
                         'Activity: %{customdata[0]:.3f}<br>' +
                         'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                         '<extra></extra>',
            customdata=[[act] for act in normalized_activities],
            name='Brain Regions'
        )
        
        # Create edges for significant connections
        edge_traces = []
        threshold = 0.3  # Only show strong connections
        
        for i in range(len(node_names)):
            for j in range(i + 1, len(node_names)):
                if i < connectivity_matrix.shape[0] and j < connectivity_matrix.shape[1]:
                    connection_strength = connectivity_matrix[i, j]
                    
                    if connection_strength > threshold:
                        # Create edge
                        x_edge = [x_coords[i], x_coords[j], None]
                        y_edge = [y_coords[i], y_coords[j], None]
                        z_edge = [z_coords[i], z_coords[j], None]
                        
                        # Edge color and width based on connection strength
                        edge_width = connection_strength * 5
                        edge_opacity = min(1.0, connection_strength * 2)
                        
                        edge_trace = go.Scatter3d(
                            x=x_edge,
                            y=y_edge,
                            z=z_edge,
                            mode='lines',
                            line=dict(
                                width=edge_width,
                                color=f'rgba(100, 100, 100, {edge_opacity})'
                            ),
                            hoverinfo='none',
                            showlegend=False
                        )
                        edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)
        
        # Update layout for 3D brain visualization
        fig.update_layout(
            title=f'🧠 Real-Time 3D Brain Network (Consciousness: {consciousness_level:.3f})',
            scene=dict(
                xaxis=dict(title='X', showgrid=False, zeroline=False),
                yaxis=dict(title='Y', showgrid=False, zeroline=False),
                zaxis=dict(title='Z', showgrid=False, zeroline=False),
                bgcolor='black',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor='black',
            font=dict(color='white')
        )
        
        return fig
        
    def create_neural_activity_heatmap(self, neural_activity: Dict[str, np.ndarray]) -> go.Figure:
        """Create heatmap of neural activity across regions."""
        
        # Prepare data
        region_names = list(neural_activity.keys())
        activity_matrix = []
        
        for region_name in region_names:
            activity = neural_activity[region_name]
            # Reshape to 2D for visualization (10x10 grid)
            if len(activity) >= 100:
                activity_2d = activity[:100].reshape(10, 10)
            else:
                # Pad if necessary
                padded_activity = np.pad(activity, (0, 100 - len(activity)), mode='constant')
                activity_2d = padded_activity.reshape(10, 10)
                
            activity_matrix.append(activity_2d)
        
        # Create subplots
        rows = int(np.ceil(len(region_names) / 3))
        cols = min(3, len(region_names))
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[name[:15] for name in region_names],
            specs=[[{"type": "heatmap"} for _ in range(cols)] for _ in range(rows)]
        )
        
        # Add heatmaps
        for i, (region_name, activity_2d) in enumerate(zip(region_names, activity_matrix)):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Heatmap(
                    z=activity_2d,
                    colorscale='viridis',
                    showscale=(i == 0),  # Only show scale for first plot
                    hovertemplate=f'{region_name}<br>Activity: %{{z:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='🧠 Neural Activity Heatmaps by Region',
            height=200 * rows,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        return fig

class ParameterMonitoringDashboard:
    """Real-time parameter monitoring dashboard."""
    
    def __init__(self):
        self.parameter_history = {
            'consciousness_level': [],
            'neural_activity': [],
            'connectivity_strength': [],
            'attention_coherence': [],
            'memory_usage': [],
            'cpu_usage': [],
            'training_loss': [],
            'biological_compliance': [],
            'timestamps': []
        }
        
        self.max_history = 1000
        
    def add_data_point(self, consciousness_level: float, neural_activity: Dict[str, np.ndarray],
                      connectivity_matrix: np.ndarray, attention_focus: Dict[str, float] = None,
                      training_metrics: Dict[str, float] = None):
        """Add new data point to monitoring history."""
        
        timestamp = datetime.now()
        
        # Calculate aggregate metrics
        avg_neural_activity = np.mean([np.mean(np.abs(activity)) for activity in neural_activity.values()])
        connectivity_strength = np.mean(connectivity_matrix[connectivity_matrix > 0])
        
        attention_coherence = 0.5
        if attention_focus:
            attention_coherence = np.mean(list(attention_focus.values()))
        
        # System metrics
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        # Training metrics
        training_loss = training_metrics.get('loss', 0.0) if training_metrics else 0.0
        biological_compliance = training_metrics.get('biological_compliance', 0.8) if training_metrics else 0.8
        
        # Add to history
        self.parameter_history['consciousness_level'].append(consciousness_level)
        self.parameter_history['neural_activity'].append(avg_neural_activity)
        self.parameter_history['connectivity_strength'].append(connectivity_strength)
        self.parameter_history['attention_coherence'].append(attention_coherence)
        self.parameter_history['memory_usage'].append(memory_usage)
        self.parameter_history['cpu_usage'].append(cpu_usage)
        self.parameter_history['training_loss'].append(training_loss)
        self.parameter_history['biological_compliance'].append(biological_compliance)
        self.parameter_history['timestamps'].append(timestamp)
        
        # Trim history if too long
        for key in self.parameter_history:
            if len(self.parameter_history[key]) > self.max_history:
                self.parameter_history[key] = self.parameter_history[key][-self.max_history:]
                
    def create_real_time_monitoring_dashboard(self) -> go.Figure:
        """Create comprehensive real-time monitoring dashboard."""
        
        if not self.parameter_history['timestamps']:
            # Return empty dashboard
            fig = go.Figure()
            fig.add_annotation(
                text="No data available yet...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Consciousness Level', 'Neural Activity', 'Training Loss',
                'Connectivity Strength', 'Attention Coherence', 'Biological Compliance',
                'System Memory Usage', 'CPU Usage', 'Parameter Correlation'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"type": "heatmap"}]
            ]
        )
        
        timestamps = self.parameter_history['timestamps']
        
        # 1. Consciousness Level
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.parameter_history['consciousness_level'],
                mode='lines+markers',
                name='Consciousness',
                line=dict(color='gold', width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # 2. Neural Activity
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.parameter_history['neural_activity'],
                mode='lines',
                name='Neural Activity',
                line=dict(color='blue', width=2),
                fill='tonexty'
            ),
            row=1, col=2
        )
        
        # 3. Training Loss
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.parameter_history['training_loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='red', width=2)
            ),
            row=1, col=3
        )
        
        # 4. Connectivity Strength
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.parameter_history['connectivity_strength'],
                mode='lines+markers',
                name='Connectivity',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # 5. Attention Coherence
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.parameter_history['attention_coherence'],
                mode='lines',
                name='Attention',
                line=dict(color='purple', width=2),
                fill='tozeroy'
            ),
            row=2, col=2
        )
        
        # 6. Biological Compliance
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.parameter_history['biological_compliance'],
                mode='lines+markers',
                name='Bio Compliance',
                line=dict(color='orange', width=2),
                marker=dict(size=5)
            ),
            row=2, col=3
        )
        
        # 7. Memory Usage
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.parameter_history['memory_usage'],
                mode='lines',
                name='Memory %',
                line=dict(color='cyan', width=2)
            ),
            row=3, col=1
        )
        
        # 8. CPU Usage
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.parameter_history['cpu_usage'],
                mode='lines',
                name='CPU %',
                line=dict(color='magenta', width=2)
            ),
            row=3, col=2
        )
        
        # 9. Parameter Correlation Heatmap
        if len(timestamps) > 5:  # Need some data for correlation
            param_data = np.array([
                self.parameter_history['consciousness_level'],
                self.parameter_history['neural_activity'],
                self.parameter_history['connectivity_strength'],
                self.parameter_history['attention_coherence'],
                self.parameter_history['biological_compliance']
            ])
            
            param_names = ['Consciousness', 'Neural Act.', 'Connectivity', 'Attention', 'Bio Compliance']
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(param_data)
            
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix,
                    x=param_names,
                    y=param_names,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.around(correlation_matrix, decimals=2),
                    texttemplate="%{text}",
                    textfont={"size":10},
                    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
                ),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text="🧠 Real-Time Brain Parameter Monitoring Dashboard",
            title_x=0.5,
            showlegend=False,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Update y-axis ranges for better visualization
        fig.update_yaxes(range=[0, 1], row=1, col=1)  # Consciousness
        fig.update_yaxes(range=[0, max(self.parameter_history['neural_activity'] + [1])], row=1, col=2)
        fig.update_yaxes(range=[0, 100], row=3, col=1)  # Memory %
        fig.update_yaxes(range=[0, 100], row=3, col=2)  # CPU %
        
        return fig
        
    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        if not self.parameter_history['timestamps']:
            return {}
            
        latest_metrics = {
            'timestamp': self.parameter_history['timestamps'][-1].isoformat(),
            'consciousness_level': self.parameter_history['consciousness_level'][-1],
            'neural_activity': self.parameter_history['neural_activity'][-1],
            'connectivity_strength': self.parameter_history['connectivity_strength'][-1],
            'attention_coherence': self.parameter_history['attention_coherence'][-1],
            'memory_usage': self.parameter_history['memory_usage'][-1],
            'cpu_usage': self.parameter_history['cpu_usage'][-1],
            'training_loss': self.parameter_history['training_loss'][-1],
            'biological_compliance': self.parameter_history['biological_compliance'][-1]
        }
        
        # Calculate trends (last 10 points)
        trends = {}
        for key in ['consciousness_level', 'neural_activity', 'connectivity_strength']:
            if len(self.parameter_history[key]) >= 10:
                recent_values = self.parameter_history[key][-10:]
                trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                trends[f'{key}_trend'] = 'increasing' if trend > 0.001 else 'decreasing' if trend < -0.001 else 'stable'
            else:
                trends[f'{key}_trend'] = 'insufficient_data'
        
        latest_metrics['trends'] = trends
        return latest_metrics

class VisualSimulationDashboard:
    """Main visual simulation dashboard orchestrator."""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or QUARK_ROOT
        self.dashboard_dir = self.base_dir / 'training' / 'visual_dashboard'
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # Initialize components
        self.neural_simulator = NeuralActivitySimulator()
        self.brain_visualizer = Real3DBrainVisualizer()
        self.parameter_monitor = ParameterMonitoringDashboard()
        
        # Simulation state
        self.is_running = False
        self.simulation_thread = None
        self.current_state = None
        
        # Configuration
        self.config = VisualizationConfig()
        
        # WebSocket connections
        self.active_connections = []
        
    def setup_logging(self):
        """Setup logging for visual dashboard."""
        self.logger = logging.getLogger("visual_simulation_dashboard")
        
    def start_simulation(self):
        """Start real-time simulation."""
        if self.is_running:
            return
            
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        
        self.logger.info("Started visual simulation dashboard")
        
    def stop_simulation(self):
        """Stop real-time simulation."""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join()
            
        self.logger.info("Stopped visual simulation dashboard")
        
    def _simulation_loop(self):
        """Main simulation loop."""
        while self.is_running:
            try:
                # Generate current consciousness and attention
                consciousness_level = 0.3 + 0.4 * np.sin(time.time() * 0.1) + 0.3 * np.random.random()
                consciousness_level = max(0, min(1, consciousness_level))
                
                attention_focus = {
                    'prefrontal': 0.3 + 0.4 * np.random.random(),
                    'visual': 0.2 + 0.3 * np.random.random(),
                    'working_memory': 0.4 + 0.3 * np.random.random()
                }
                
                # Update neural activity
                neural_activity = self.neural_simulator.update_neural_activity(
                    consciousness_level, attention_focus
                )
                
                # Get connectivity
                connectivity_matrix = self.neural_simulator.get_connectivity_matrix()
                
                # Create simulation state
                self.current_state = SimulationState(
                    timestamp=datetime.now(),
                    neural_activity=neural_activity,
                    connectivity_matrix=connectivity_matrix,
                    consciousness_level=consciousness_level,
                    attention_focus=attention_focus,
                    training_progress={'loss': 0.1 * np.random.random()},
                    biological_compliance=0.7 + 0.2 * np.random.random(),
                    energy_consumption=0.5 + 0.3 * np.random.random(),
                    memory_usage=psutil.virtual_memory().percent,
                    active_modules=self.neural_simulator.get_region_names()
                )
                
                # Update parameter monitoring
                self.parameter_monitor.add_data_point(
                    consciousness_level,
                    neural_activity,
                    connectivity_matrix,
                    attention_focus,
                    {'loss': self.current_state.training_progress['loss'],
                     'biological_compliance': self.current_state.biological_compliance}
                )
                
                # Sleep for update interval
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in simulation loop: {e}")
                time.sleep(1.0)
                
    def generate_comprehensive_dashboard(self) -> Dict[str, str]:
        """Generate all dashboard visualizations."""
        if not self.current_state:
            return {}
            
        generated_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # 1. 3D Brain Network
            brain_network_fig = self.brain_visualizer.create_3d_brain_network(
                self.current_state.neural_activity,
                self.current_state.connectivity_matrix,
                self.current_state.consciousness_level,
                self.current_state.attention_focus
            )
            
            brain_network_file = self.dashboard_dir / f'3d_brain_network_{timestamp}.html'
            brain_network_fig.write_html(str(brain_network_file))
            generated_files['3d_brain_network'] = str(brain_network_file)
            
            # 2. Neural Activity Heatmaps
            heatmap_fig = self.brain_visualizer.create_neural_activity_heatmap(
                self.current_state.neural_activity
            )
            
            heatmap_file = self.dashboard_dir / f'neural_heatmaps_{timestamp}.html'
            heatmap_fig.write_html(str(heatmap_file))
            generated_files['neural_heatmaps'] = str(heatmap_file)
            
            # 3. Real-time Parameter Monitoring
            monitoring_fig = self.parameter_monitor.create_real_time_monitoring_dashboard()
            
            monitoring_file = self.dashboard_dir / f'parameter_monitoring_{timestamp}.html'
            monitoring_fig.write_html(str(monitoring_file))
            generated_files['parameter_monitoring'] = str(monitoring_file)
            
            # 4. Combined Dashboard (all in one)
            combined_fig = self._create_combined_dashboard()
            
            combined_file = self.dashboard_dir / f'combined_dashboard_{timestamp}.html'
            combined_fig.write_html(str(combined_file))
            generated_files['combined_dashboard'] = str(combined_file)
            
            self.logger.info(f"Generated {len(generated_files)} dashboard files")
            
        except Exception as e:
            self.logger.error(f"Error generating dashboards: {e}")
            
        return generated_files
        
    def _create_combined_dashboard(self) -> go.Figure:
        """Create combined dashboard with all visualizations."""
        
        # Create main dashboard layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Real-Time Consciousness Level',
                'Neural Activity Distribution', 
                'Connectivity Network',
                'System Performance'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "histogram"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Consciousness level gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=self.current_state.consciousness_level,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Consciousness Level"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "gold"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "orange"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Neural activity distribution
        all_activities = []
        for activity in self.current_state.neural_activity.values():
            all_activities.extend(activity)
            
        fig.add_trace(
            go.Histogram(
                x=all_activities,
                nbinsx=30,
                name="Neural Activity",
                marker_color='blue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. Connectivity heatmap (simplified)
        connectivity_sample = self.current_state.connectivity_matrix[:10, :10]  # Sample for visibility
        region_names = self.neural_simulator.get_region_names()[:10]
        
        fig.add_trace(
            go.Heatmap(
                z=connectivity_sample,
                x=region_names,
                y=region_names,
                colorscale='viridis',
                showscale=False
            ),
            row=2, col=1
        )
        
        # 4. System performance over time
        if len(self.parameter_monitor.parameter_history['timestamps']) > 1:
            fig.add_trace(
                go.Scatter(
                    x=self.parameter_monitor.parameter_history['timestamps'][-50:],
                    y=self.parameter_monitor.parameter_history['consciousness_level'][-50:],
                    mode='lines',
                    name='Consciousness',
                    line=dict(color='gold')
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.parameter_monitor.parameter_history['timestamps'][-50:],
                    y=self.parameter_monitor.parameter_history['neural_activity'][-50:],
                    mode='lines',
                    name='Neural Activity',
                    line=dict(color='blue'),
                    yaxis='y2'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="🧠 Comprehensive Brain Simulation Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        return fig
        
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get current simulation summary."""
        if not self.current_state:
            return {'status': 'not_running'}
            
        # Calculate summary statistics
        neural_activities = list(self.current_state.neural_activity.values())
        avg_activity = np.mean([np.mean(np.abs(activity)) for activity in neural_activities])
        max_activity = np.max([np.max(np.abs(activity)) for activity in neural_activities])
        
        connectivity_stats = {
            'mean': float(np.mean(self.current_state.connectivity_matrix)),
            'max': float(np.max(self.current_state.connectivity_matrix)),
            'density': float(np.sum(self.current_state.connectivity_matrix > 0.1) / self.current_state.connectivity_matrix.size)
        }
        
        summary = {
            'status': 'running' if self.is_running else 'stopped',
            'timestamp': self.current_state.timestamp.isoformat(),
            'consciousness_level': self.current_state.consciousness_level,
            'neural_activity_stats': {
                'average': float(avg_activity),
                'maximum': float(max_activity),
                'num_regions': len(neural_activities)
            },
            'connectivity_stats': connectivity_stats,
            'attention_focus': self.current_state.attention_focus,
            'system_metrics': {
                'memory_usage': self.current_state.memory_usage,
                'energy_consumption': self.current_state.energy_consumption,
                'biological_compliance': self.current_state.biological_compliance
            },
            'active_modules': self.current_state.active_modules,
            'parameter_trends': self.parameter_monitor.get_current_metrics_summary().get('trends', {})
        }
        
        return summary

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visual Simulation Dashboard')
    parser.add_argument('--start', action='store_true', help='Start real-time simulation')
    parser.add_argument('--generate', action='store_true', help='Generate dashboard visualizations')
    parser.add_argument('--duration', type=int, default=60, help='Simulation duration in seconds')
    parser.add_argument('--output-dir', type=str, help='Output directory for dashboards')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize dashboard
    base_dir = Path(args.output_dir) if args.output_dir else None
    dashboard = VisualSimulationDashboard(base_dir)
    
    if args.start:
        # Start real-time simulation
        print("🧠 Starting Visual Simulation Dashboard...")
        dashboard.start_simulation()
        
        try:
            # Run for specified duration
            start_time = time.time()
            while time.time() - start_time < args.duration:
                # Generate dashboards periodically
                if int(time.time() - start_time) % 10 == 0:  # Every 10 seconds
                    generated_files = dashboard.generate_comprehensive_dashboard()
                    if generated_files:
                        print(f"📊 Generated dashboards: {list(generated_files.keys())}")
                
                # Print current status
                summary = dashboard.get_simulation_summary()
                if summary.get('status') == 'running':
                    print(f"Consciousness: {summary['consciousness_level']:.3f}, "
                          f"Neural Activity: {summary['neural_activity_stats']['average']:.3f}, "
                          f"Memory: {summary['system_metrics']['memory_usage']:.1f}%")
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        
        finally:
            dashboard.stop_simulation()
            
            # Generate final dashboards
            print("📊 Generating final dashboards...")
            final_dashboards = dashboard.generate_comprehensive_dashboard()
            
            print(f"✅ Simulation completed. Generated files:")
            for dashboard_type, filepath in final_dashboards.items():
                print(f"  {dashboard_type}: {filepath}")
                
    elif args.generate:
        # Generate static dashboards
        print("📊 Generating static dashboard visualizations...")
        
        # Start simulation briefly to get data
        dashboard.start_simulation()
        time.sleep(5)  # Let it collect some data
        
        generated_files = dashboard.generate_comprehensive_dashboard()
        dashboard.stop_simulation()
        
        print(f"✅ Generated {len(generated_files)} dashboard files:")
        for dashboard_type, filepath in generated_files.items():
            print(f"  {dashboard_type}: {filepath}")
    
    else:
        print("Please specify --start or --generate")
        print("Use --help for more options")

if __name__ == '__main__':
    main()

