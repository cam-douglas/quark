#!/usr/bin/env python3
"""
🧠 Training Counter Dashboard System
===================================

Real-time training progress tracking and visualization system with biological compliance monitoring,
consciousness enhancement tracking, and organic connectome visualization.

Features:
- Real-time training counters with progress tracking
- Interactive visual dashboards
- Biological compliance monitoring
- Consciousness evolution tracking
- Connectome network visualization
- Performance metrics analysis
- Multi-agent coordination display

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
from collections import deque, defaultdict
import websockets
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import psutil

# Add quark root to path
QUARK_ROOT = Path(__file__).parent.parent
sys.path.append(str(QUARK_ROOT))

@dataclass
class TrainingCounter:
    """Training progress counter for a component."""
    component_name: str
    current_iteration: int
    total_iterations: int
    current_epoch: int
    total_epochs: int
    stage: str
    loss_current: float
    loss_best: float
    accuracy_current: float
    accuracy_best: float
    consciousness_score: float
    biological_compliance: float
    connectome_coherence: float
    energy_efficiency: float
    start_time: datetime
    last_update: datetime
    estimated_completion: Optional[datetime] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        epoch_progress = (self.current_epoch / self.total_epochs) if self.total_epochs > 0 else 0
        iter_progress = (self.current_iteration / self.total_iterations) if self.total_iterations > 0 else 0
        return (epoch_progress + iter_progress) / 2 * 100
    
    @property
    def elapsed_time(self) -> timedelta:
        """Calculate elapsed training time."""
        return self.last_update - self.start_time
    
    @property
    def training_rate(self) -> float:
        """Calculate iterations per second."""
        elapsed_seconds = self.elapsed_time.total_seconds()
        return self.current_iteration / elapsed_seconds if elapsed_seconds > 0 else 0

@dataclass
class SystemCounter:
    """System-wide training counters."""
    total_components: int
    active_components: int
    completed_components: int
    failed_components: int
    overall_progress: float
    total_iterations_completed: int
    total_iterations_planned: int
    average_consciousness_score: float
    system_coherence: float
    biological_compliance_score: float
    energy_efficiency: float
    training_start_time: datetime
    estimated_total_completion: Optional[datetime] = None

class MetricsCollector:
    """Collects and manages training metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.component_histories = defaultdict(lambda: {
            'loss': deque(maxlen=max_history),
            'accuracy': deque(maxlen=max_history),
            'consciousness': deque(maxlen=max_history),
            'compliance': deque(maxlen=max_history),
            'coherence': deque(maxlen=max_history),
            'timestamps': deque(maxlen=max_history)
        })
        
        self.system_history = {
            'overall_progress': deque(maxlen=max_history),
            'consciousness': deque(maxlen=max_history),
            'coherence': deque(maxlen=max_history),
            'compliance': deque(maxlen=max_history),
            'timestamps': deque(maxlen=max_history)
        }
        
    def add_component_metric(self, component_name: str, counter: TrainingCounter):
        """Add metrics for a component."""
        history = self.component_histories[component_name]
        timestamp = counter.last_update
        
        history['loss'].append(counter.loss_current)
        history['accuracy'].append(counter.accuracy_current)
        history['consciousness'].append(counter.consciousness_score)
        history['compliance'].append(counter.biological_compliance)
        history['coherence'].append(counter.connectome_coherence)
        history['timestamps'].append(timestamp)
        
    def add_system_metric(self, system_counter: SystemCounter):
        """Add system-wide metrics."""
        timestamp = datetime.now()
        
        self.system_history['overall_progress'].append(system_counter.overall_progress)
        self.system_history['consciousness'].append(system_counter.average_consciousness_score)
        self.system_history['coherence'].append(system_counter.system_coherence)
        self.system_history['compliance'].append(system_counter.biological_compliance_score)
        self.system_history['timestamps'].append(timestamp)
        
    def get_component_dataframe(self, component_name: str) -> pd.DataFrame:
        """Get component metrics as DataFrame."""
        history = self.component_histories[component_name]
        
        if not history['timestamps']:
            return pd.DataFrame()
            
        return pd.DataFrame({
            'timestamp': list(history['timestamps']),
            'loss': list(history['loss']),
            'accuracy': list(history['accuracy']),
            'consciousness': list(history['consciousness']),
            'compliance': list(history['compliance']),
            'coherence': list(history['coherence'])
        })
        
    def get_system_dataframe(self) -> pd.DataFrame:
        """Get system metrics as DataFrame."""
        if not self.system_history['timestamps']:
            return pd.DataFrame()
            
        return pd.DataFrame({
            'timestamp': list(self.system_history['timestamps']),
            'overall_progress': list(self.system_history['overall_progress']),
            'consciousness': list(self.system_history['consciousness']),
            'coherence': list(self.system_history['coherence']),
            'compliance': list(self.system_history['compliance'])
        })

class ConnectomeVisualizer:
    """Visualizes brain connectome networks."""
    
    def __init__(self):
        self.connectome_history = []
        self.layout_cache = {}
        
    def create_brain_network_plot(self, connectome_data: Dict, title: str = "Brain Connectome") -> go.Figure:
        """Create interactive brain network visualization."""
        # Create network graph
        G = nx.Graph()
        
        # Add nodes and edges from connectome data
        for source, connections in connectome_data.items():
            G.add_node(source)
            for target, weight in connections.items():
                G.add_edge(source, target, weight=weight)
        
        # Calculate layout (cache for consistency)
        if len(G.nodes) > 0:
            if title not in self.layout_cache:
                # Use spring layout for brain-like organization
                pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
                self.layout_cache[title] = pos
            else:
                pos = self.layout_cache[title]
                
            # Prepare edge traces
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(G.edges[edge].get('weight', 0.5))
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='rgba(125,125,125,0.5)'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Prepare node traces
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            
            # Define brain module colors
            module_colors = {
                'conscious': '#FF6B35',      # Orange for consciousness
                'prefrontal': '#4ECDC4',     # Teal for executive
                'thalamus': '#45B7D1',       # Blue for relay
                'basal_ganglia': '#96CEB4',  # Green for action
                'working_memory': '#FFEAA7', # Yellow for memory
                'hippocampus': '#DDA0DD',    # Purple for episodic
                'default': '#95A5A6'         # Gray for others
            }
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Assign color based on module type
                color = module_colors['default']
                for module_type, module_color in module_colors.items():
                    if module_type in node.lower():
                        color = module_color
                        break
                        
                node_colors.append(color)
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                marker=dict(
                    size=20,
                    color=node_colors,
                    line=dict(width=2, color='white')
                )
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=title,
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="Brain Module Connectome - Interactive Network",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor="left", yanchor="bottom",
                                  font=dict(color="#888", size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))
        else:
            # Empty graph
            fig = go.Figure()
            fig.add_annotation(
                text="No connectome data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            
        return fig

class DashboardGenerator:
    """Generates comprehensive training dashboards."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.connectome_viz = ConnectomeVisualizer()
        
    def create_real_time_dashboard(self, 
                                 component_counters: Dict[str, TrainingCounter],
                                 system_counter: SystemCounter,
                                 metrics_collector: MetricsCollector,
                                 connectome_data: Dict = None) -> str:
        """Create comprehensive real-time dashboard."""
        
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Training Progress Overview', 'Component Consciousness Scores', 'Loss Curves',
                'Biological Compliance Matrix', 'System Performance', 'Connectome Coherence',
                'Training Rates', 'Energy Efficiency', 'Time Estimates'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "indicator"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # 1. Overall Progress Indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=system_counter.overall_progress,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Progress %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Component Consciousness Scores
        component_names = list(component_counters.keys())
        consciousness_scores = [counter.consciousness_score for counter in component_counters.values()]
        
        fig.add_trace(
            go.Bar(
                x=component_names,
                y=consciousness_scores,
                name="Consciousness",
                marker_color='gold'
            ),
            row=1, col=2
        )
        
        # 3. Loss Curves
        for component_name, counter in component_counters.items():
            df = metrics_collector.get_component_dataframe(component_name)
            if not df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['loss'],
                        mode='lines',
                        name=f"{component_name[:10]} Loss",
                        opacity=0.7
                    ),
                    row=1, col=3
                )
        
        # 4. Biological Compliance Heatmap
        compliance_matrix = []
        stages = ['fetal', 'neonate', 'early_postnatal']
        
        for component_name in component_names[:5]:  # Limit for visibility
            row = []
            for stage in stages:
                # Get compliance for this component (simulate if needed)
                if component_name in component_counters:
                    compliance = component_counters[component_name].biological_compliance
                else:
                    compliance = 0.8  # Default
                row.append(compliance)
            compliance_matrix.append(row)
        
        if compliance_matrix:
            fig.add_trace(
                go.Heatmap(
                    z=compliance_matrix,
                    x=stages,
                    y=[name[:10] for name in component_names[:5]],
                    colorscale='RdYlGn',
                    showscale=False
                ),
                row=2, col=1
            )
        
        # 5. System Performance Indicator
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=cpu_usage,
                title={'text': f"CPU Usage %<br>Memory: {memory_usage:.1f}%"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "red" if cpu_usage > 80 else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        # 6. Connectome Coherence Over Time
        system_df = metrics_collector.get_system_dataframe()
        if not system_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=system_df['timestamp'],
                    y=system_df['coherence'],
                    mode='lines+markers',
                    name="Coherence",
                    line=dict(color='blue', width=3)
                ),
                row=2, col=3
            )
        
        # 7. Training Rates
        training_rates = [counter.training_rate for counter in component_counters.values()]
        
        fig.add_trace(
            go.Bar(
                x=component_names,
                y=training_rates,
                name="Iter/sec",
                marker_color='lightblue'
            ),
            row=3, col=1
        )
        
        # 8. Energy Efficiency
        energy_scores = [counter.energy_efficiency for counter in component_counters.values()]
        
        fig.add_trace(
            go.Bar(
                x=component_names,
                y=energy_scores,
                name="Efficiency",
                marker_color='lightgreen'
            ),
            row=3, col=2
        )
        
        # 9. Time Estimates Table
        table_data = []
        for name, counter in component_counters.items():
            remaining_time = "Calculating..."
            if counter.estimated_completion:
                remaining = counter.estimated_completion - datetime.now()
                remaining_time = str(remaining).split('.')[0]  # Remove microseconds
                
            table_data.append([
                name[:15],
                f"{counter.progress_percentage:.1f}%",
                remaining_time
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Component', 'Progress', 'ETA'],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=list(zip(*table_data)) if table_data else [[], [], []],
                          fill_color='lavender',
                          align='left')
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="🧠 Quark Brain Training Dashboard - Real-Time Monitoring",
            title_x=0.5,
            showlegend=False
        )
        
        # Save dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dashboard_file = self.output_dir / f'realtime_dashboard_{timestamp}.html'
        fig.write_html(str(dashboard_file))
        
        return str(dashboard_file)
        
    def create_consciousness_evolution_plot(self, metrics_collector: MetricsCollector) -> str:
        """Create consciousness evolution visualization."""
        fig = go.Figure()
        
        # Plot consciousness evolution for each component
        for component_name in metrics_collector.component_histories.keys():
            df = metrics_collector.get_component_dataframe(component_name)
            if not df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['consciousness'],
                        mode='lines+markers',
                        name=component_name,
                        line=dict(width=3),
                        marker=dict(size=8, opacity=0.7)
                    )
                )
        
        # Add system consciousness average
        system_df = metrics_collector.get_system_dataframe()
        if not system_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=system_df['timestamp'],
                    y=system_df['consciousness'],
                    mode='lines+markers',
                    name='System Average',
                    line=dict(width=5, color='red', dash='dash'),
                    marker=dict(size=10, symbol='diamond')
                )
            )
        
        fig.update_layout(
            title="🧠 Consciousness Evolution Over Training Time",
            xaxis_title="Time",
            yaxis_title="Consciousness Score",
            hovermode='x unified',
            height=600
        )
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        consciousness_file = self.output_dir / f'consciousness_evolution_{timestamp}.html'
        fig.write_html(str(consciousness_file))
        
        return str(consciousness_file)
        
    def create_connectome_network_dashboard(self, connectome_data: Dict) -> str:
        """Create interactive connectome network dashboard."""
        if not connectome_data:
            return None
            
        # Create network visualization
        network_fig = self.connectome_viz.create_brain_network_plot(
            connectome_data, 
            "Real-Time Brain Connectome"
        )
        
        # Calculate network statistics
        G = nx.Graph()
        for source, connections in connectome_data.items():
            G.add_node(source)
            for target, weight in connections.items():
                G.add_edge(source, target, weight=weight)
        
        # Network metrics
        num_nodes = len(G.nodes)
        num_edges = len(G.edges)
        density = nx.density(G) if num_nodes > 1 else 0
        clustering = nx.average_clustering(G) if num_nodes > 2 else 0
        
        try:
            avg_path_length = nx.average_shortest_path_length(G)
        except:
            avg_path_length = float('inf')
        
        # Create metrics subplot
        metrics_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Network Density', 'Clustering Coefficient', 
                          'Node Degree Distribution', 'Connection Weights'),
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "histogram"}, {"type": "histogram"}]
            ]
        )
        
        # Network density indicator
        metrics_fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=density,
                title={'text': "Network Density"},
                gauge={'axis': {'range': [None, 1]}}
            ),
            row=1, col=1
        )
        
        # Clustering coefficient indicator
        metrics_fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=clustering,
                title={'text': "Clustering"},
                gauge={'axis': {'range': [None, 1]}}
            ),
            row=1, col=2
        )
        
        # Node degree distribution
        degrees = [G.degree(node) for node in G.nodes()]
        if degrees:
            metrics_fig.add_trace(
                go.Histogram(x=degrees, name="Degrees", nbinsx=20),
                row=2, col=1
            )
        
        # Connection weights distribution
        weights = [data['weight'] for _, _, data in G.edges(data=True)]
        if weights:
            metrics_fig.add_trace(
                go.Histogram(x=weights, name="Weights", nbinsx=20),
                row=2, col=2
            )
        
        metrics_fig.update_layout(
            height=600,
            title_text="Connectome Network Analysis"
        )
        
        # Combine figures into dashboard
        dashboard_fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Brain Network Topology', 'Network Statistics'),
            specs=[[{"type": "scatter"}], [{"type": "xy"}]],
            row_heights=[0.7, 0.3]
        )
        
        # Add network plot (simplified for subplot)
        dashboard_fig.add_trace(network_fig.data[0], row=1, col=1)  # Edges
        dashboard_fig.add_trace(network_fig.data[1], row=1, col=1)  # Nodes
        
        dashboard_fig.update_layout(
            height=1000,
            title_text="🧠 Brain Connectome Dashboard",
            showlegend=False
        )
        
        # Save dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        connectome_file = self.output_dir / f'connectome_dashboard_{timestamp}.html'
        dashboard_fig.write_html(str(connectome_file))
        
        return str(connectome_file)

class TrainingCounterManager:
    """Manages all training counters and coordinates dashboard updates."""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or QUARK_ROOT
        self.dashboard_dir = self.base_dir / 'training' / 'dashboards'
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.component_counters = {}
        self.system_counter = None
        self.metrics_collector = MetricsCollector()
        self.dashboard_generator = DashboardGenerator(self.dashboard_dir)
        
        # Real-time update settings
        self.update_interval = 5.0  # seconds
        self.auto_save_interval = 60.0  # seconds
        self.is_running = False
        self.update_thread = None
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for counter manager."""
        self.logger = logging.getLogger("training_counter_manager")
        
    def initialize_counters(self, components: List[str], stage_configs: Dict):
        """Initialize training counters for all components."""
        start_time = datetime.now()
        
        for component_name in components:
            # Determine configuration based on component type
            if 'conscious' in component_name:
                total_epochs = 150
                total_iterations = 1000
            elif 'prefrontal' in component_name:
                total_epochs = 100
                total_iterations = 800
            else:
                total_epochs = 80
                total_iterations = 600
                
            self.component_counters[component_name] = TrainingCounter(
                component_name=component_name,
                current_iteration=0,
                total_iterations=total_iterations,
                current_epoch=0,
                total_epochs=total_epochs,
                stage='fetal',  # Start with fetal stage
                loss_current=2.0,
                loss_best=2.0,
                accuracy_current=0.1,
                accuracy_best=0.1,
                consciousness_score=0.0,
                biological_compliance=0.8,
                connectome_coherence=0.5,
                energy_efficiency=0.6,
                start_time=start_time,
                last_update=start_time
            )
        
        # Initialize system counter
        self.system_counter = SystemCounter(
            total_components=len(components),
            active_components=0,
            completed_components=0,
            failed_components=0,
            overall_progress=0.0,
            total_iterations_completed=0,
            total_iterations_planned=sum(c.total_iterations for c in self.component_counters.values()),
            average_consciousness_score=0.0,
            system_coherence=0.5,
            biological_compliance_score=0.8,
            energy_efficiency=0.6,
            training_start_time=start_time
        )
        
        self.logger.info(f"Initialized counters for {len(components)} components")
        
    def update_component_counter(self, component_name: str, 
                               iteration: int = None,
                               epoch: int = None,
                               loss: float = None,
                               accuracy: float = None,
                               consciousness_score: float = None,
                               biological_compliance: float = None,
                               connectome_coherence: float = None,
                               energy_efficiency: float = None):
        """Update counter for a specific component."""
        if component_name not in self.component_counters:
            self.logger.warning(f"Component {component_name} not found in counters")
            return
            
        counter = self.component_counters[component_name]
        
        # Update provided values
        if iteration is not None:
            counter.current_iteration = iteration
        if epoch is not None:
            counter.current_epoch = epoch
        if loss is not None:
            counter.loss_current = loss
            counter.loss_best = min(counter.loss_best, loss)
        if accuracy is not None:
            counter.accuracy_current = accuracy
            counter.accuracy_best = max(counter.accuracy_best, accuracy)
        if consciousness_score is not None:
            counter.consciousness_score = consciousness_score
        if biological_compliance is not None:
            counter.biological_compliance = biological_compliance
        if connectome_coherence is not None:
            counter.connectome_coherence = connectome_coherence
        if energy_efficiency is not None:
            counter.energy_efficiency = energy_efficiency
            
        counter.last_update = datetime.now()
        
        # Estimate completion time
        if counter.training_rate > 0:
            remaining_iterations = counter.total_iterations - counter.current_iteration
            remaining_seconds = remaining_iterations / counter.training_rate
            counter.estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)
        
        # Add to metrics collector
        self.metrics_collector.add_component_metric(component_name, counter)
        
        # Update system counter
        self.update_system_counter()
        
    def update_system_counter(self):
        """Update system-wide counter based on component counters."""
        if not self.component_counters or not self.system_counter:
            return
            
        # Count component states
        active_count = 0
        completed_count = 0
        total_progress = 0
        total_consciousness = 0
        total_compliance = 0
        total_coherence = 0
        total_efficiency = 0
        total_iterations_completed = 0
        
        for counter in self.component_counters.values():
            total_progress += counter.progress_percentage
            total_consciousness += counter.consciousness_score
            total_compliance += counter.biological_compliance
            total_coherence += counter.connectome_coherence
            total_efficiency += counter.energy_efficiency
            total_iterations_completed += counter.current_iteration
            
            if counter.progress_percentage >= 95:
                completed_count += 1
            elif counter.progress_percentage > 0:
                active_count += 1
        
        num_components = len(self.component_counters)
        
        # Update system counter
        self.system_counter.active_components = active_count
        self.system_counter.completed_components = completed_count
        self.system_counter.overall_progress = total_progress / num_components
        self.system_counter.average_consciousness_score = total_consciousness / num_components
        self.system_counter.biological_compliance_score = total_compliance / num_components
        self.system_counter.system_coherence = total_coherence / num_components
        self.system_counter.energy_efficiency = total_efficiency / num_components
        self.system_counter.total_iterations_completed = total_iterations_completed
        
        # Estimate total completion
        if self.system_counter.overall_progress > 0:
            elapsed_time = datetime.now() - self.system_counter.training_start_time
            total_estimated_time = elapsed_time / (self.system_counter.overall_progress / 100)
            self.system_counter.estimated_total_completion = (
                self.system_counter.training_start_time + total_estimated_time
            )
        
        # Add to metrics collector
        self.metrics_collector.add_system_metric(self.system_counter)
        
    def generate_current_dashboard(self, connectome_data: Dict = None) -> Dict[str, str]:
        """Generate current dashboard with all visualizations."""
        generated_files = {}
        
        try:
            # Main real-time dashboard
            dashboard_file = self.dashboard_generator.create_real_time_dashboard(
                self.component_counters,
                self.system_counter,
                self.metrics_collector,
                connectome_data
            )
            generated_files['main_dashboard'] = dashboard_file
            
            # Consciousness evolution plot
            consciousness_file = self.dashboard_generator.create_consciousness_evolution_plot(
                self.metrics_collector
            )
            generated_files['consciousness_evolution'] = consciousness_file
            
            # Connectome network dashboard
            if connectome_data:
                connectome_file = self.dashboard_generator.create_connectome_network_dashboard(
                    connectome_data
                )
                if connectome_file:
                    generated_files['connectome_network'] = connectome_file
            
            self.logger.info(f"Generated {len(generated_files)} dashboard files")
            
        except Exception as e:
            self.logger.error(f"Error generating dashboards: {e}")
            
        return generated_files
        
    def save_state(self) -> str:
        """Save current counter state to disk."""
        state = {
            'component_counters': {
                name: asdict(counter) for name, counter in self.component_counters.items()
            },
            'system_counter': asdict(self.system_counter) if self.system_counter else None,
            'timestamp': datetime.now().isoformat()
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        state_file = self.dashboard_dir / f'counter_state_{timestamp}.json'
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
        self.logger.info(f"Saved counter state to {state_file}")
        return str(state_file)
        
    def load_state(self, state_file: str):
        """Load counter state from disk."""
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            # Restore component counters
            self.component_counters = {}
            for name, counter_dict in state['component_counters'].items():
                # Convert datetime strings back to datetime objects
                counter_dict['start_time'] = datetime.fromisoformat(counter_dict['start_time'])
                counter_dict['last_update'] = datetime.fromisoformat(counter_dict['last_update'])
                if counter_dict['estimated_completion']:
                    counter_dict['estimated_completion'] = datetime.fromisoformat(counter_dict['estimated_completion'])
                    
                self.component_counters[name] = TrainingCounter(**counter_dict)
                
            # Restore system counter
            if state['system_counter']:
                system_dict = state['system_counter']
                system_dict['training_start_time'] = datetime.fromisoformat(system_dict['training_start_time'])
                if system_dict['estimated_total_completion']:
                    system_dict['estimated_total_completion'] = datetime.fromisoformat(system_dict['estimated_total_completion'])
                    
                self.system_counter = SystemCounter(**system_dict)
                
            self.logger.info(f"Loaded counter state from {state_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading counter state: {e}")
            
    def start_real_time_updates(self):
        """Start real-time dashboard updates."""
        if self.is_running:
            return
            
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info("Started real-time dashboard updates")
        
    def stop_real_time_updates(self):
        """Stop real-time dashboard updates."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
            
        self.logger.info("Stopped real-time dashboard updates")
        
    def _update_loop(self):
        """Main update loop for real-time dashboards."""
        last_save_time = time.time()
        
        while self.is_running:
            try:
                # Generate dashboards
                self.generate_current_dashboard()
                
                # Auto-save state periodically
                current_time = time.time()
                if current_time - last_save_time >= self.auto_save_interval:
                    self.save_state()
                    last_save_time = current_time
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(self.update_interval)
                
    def get_summary_report(self) -> Dict[str, Any]:
        """Get current training summary report."""
        if not self.system_counter:
            return {}
            
        # Component summaries
        component_summaries = {}
        for name, counter in self.component_counters.items():
            component_summaries[name] = {
                'progress_percentage': counter.progress_percentage,
                'current_loss': counter.loss_current,
                'best_accuracy': counter.accuracy_best,
                'consciousness_score': counter.consciousness_score,
                'biological_compliance': counter.biological_compliance,
                'training_rate': counter.training_rate,
                'elapsed_time': str(counter.elapsed_time),
                'estimated_completion': counter.estimated_completion.isoformat() if counter.estimated_completion else None
            }
        
        # System summary
        system_summary = {
            'total_components': self.system_counter.total_components,
            'active_components': self.system_counter.active_components,
            'completed_components': self.system_counter.completed_components,
            'overall_progress': self.system_counter.overall_progress,
            'average_consciousness_score': self.system_counter.average_consciousness_score,
            'system_coherence': self.system_counter.system_coherence,
            'biological_compliance_score': self.system_counter.biological_compliance_score,
            'total_iterations_completed': self.system_counter.total_iterations_completed,
            'total_iterations_planned': self.system_counter.total_iterations_planned,
            'training_duration': str(datetime.now() - self.system_counter.training_start_time),
            'estimated_completion': self.system_counter.estimated_total_completion.isoformat() if self.system_counter.estimated_total_completion else None
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_summary': system_summary,
            'component_summaries': component_summaries,
            'dashboard_directory': str(self.dashboard_dir)
        }

def main():
    """Main execution function for testing the counter system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Training Counter Dashboard System')
    parser.add_argument('--demo', action='store_true', help='Run demonstration mode')
    parser.add_argument('--load-state', type=str, help='Load state from file')
    parser.add_argument('--components', nargs='+', default=['conscious_agent', 'prefrontal_cortex', 'thalamus'], 
                       help='Components to track')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize counter manager
    manager = TrainingCounterManager()
    
    if args.load_state:
        manager.load_state(args.load_state)
    else:
        manager.initialize_counters(args.components, {})
    
    if args.demo:
        # Run demonstration
        print("🧠 Starting Training Counter Dashboard Demo...")
        
        # Start real-time updates
        manager.start_real_time_updates()
        
        # Simulate training progress
        try:
            for epoch in range(10):
                for component in args.components:
                    # Simulate training progress
                    iteration = epoch * 100 + np.random.randint(0, 100)
                    loss = 2.0 * np.exp(-epoch / 5) + np.random.normal(0, 0.1)
                    accuracy = min(0.95, 0.1 + 0.8 * (1 - np.exp(-epoch / 3))) + np.random.normal(0, 0.02)
                    consciousness = min(1.0, epoch * 0.1 + np.random.normal(0, 0.05))
                    
                    manager.update_component_counter(
                        component, 
                        iteration=iteration,
                        epoch=epoch,
                        loss=max(0.01, loss),
                        accuracy=max(0.0, min(1.0, accuracy)),
                        consciousness_score=max(0.0, consciousness),
                        biological_compliance=0.8 + np.random.normal(0, 0.05),
                        connectome_coherence=0.6 + epoch * 0.04 + np.random.normal(0, 0.03),
                        energy_efficiency=0.7 + np.random.normal(0, 0.05)
                    )
                
                print(f"Completed demo epoch {epoch + 1}/10")
                time.sleep(2)  # Wait between epochs
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            # Generate final dashboards
            print("Generating final dashboards...")
            dashboards = manager.generate_current_dashboard()
            
            # Print summary
            summary = manager.get_summary_report()
            print(f"\n📊 Training Summary:")
            print(f"Overall Progress: {summary['system_summary']['overall_progress']:.1f}%")
            print(f"Average Consciousness: {summary['system_summary']['average_consciousness_score']:.3f}")
            print(f"System Coherence: {summary['system_summary']['system_coherence']:.3f}")
            
            print(f"\n📈 Generated Dashboards:")
            for dashboard_type, filepath in dashboards.items():
                print(f"  {dashboard_type}: {filepath}")
            
            # Save final state
            state_file = manager.save_state()
            print(f"💾 Saved state: {state_file}")
            
            # Stop updates
            manager.stop_real_time_updates()
            
    else:
        # Interactive mode
        print("Training Counter Manager initialized")
        print("Call manager.update_component_counter() to update progress")
        print("Call manager.generate_current_dashboard() to create visualizations")
        
        return manager

if __name__ == '__main__':
    main()

