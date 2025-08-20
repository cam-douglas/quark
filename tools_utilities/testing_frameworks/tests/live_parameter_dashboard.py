#!/usr/bin/env python3
"""
LIVE PARAMETER DASHBOARD: Real-time monitoring of brain simulation parameters
Purpose: Provide live visualization of neural activity, sleep cycles, memory consolidation, and system performance
Inputs: Real-time simulation data and system metrics
Outputs: Interactive HTML dashboard with live updates
Seeds: Random seed for reproducible simulations
Deps: plotly, dash, numpy, psutil, threading, time
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import psutil
import threading
import time
from pathlib import Path
import json
from datetime import datetime
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class LiveParameterMonitor:
    """Real-time parameter monitoring for brain simulation"""
    
    def __init__(self):
        self.output_dir = Path("tests/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.neural_data = {
            'spike_times': [],
            'spike_neurons': [],
            'firing_rates': [],
            'timestamps': []
        }
        
        self.sleep_data = {
            'phases': [],
            'brain_waves': [],
            'consolidation': [],
            'timestamps': []
        }
        
        self.performance_data = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'network_io': [],
            'timestamps': []
        }
        
        self.memory_data = {
            'episodic': [],
            'semantic': [],
            'procedural': [],
            'working_memory': [],
            'timestamps': []
        }
        
        # Simulation parameters
        self.simulation_time = 0
        self.num_neurons = 100
        self.update_interval = 1.0  # seconds
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.setup_dashboard()
        
    def setup_dashboard(self):
        """Setup the Dash dashboard layout"""
        self.app.layout = html.Div([
            html.H1("🧠 Live Brain Simulation Parameter Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Control Panel
            html.Div([
                html.Button("Start Simulation", id="start-btn", n_clicks=0, 
                           style={'margin': '10px', 'padding': '10px 20px'}),
                html.Button("Stop Simulation", id="stop-btn", n_clicks=0, 
                           style={'margin': '10px', 'padding': '10px 20px'}),
                html.Button("Reset", id="reset-btn", n_clicks=0, 
                           style={'margin': '10px', 'padding': '10px 20px'}),
                html.Div(id="status-display", style={'margin': '10px', 'fontWeight': 'bold'})
            ], style={'textAlign': 'center', 'margin': '20px'}),
            
            # Parameter Controls
            html.Div([
                html.Label("Update Interval (seconds):"),
                dcc.Slider(id="interval-slider", min=0.1, max=5.0, step=0.1, value=1.0,
                          marks={0.1: '0.1s', 1.0: '1s', 2.0: '2s', 5.0: '5s'}),
                html.Label("Number of Neurons:"),
                dcc.Slider(id="neurons-slider", min=10, max=500, step=10, value=100,
                          marks={10: '10', 100: '100', 250: '250', 500: '500'})
            ], style={'margin': '20px', 'padding': '20px', 'border': '1px solid #ddd'}),
            
            # Real-time Metrics
            html.Div([
                html.H3("📊 Real-time System Metrics", style={'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        html.H4("CPU Usage"),
                        html.Div(id="cpu-display", style={'fontSize': '24px', 'color': '#e74c3c'})
                    ], style={'textAlign': 'center', 'margin': '10px'}),
                    html.Div([
                        html.H4("Memory Usage"),
                        html.Div(id="memory-display", style={'fontSize': '24px', 'color': '#3498db'})
                    ], style={'textAlign': 'center', 'margin': '10px'}),
                    html.Div([
                        html.H4("Simulation Time"),
                        html.Div(id="time-display", style={'fontSize': '24px', 'color': '#27ae60'})
                    ], style={'textAlign': 'center', 'margin': '10px'})
                ], style={'display': 'flex', 'justifyContent': 'space-around'})
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa'}),
            
            # Neural Activity Chart
            html.Div([
                html.H3("⚡ Neural Activity Monitor", style={'textAlign': 'center'}),
                dcc.Graph(id="neural-activity-chart", style={'height': '400px'})
            ], style={'margin': '20px'}),
            
            # Sleep Cycles Chart
            html.Div([
                html.H3("💤 Sleep-Wake Cycles", style={'textAlign': 'center'}),
                dcc.Graph(id="sleep-cycles-chart", style={'height': '400px'})
            ], style={'margin': '20px'}),
            
            # Memory Consolidation Chart
            html.Div([
                html.H3("🧠 Memory Consolidation", style={'textAlign': 'center'}),
                dcc.Graph(id="memory-chart", style={'height': '400px'})
            ], style={'margin': '20px'}),
            
            # Performance Metrics Chart
            html.Div([
                html.H3("🚀 Performance Metrics", style={'textAlign': 'center'}),
                dcc.Graph(id="performance-chart", style={'height': '400px'})
            ], style={'margin': '20px'}),
            
            # Data Export
            html.Div([
                html.H3("📁 Data Export", style={'textAlign': 'center'}),
                html.Button("Export Dashboard", id="export-btn", n_clicks=0,
                           style={'margin': '10px', 'padding': '10px 20px'}),
                html.Div(id="export-status", style={'margin': '10px'})
            ], style={'textAlign': 'center', 'margin': '20px'}),
            
            # Hidden div for storing data
            html.Div(id="data-store", style={'display': 'none'}),
            
            # Interval component for updates
            dcc.Interval(id="interval-component", interval=1000, n_intervals=0)
        ])
        
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup Dash callbacks for interactivity"""
        
        @self.app.callback(
            Output("status-display", "children"),
            [Input("start-btn", "n_clicks"),
             Input("stop-btn", "n_clicks"),
             Input("reset-btn", "n_clicks")]
        )
        def update_status(start_clicks, stop_clicks, reset_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return "Ready to start simulation"
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == "start-btn":
                self.start_simulation()
                return "🟢 Simulation running..."
            elif button_id == "stop-btn":
                self.stop_simulation()
                return "🔴 Simulation stopped"
            elif button_id == "reset-btn":
                self.reset_simulation()
                return "🔄 Simulation reset"
            
            return "Ready to start simulation"
        
        @self.app.callback(
            [Output("neural-activity-chart", "figure"),
             Output("sleep-cycles-chart", "figure"),
             Output("memory-chart", "figure"),
             Output("performance-chart", "figure")],
            [Input("interval-component", "n_intervals")]
        )
        def update_charts(n_intervals):
            if not hasattr(self, 'simulation_running') or not self.simulation_running:
                return self.create_empty_charts()
            
            return [
                self.create_neural_activity_chart(),
                self.create_sleep_cycles_chart(),
                self.create_memory_chart(),
                self.create_performance_chart()
            ]
        
        @self.app.callback(
            [Output("cpu-display", "children"),
             Output("memory-display", "children"),
             Output("time-display", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_metrics(n_intervals):
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            return [
                f"{cpu_percent:.1f}%",
                f"{memory_percent:.1f}%",
                f"{self.simulation_time:.1f}s"
            ]
        
        @self.app.callback(
            Output("export-status", "children"),
            [Input("export-btn", "n_clicks")]
        )
        def export_dashboard(n_clicks):
            if n_clicks > 0:
                return self.export_dashboard_data()
            return ""
    
    def create_empty_charts(self):
        """Create empty charts when simulation is not running"""
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Start simulation to see live data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        empty_fig.update_layout(title="No Data Available")
        
        return [empty_fig] * 4
    
    def create_neural_activity_chart(self):
        """Create neural activity visualization"""
        fig = go.Figure()
        
        if self.neural_data['spike_times']:
            fig.add_trace(go.Scatter(
                x=self.neural_data['spike_times'],
                y=self.neural_data['spike_neurons'],
                mode='markers',
                marker=dict(size=5, color='red', opacity=0.7),
                name='Neural Spikes'
            ))
        
        fig.update_layout(
            title="Real-time Neural Spike Activity",
            xaxis_title="Time (ms)",
            yaxis_title="Neuron ID",
            height=400
        )
        
        return fig
    
    def create_sleep_cycles_chart(self):
        """Create sleep cycles visualization"""
        fig = go.Figure()
        
        if self.sleep_data['timestamps']:
            # Sleep phases
            fig.add_trace(go.Scatter(
                x=self.sleep_data['timestamps'],
                y=self.sleep_data['phases'],
                mode='lines+markers',
                name='Sleep Phase',
                line=dict(color='blue', width=3)
            ))
            
            # Brain waves
            fig.add_trace(go.Scatter(
                x=self.sleep_data['timestamps'],
                y=self.sleep_data['brain_waves'],
                mode='lines',
                name='Brain Waves',
                line=dict(color='orange', width=2),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title="Sleep-Wake Cycles & Brain Waves",
            xaxis_title="Time (s)",
            yaxis_title="Sleep Phase",
            yaxis2=dict(title="Brain Wave Frequency (Hz)", overlaying='y', side='right'),
            height=400
        )
        
        return fig
    
    def create_memory_chart(self):
        """Create memory consolidation visualization"""
        fig = go.Figure()
        
        if self.memory_data['timestamps']:
            fig.add_trace(go.Scatter(
                x=self.memory_data['timestamps'],
                y=self.memory_data['episodic'],
                mode='lines',
                name='Episodic Memory',
                line=dict(color='red', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=self.memory_data['timestamps'],
                y=self.memory_data['semantic'],
                mode='lines',
                name='Semantic Memory',
                line=dict(color='green', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=self.memory_data['timestamps'],
                y=self.memory_data['procedural'],
                mode='lines',
                name='Procedural Memory',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=self.memory_data['timestamps'],
                y=self.memory_data['working_memory'],
                mode='lines',
                name='Working Memory',
                line=dict(color='purple', width=2)
            ))
        
        fig.update_layout(
            title="Memory Consolidation Types",
            xaxis_title="Time (s)",
            yaxis_title="Consolidation Strength",
            height=400
        )
        
        return fig
    
    def create_performance_chart(self):
        """Create performance metrics visualization"""
        fig = go.Figure()
        
        if self.performance_data['timestamps']:
            fig.add_trace(go.Scatter(
                x=self.performance_data['timestamps'],
                y=self.performance_data['cpu_usage'],
                mode='lines',
                name='CPU Usage',
                line=dict(color='red', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=self.performance_data['timestamps'],
                y=self.performance_data['memory_usage'],
                mode='lines',
                name='Memory Usage',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=self.performance_data['timestamps'],
                y=self.performance_data['gpu_usage'],
                mode='lines',
                name='GPU Usage',
                line=dict(color='green', width=2)
            ))
        
        fig.update_layout(
            title="System Performance Metrics",
            xaxis_title="Time (s)",
            yaxis_title="Usage (%)",
            height=400
        )
        
        return fig
    
    def start_simulation(self):
        """Start the brain simulation"""
        self.simulation_running = True
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def stop_simulation(self):
        """Stop the brain simulation"""
        self.simulation_running = False
    
    def reset_simulation(self):
        """Reset simulation data"""
        self.simulation_time = 0
        self.neural_data = {key: [] for key in self.neural_data}
        self.sleep_data = {key: [] for key in self.sleep_data}
        self.performance_data = {key: [] for key in self.performance_data}
        self.memory_data = {key: [] for key in self.memory_data}
    
    def run_simulation(self):
        """Run the brain simulation in background thread"""
        while self.simulation_running:
            # Update simulation time
            self.simulation_time += self.update_interval
            
            # Generate neural activity
            self.generate_neural_activity()
            
            # Generate sleep cycles
            self.generate_sleep_cycles()
            
            # Generate memory consolidation
            self.generate_memory_consolidation()
            
            # Update performance metrics
            self.update_performance_metrics()
            
            # Wait for next update
            time.sleep(self.update_interval)
    
    def generate_neural_activity(self):
        """Generate realistic neural activity data"""
        current_time = self.simulation_time
        
        # Generate spikes for some neurons
        num_spikes = np.random.poisson(5)  # Average 5 spikes per update
        
        for _ in range(num_spikes):
            neuron_id = np.random.randint(0, self.num_neurons)
            spike_time = current_time * 1000 + np.random.uniform(0, self.update_interval * 1000)
            
            self.neural_data['spike_times'].append(spike_time)
            self.neural_data['spike_neurons'].append(neuron_id)
            self.neural_data['timestamps'].append(current_time)
        
        # Keep only recent data (last 1000 points)
        max_points = 1000
        if len(self.neural_data['spike_times']) > max_points:
            self.neural_data['spike_times'] = self.neural_data['spike_times'][-max_points:]
            self.neural_data['spike_neurons'] = self.neural_data['spike_neurons'][-max_points:]
            self.neural_data['timestamps'] = self.neural_data['timestamps'][-max_points:]
    
    def generate_sleep_cycles(self):
        """Generate sleep cycle data"""
        current_time = self.simulation_time
        
        # 90-minute sleep cycle
        cycle_position = (current_time % 5400) / 5400
        
        if cycle_position < 0.2:  # Wake
            phase = 0
            brain_wave = np.random.normal(20, 5)  # Beta waves
        elif cycle_position < 0.4:  # NREM-1
            phase = 1
            brain_wave = np.random.normal(10, 3)  # Alpha waves
        elif cycle_position < 0.7:  # NREM-2
            phase = 2
            brain_wave = np.random.normal(5, 2)   # Theta waves
        else:  # REM
            phase = 3
            brain_wave = np.random.normal(15, 4)  # Mixed waves
        
        self.sleep_data['phases'].append(phase)
        self.sleep_data['brain_waves'].append(brain_wave)
        self.sleep_data['timestamps'].append(current_time)
        
        # Keep only recent data
        max_points = 1000
        if len(self.sleep_data['phases']) > max_points:
            self.sleep_data['phases'] = self.sleep_data['phases'][-max_points:]
            self.sleep_data['brain_waves'] = self.sleep_data['brain_waves'][-max_points:]
            self.sleep_data['timestamps'] = self.sleep_data['timestamps'][-max_points:]
    
    def generate_memory_consolidation(self):
        """Generate memory consolidation data"""
        current_time = self.simulation_time
        
        # Memory consolidation increases during sleep
        cycle_position = (current_time % 5400) / 5400
        
        if cycle_position > 0.2:  # During sleep
            episodic = 0.6 + 0.4 * np.sin(2 * np.pi * current_time / 1000)
            semantic = 0.5 + 0.3 * np.sin(2 * np.pi * current_time / 800)
            procedural = 0.7 + 0.3 * np.sin(2 * np.pi * current_time / 1200)
            working = 0.3 + 0.2 * np.sin(2 * np.pi * current_time / 600)
        else:  # During wake
            episodic = 0.1 + 0.1 * np.sin(2 * np.pi * current_time / 200)
            semantic = 0.2 + 0.1 * np.sin(2 * np.pi * current_time / 300)
            procedural = 0.15 + 0.1 * np.sin(2 * np.pi * current_time / 250)
            working = 0.8 + 0.2 * np.sin(2 * np.pi * current_time / 400)
        
        self.memory_data['episodic'].append(episodic)
        self.memory_data['semantic'].append(semantic)
        self.memory_data['procedural'].append(procedural)
        self.memory_data['working_memory'].append(working)
        self.memory_data['timestamps'].append(current_time)
        
        # Keep only recent data
        max_points = 1000
        if len(self.memory_data['episodic']) > max_points:
            for key in self.memory_data:
                if key != 'timestamps':
                    self.memory_data[key] = self.memory_data[key][-max_points:]
            self.memory_data['timestamps'] = self.memory_data['timestamps'][-max_points:]
    
    def update_performance_metrics(self):
        """Update system performance metrics"""
        current_time = self.simulation_time
        
        # Get real system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Simulate GPU usage (since we might not have GPU)
        gpu_usage = np.random.uniform(20, 80) if hasattr(psutil, 'gpu_percent') else np.random.uniform(30, 70)
        
        self.performance_data['cpu_usage'].append(cpu_percent)
        self.performance_data['memory_usage'].append(memory_percent)
        self.performance_data['gpu_usage'].append(gpu_usage)
        self.performance_data['timestamps'].append(current_time)
        
        # Keep only recent data
        max_points = 1000
        if len(self.performance_data['cpu_usage']) > max_points:
            for key in self.performance_data:
                if key != 'timestamps':
                    self.performance_data[key] = self.performance_data[key][-max_points:]
            self.performance_data['timestamps'] = self.performance_data['timestamps'][-max_points:]
    
    def export_dashboard_data(self):
        """Export dashboard data to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export JSON data
            export_data = {
                'neural_data': self.neural_data,
                'sleep_data': self.sleep_data,
                'memory_data': self.memory_data,
                'performance_data': self.performance_data,
                'simulation_time': self.simulation_time,
                'export_timestamp': timestamp
            }
            
            json_file = self.output_dir / f"live_dashboard_data_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            # Export summary report
            report_file = self.output_dir / f"live_dashboard_report_{timestamp}.md"
            with open(report_file, 'w') as f:
                f.write(self.generate_export_report(export_data))
            
            return f"✅ Data exported to {json_file.name} and {report_file.name}"
            
        except Exception as e:
            return f"❌ Export failed: {str(e)}"
    
    def generate_export_report(self, data):
        """Generate export report"""
        return f"""# Live Brain Simulation Dashboard Export Report

## Export Information
- **Timestamp**: {data['export_timestamp']}
- **Simulation Time**: {data['simulation_time']:.2f} seconds
- **Total Data Points**: {len(data['neural_data']['timestamps'])}

## Data Summary

### Neural Activity
- **Total Spikes**: {len(data['neural_data']['spike_times'])}
- **Active Neurons**: {len(set(data['neural_data']['spike_neurons'])) if data['neural_data']['spike_neurons'] else 0}
- **Average Firing Rate**: {len(data['neural_data']['spike_times']) / max(data['neural_data']['timestamps']) if data['neural_data']['timestamps'] else 0:.2f} Hz

### Sleep Cycles
- **Data Points**: {len(data['sleep_data']['phases'])}
- **Sleep Phases**: {set(data['sleep_data']['phases']) if data['sleep_data']['phases'] else set()}
- **Brain Wave Range**: {min(data['sleep_data']['brain_waves']):.1f} - {max(data['sleep_data']['brain_waves']):.1f} Hz

### Memory Consolidation
- **Episodic Memory**: {np.mean(data['memory_data']['episodic']):.3f} ± {np.std(data['memory_data']['episodic']):.3f}
- **Semantic Memory**: {np.mean(data['memory_data']['semantic']):.3f} ± {np.std(data['memory_data']['semantic']):.3f}
- **Procedural Memory**: {np.mean(data['memory_data']['procedural']):.3f} ± {np.std(data['memory_data']['procedural']):.3f}
- **Working Memory**: {np.mean(data['memory_data']['working_memory']):.3f} ± {np.std(data['memory_data']['working_memory']):.3f}

### Performance Metrics
- **CPU Usage**: {np.mean(data['performance_data']['cpu_usage']):.1f}% ± {np.std(data['performance_data']['cpu_usage']):.1f}%
- **Memory Usage**: {np.mean(data['performance_data']['memory_usage']):.1f}% ± {np.std(data['performance_data']['memory_usage']):.1f}%
- **GPU Usage**: {np.mean(data['performance_data']['gpu_usage']):.1f}% ± {np.std(data['performance_data']['gpu_usage']):.1f}%

## Export Files
- **JSON Data**: live_dashboard_data_{timestamp}.json
- **Report**: live_dashboard_report_{timestamp}.md

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    def run_dashboard(self, host='127.0.0.1', port=8050, debug=True):
        """Run the live dashboard"""
        print(f"🚀 Starting Live Parameter Dashboard...")
        print(f"🌐 Dashboard URL: http://{host}:{port}")
        print(f"📊 Monitoring: Neural activity, sleep cycles, memory, performance")
        print(f"⏱️ Update interval: {self.update_interval} seconds")
        print(f"🧠 Neurons: {self.num_neurons}")
        print(f"📁 Output directory: {self.output_dir}")
        
        self.app.run_server(host=host, port=port, debug=debug)

def main():
    """Main function to run the live dashboard"""
    monitor = LiveParameterMonitor()
    monitor.run_dashboard()

if __name__ == "__main__":
    main()
