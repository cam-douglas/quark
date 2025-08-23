#!/usr/bin/env python3
"""
SIMPLE LIVE PARAMETER DASHBOARD: Real-time monitoring of brain simulation
Purpose: Provide live visualization of neural activity and system parameters
Inputs: Real-time simulation data
Outputs: Interactive HTML dashboard with live updates
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import psutil
import time
from pathlib import Path
import threading

# Create output directory
output_dir = Path("tests/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize Dash app
app = dash.Dash(__name__)

# Global data storage
neural_data = {'times': [], 'spikes': [], 'rates': []}
sleep_data = {'times': [], 'phases': [], 'waves': []}
performance_data = {'times': [], 'cpu': [], 'memory': []}

# Simulation state
simulation_running = False
simulation_time = 0

# Dashboard layout
app.layout = html.Div([
    html.H1("🧠 Live Brain Simulation Dashboard", style={'textAlign': 'center'}),
    
    # Control buttons
    html.Div([
        html.Button("Start", id="start-btn", n_clicks=0),
        html.Button("Stop", id="stop-btn", n_clicks=0),
        html.Button("Reset", id="reset-btn", n_clicks=0),
        html.Div(id="status", style={'margin': '10px'})
    ], style={'textAlign': 'center'}),
    
    # Real-time metrics
    html.Div([
        html.Div([
            html.H4("CPU Usage"),
            html.Div(id="cpu-display", style={'fontSize': '24px'})
        ]),
        html.Div([
            html.H4("Memory Usage"),
            html.Div(id="memory-display", style={'fontSize': '24px'})
        ]),
        html.Div([
            html.H4("Simulation Time"),
            html.Div(id="time-display", style={'fontSize': '24px'})
        ])
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
    
    # Charts
    dcc.Graph(id="neural-chart"),
    dcc.Graph(id="sleep-chart"),
    dcc.Graph(id="performance-chart"),
    
    # Update interval
    dcc.Interval(id="interval", interval=1000, n_intervals=0)
])

# Callbacks
@app.callback(
    Output("status", "children"),
    [Input("start-btn", "n_clicks"),
     Input("stop-btn", "n_clicks"),
     Input("reset-btn", "n_clicks")]
)
def update_status(start, stop, reset):
    global simulation_running
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return "Ready"
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "start-btn":
        simulation_running = True
        start_simulation()
        return "🟢 Running"
    elif button_id == "stop-btn":
        simulation_running = False
        return "🔴 Stopped"
    elif button_id == "reset-btn":
        reset_simulation()
        return "🔄 Reset"
    
    return "Ready"

@app.callback(
    [Output("neural-chart", "figure"),
     Output("sleep-chart", "figure"),
     Output("performance-chart", "figure")],
    [Input("interval", "n_intervals")]
)
def update_charts(n):
    if not simulation_running:
        return create_empty_charts()
    
    return [
        create_neural_chart(),
        create_sleep_chart(),
        create_performance_chart()
    ]

@app.callback(
    [Output("cpu-display", "children"),
     Output("memory-display", "children"),
     Output("time-display", "children")],
    [Input("interval", "n_intervals")]
)
def update_metrics(n):
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent
    
    return f"{cpu:.1f}%", f"{memory:.1f}%", f"{simulation_time:.1f}s"

def create_empty_charts():
    """Create empty charts"""
    empty = go.Figure()
    empty.add_annotation(text="Start simulation to see data", x=0.5, y=0.5, showarrow=False)
    empty.update_layout(title="No Data")
    
    return [empty] * 3

def create_neural_chart():
    """Create neural activity chart"""
    fig = go.Figure()
    
    if neural_data['times']:
        fig.add_trace(go.Scatter(
            x=neural_data['times'],
            y=neural_data['spikes'],
            mode='markers',
            name='Spikes'
        ))
    
    fig.update_layout(title="Neural Activity", xaxis_title="Time", yaxis_title="Neuron ID")
    return fig

def create_sleep_chart():
    """Create sleep cycles chart"""
    fig = go.Figure()
    
    if sleep_data['times']:
        fig.add_trace(go.Scatter(
            x=sleep_data['times'],
            y=sleep_data['phases'],
            mode='lines',
            name='Sleep Phase'
        ))
    
    fig.update_layout(title="Sleep Cycles", xaxis_title="Time", yaxis_title="Phase")
    return fig

def create_performance_chart():
    """Create performance chart"""
    fig = go.Figure()
    
    if performance_data['times']:
        fig.add_trace(go.Scatter(
            x=performance_data['times'],
            y=performance_data['cpu'],
            mode='lines',
            name='CPU'
        ))
        fig.add_trace(go.Scatter(
            x=performance_data['times'],
            y=performance_data['memory'],
            mode='lines',
            name='Memory'
        ))
    
    fig.update_layout(title="Performance", xaxis_title="Time", yaxis_title="Usage %")
    return fig

def start_simulation():
    """Start simulation thread"""
    thread = threading.Thread(target=run_simulation)
    thread.daemon = True
    thread.start()

def run_simulation():
    """Run simulation in background"""
    global simulation_time
    
    while simulation_running:
        # Update time
        simulation_time += 1
        
        # Generate neural data
        if np.random.random() < 0.3:  # 30% chance of spike
            neural_data['times'].append(simulation_time)
            neural_data['spikes'].append(np.random.randint(0, 100))
        
        # Generate sleep data
        sleep_data['times'].append(simulation_time)
        sleep_data['phases'].append((simulation_time // 10) % 4)
        sleep_data['waves'].append(np.random.normal(10, 5))
        
        # Update performance data
        performance_data['times'].append(simulation_time)
        performance_data['cpu'].append(psutil.cpu_percent())
        performance_data['memory'].append(psutil.virtual_memory().percent)
        
        # Keep only recent data
        max_points = 100
        for data in [neural_data, sleep_data, performance_data]:
            for key in data:
                if len(data[key]) > max_points:
                    data[key] = data[key][-max_points:]
        
        time.sleep(1)

def reset_simulation():
    """Reset simulation data"""
    global simulation_time
    simulation_time = 0
    
    for data in [neural_data, sleep_data, performance_data]:
        for key in data:
            data[key] = []

if __name__ == "__main__":
    print("🚀 Starting Live Brain Simulation Dashboard...")
    print("🌐 Dashboard URL: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)
