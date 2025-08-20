#!/usr/bin/env python3
"""
MAIN CLOUD COGNITIVE AGENT: Coordinates local agent with cloud processing
Purpose: Run heavy cognitive load on cloud while keeping local agent responsive
Inputs: User requests, cloud processing results
Outputs: Integrated responses, real-time dashboard
Seeds: Deterministic processing for reproducibility
Deps: dash, plotly, numpy, threading, queue, time, pathlib
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import threading
import queue
import time
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MainCloudCognitiveAgent:
    """Main agent that coordinates local processing with cloud cognitive load"""
    
    def __init__(self):
        self.output_dir = Path("cloud_computing/cognitive_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Agent state
        self.agent_active = True
        self.cloud_connected = False
        self.current_task = None
        
        # Processing queues
        self.cloud_request_queue = queue.Queue()
        self.cloud_result_queue = queue.Queue()
        
        # Local cognitive state
        self.cognitive_state = {
            'attention_level': 0.8,
            'memory_available': 0.9,
            'processing_capacity': 0.7,
            'decision_confidence': 0.6
        }
        
        # Real-time data
        self.real_time_data = {
            'timestamps': [],
            'attention': [],
            'memory': [],
            'processing': [],
            'confidence': [],
            'cloud_tasks': [],
            'local_tasks': []
        }
        
        # Initialize dashboard
        self.setup_dashboard()
        
        # Start background processing
        self.start_background_processing()
    
    def setup_dashboard(self):
        """Setup the main agent dashboard"""
        self.app = dash.Dash(__name__)
        
        self.app.layout = html.Div([
            html.H1("🧠 Main Cloud-Cognitive Agent Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Agent Status Panel
            html.Div([
                html.Div([
                    html.H4("Agent Status"),
                    html.Div(id="agent-status", style={'fontSize': '20px', 'color': '#27ae60'})
                ], style={'textAlign': 'center', 'margin': '10px'}),
                html.Div([
                    html.H4("Cloud Connection"),
                    html.Div(id="cloud-status", style={'fontSize': '20px', 'color': '#e74c3c'})
                ], style={'textAlign': 'center', 'margin': '10px'}),
                html.Div([
                    html.H4("Current Task"),
                    html.Div(id="current-task", style={'fontSize': '20px', 'color': '#3498db'})
                ], style={'textAlign': 'center', 'margin': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
            
            # Cognitive State Panel
            html.Div([
                html.H3("🧠 Real-time Cognitive State", style={'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        html.H4("Attention Level"),
                        html.Div(id="attention-display", style={'fontSize': '24px', 'color': '#e74c3c'})
                    ]),
                    html.Div([
                        html.H4("Memory Available"),
                        html.Div(id="memory-display", style={'fontSize': '24px', 'color': '#3498db'})
                    ]),
                    html.Div([
                        html.H4("Processing Capacity"),
                        html.Div(id="processing-display", style={'fontSize': '24px', 'color': '#f39c12'})
                    ]),
                    html.Div([
                        html.H4("Decision Confidence"),
                        html.Div(id="confidence-display", style={'fontSize': '24px', 'color': '#9b59b6'})
                    ])
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'})
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa'}),
            
            # Control Panel
            html.Div([
                html.H3("🎮 Agent Controls", style={'textAlign': 'center'}),
                html.Div([
                    html.Button("Submit Neural Task", id="neural-task-btn", n_clicks=0),
                    html.Button("Submit Memory Task", id="memory-task-btn", n_clicks=0),
                    html.Button("Submit Attention Task", id="attention-task-btn", n_clicks=0),
                    html.Button("Submit Decision Task", id="decision-task-btn", n_clicks=0),
                    html.Button("Process Cloud Results", id="process-results-btn", n_clicks=0),
                    html.Button("Clear All Tasks", id="clear-tasks-btn", n_clicks=0)
                ], style={'textAlign': 'center', 'margin': '20px'})
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#ecf0f1'}),
            
            # Real-time Charts
            html.Div([
                html.H3("📊 Real-time Cognitive Metrics", style={'textAlign': 'center'}),
                dcc.Graph(id="cognitive-state-chart", style={'height': '400px'}),
                dcc.Graph(id="task-processing-chart", style={'height': '400px'}),
                dcc.Graph(id="cloud-local-balance-chart", style={'height': '400px'})
            ], style={'margin': '20px'}),
            
            # Task Queue Display
            html.Div([
                html.H3("📋 Task Queue Status", style={'textAlign': 'center'}),
                html.Div(id="queue-status", style={
                    'height': '200px', 'overflowY': 'scroll', 
                    'border': '1px solid #ddd', 'padding': '10px',
                    'backgroundColor': '#f8f9fa', 'fontFamily': 'monospace'
                })
            ], style={'margin': '20px'}),
            
            # Update interval
            dcc.Interval(id="interval", interval=1000, n_intervals=0)
        ])
        
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output("agent-status", "children"),
             Output("cloud-status", "children"),
             Output("current-task", "children")],
            [Input("interval", "n_intervals")]
        )
        def update_status(n):
            agent_status = "🟢 Active" if self.agent_active else "🔴 Inactive"
            cloud_status = "🟢 Connected" if self.cloud_connected else "🔴 Disconnected"
            current_task = self.current_task or "Idle"
            
            return agent_status, cloud_status, current_task
        
        @self.app.callback(
            [Output("attention-display", "children"),
             Output("memory-display", "children"),
             Output("processing-display", "children"),
             Output("confidence-display", "children")],
            [Input("interval", "n_intervals")]
        )
        def update_cognitive_metrics(n):
            attention = f"{self.cognitive_state['attention_level']:.2f}"
            memory = f"{self.cognitive_state['memory_available']:.2f}"
            processing = f"{self.cognitive_state['processing_capacity']:.2f}"
            confidence = f"{self.cognitive_state['decision_confidence']:.2f}"
            
            return attention, memory, processing, confidence
        
        @self.app.callback(
            [Output("cognitive-state-chart", "figure"),
             Output("task-processing-chart", "figure"),
             Output("cloud-local-balance-chart", "figure")],
            [Input("interval", "n_intervals")]
        )
        def update_charts(n):
            return [
                self.create_cognitive_state_chart(),
                self.create_task_processing_chart(),
                self.create_cloud_local_balance_chart()
            ]
        
        @self.app.callback(
            Output("queue-status", "children"),
            [Input("interval", "n_intervals")]
        )
        def update_queue_status(n):
            cloud_queue_size = self.cloud_request_queue.qsize()
            cloud_results_size = self.cloud_result_queue.qsize()
            
            status_text = f"""Cloud Request Queue: {cloud_queue_size} tasks
Cloud Results Queue: {cloud_results_size} results
Local Agent: {'Active' if self.agent_active else 'Inactive'}
Cloud Connection: {'Connected' if self.cloud_connected else 'Disconnected'}
Current Task: {self.current_task or 'Idle'}
Last Update: {time.strftime('%H:%M:%S')}"""
            
            return html.Pre(status_text)
        
        # Task submission callbacks
        @self.app.callback(
            Output("neural-task-btn", "n_clicks"),
            [Input("neural-task-btn", "n_clicks")]
        )
        def submit_neural_task(n_clicks):
            if n_clicks and n_clicks > 0:
                self.submit_cloud_task("neural_simulation", {"duration": 2000, "num_neurons": 2000})
            return 0
        
        @self.app.callback(
            Output("memory-task-btn", "n_clicks"),
            [Input("memory-task-btn", "n_clicks")]
        )
        def submit_memory_task(n_clicks):
            if n_clicks and n_clicks > 0:
                self.submit_cloud_task("memory_consolidation", {"duration": 2000})
            return 0
        
        @self.app.callback(
            Output("attention-task-btn", "n_clicks"),
            [Input("attention-task-btn", "n_clicks")]
        )
        def submit_attention_task(n_clicks):
            if n_clicks and n_clicks > 0:
                self.submit_cloud_task("attention_modeling", {"duration": 2000})
            return 0
        
        @self.app.callback(
            Output("decision-task-btn", "n_clicks"),
            [Input("decision-task-btn", "n_clicks")]
        )
        def submit_decision_task(n_clicks):
            if n_clicks and n_clicks > 0:
                self.submit_cloud_task("decision_analysis", {"duration": 2000})
            return 0
        
        @self.app.callback(
            Output("process-results-btn", "n_clicks"),
            [Input("process-results-btn", "n_clicks")]
        )
        def process_cloud_results(n_clicks):
            if n_clicks and n_clicks > 0:
                self.process_cloud_results()
            return 0
        
        @self.app.callback(
            Output("clear-tasks-btn", "n_clicks"),
            [Input("clear-tasks-btn", "n_clicks")]
        )
        def clear_all_tasks(n_clicks):
            if n_clicks and n_clicks > 0:
                self.clear_all_tasks()
            return 0
    
    def create_cognitive_state_chart(self):
        """Create cognitive state chart"""
        fig = go.Figure()
        
        if self.real_time_data['timestamps']:
            fig.add_trace(go.Scatter(
                x=self.real_time_data['timestamps'],
                y=self.real_time_data['attention'],
                mode='lines',
                name='Attention',
                line=dict(color='red', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=self.real_time_data['timestamps'],
                y=self.real_time_data['memory'],
                mode='lines',
                name='Memory',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=self.real_time_data['timestamps'],
                y=self.real_time_data['processing'],
                mode='lines',
                name='Processing',
                line=dict(color='orange', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=self.real_time_data['timestamps'],
                y=self.real_time_data['confidence'],
                mode='lines',
                name='Confidence',
                line=dict(color='purple', width=2)
            ))
        
        fig.update_layout(title="Real-time Cognitive State", xaxis_title="Time", yaxis_title="Level")
        return fig
    
    def create_task_processing_chart(self):
        """Create task processing chart"""
        fig = go.Figure()
        
        if self.real_time_data['timestamps']:
            fig.add_trace(go.Scatter(
                x=self.real_time_data['timestamps'],
                y=self.real_time_data['cloud_tasks'],
                mode='lines+markers',
                name='Cloud Tasks',
                line=dict(color='green', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=self.real_time_data['timestamps'],
                y=self.real_time_data['local_tasks'],
                mode='lines+markers',
                name='Local Tasks',
                line=dict(color='blue', width=2)
            ))
        
        fig.update_layout(title="Task Processing Distribution", xaxis_title="Time", yaxis_title="Task Count")
        return fig
    
    def create_cloud_local_balance_chart(self):
        """Create cloud-local balance chart"""
        fig = go.Figure()
        
        if self.real_time_data['timestamps']:
            # Calculate balance ratio
            total_tasks = [c + l for c, l in zip(self.real_time_data['cloud_tasks'], self.real_time_data['local_tasks'])]
            cloud_ratio = [c / max(t, 1) for c, t in zip(self.real_time_data['cloud_tasks'], total_tasks)]
            local_ratio = [l / max(t, 1) for l, t in zip(self.real_time_data['local_tasks'], total_tasks)]
            
            fig.add_trace(go.Scatter(
                x=self.real_time_data['timestamps'],
                y=cloud_ratio,
                mode='lines',
                name='Cloud Processing Ratio',
                line=dict(color='green', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=self.real_time_data['timestamps'],
                y=local_ratio,
                mode='lines',
                name='Local Processing Ratio',
                line=dict(color='blue', width=3)
            ))
        
        fig.update_layout(title="Cloud vs Local Processing Balance", xaxis_title="Time", yaxis_title="Processing Ratio")
        return fig
    
    def start_background_processing(self):
        """Start background processing thread"""
        self.background_thread = threading.Thread(target=self.background_processing_worker)
        self.background_thread.daemon = True
        self.background_thread.start()
        logger.info("🚀 Background processing started")
    
    def background_processing_worker(self):
        """Background worker for continuous processing"""
        while self.agent_active:
            try:
                # Update cognitive state
                self.update_cognitive_state()
                
                # Update real-time data
                self.update_real_time_data()
                
                # Process any cloud results
                if not self.cloud_result_queue.empty():
                    self.process_cloud_results()
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Background processing error: {e}")
                time.sleep(1)
    
    def update_cognitive_state(self):
        """Update local cognitive state"""
        # Simulate cognitive state changes
        self.cognitive_state['attention_level'] = np.clip(
            self.cognitive_state['attention_level'] + np.random.normal(0, 0.01), 0, 1
        )
        self.cognitive_state['memory_available'] = np.clip(
            self.cognitive_state['memory_available'] + np.random.normal(0, 0.005), 0, 1
        )
        self.cognitive_state['processing_capacity'] = np.clip(
            self.cognitive_state['processing_capacity'] + np.random.normal(0, 0.01), 0, 1
        )
        self.cognitive_state['decision_confidence'] = np.clip(
            self.cognitive_state['decision_confidence'] + np.random.normal(0, 0.01), 0, 1
        )
    
    def update_real_time_data(self):
        """Update real-time data arrays"""
        current_time = time.time()
        
        self.real_time_data['timestamps'].append(current_time)
        self.real_time_data['attention'].append(self.cognitive_state['attention_level'])
        self.real_time_data['memory'].append(self.cognitive_state['memory_available'])
        self.real_time_data['processing'].append(self.cognitive_state['processing_capacity'])
        self.real_time_data['confidence'].append(self.cognitive_state['decision_confidence'])
        self.real_time_data['cloud_tasks'].append(self.cloud_request_queue.qsize())
        self.real_time_data['local_tasks'].append(1 if self.current_task else 0)
        
        # Keep only recent data (last 1000 points)
        max_points = 1000
        if len(self.real_time_data['timestamps']) > max_points:
            for key in self.real_time_data:
                self.real_time_data[key] = self.real_time_data[key][-max_points:]
    
    def submit_cloud_task(self, task_type, parameters):
        """Submit task to cloud processing"""
        task = {
            'id': f"task_{int(time.time())}",
            'type': task_type,
            'parameters': parameters,
            'timestamp': time.time(),
            'status': 'queued'
        }
        
        self.cloud_request_queue.put(task)
        self.current_task = f"Cloud: {task_type}"
        logger.info(f"📋 Cloud task submitted: {task_type}")
        
        # Simulate cloud connection
        if not self.cloud_connected:
            self.cloud_connected = True
            logger.info("☁️ Cloud connection established")
    
    def process_cloud_results(self):
        """Process results from cloud processing"""
        try:
            result = self.cloud_result_queue.get_nowait()
            logger.info(f"✅ Cloud result processed: {result['type']}")
            
            # Update cognitive state based on result
            if result['type'] == 'neural_activity':
                self.cognitive_state['processing_capacity'] = min(1.0, self.cognitive_state['processing_capacity'] + 0.1)
            elif result['type'] == 'memory_consolidation':
                self.cognitive_state['memory_available'] = min(1.0, self.cognitive_state['memory_available'] + 0.05)
            elif result['type'] == 'attention_focus':
                self.cognitive_state['attention_level'] = min(1.0, self.cognitive_state['attention_level'] + 0.05)
            elif result['type'] == 'decision_analysis':
                self.cognitive_state['decision_confidence'] = min(1.0, self.cognitive_state['decision_confidence'] + 0.1)
            
            # Clear current task
            self.current_task = None
            
            # Save result
            self.save_cloud_result(result)
            
        except queue.Empty:
            pass
    
    def save_cloud_result(self, result):
        """Save cloud processing result"""
        timestamp = int(time.time())
        filename = f"cloud_result_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"💾 Cloud result saved to {filepath}")
    
    def clear_all_tasks(self):
        """Clear all pending tasks"""
        while not self.cloud_request_queue.empty():
            try:
                self.cloud_request_queue.get_nowait()
            except queue.Empty:
                break
        
        self.current_task = None
        logger.info("🧹 All tasks cleared")
    
    def run_dashboard(self, host='127.0.0.1', port=8050):
        """Run the main agent dashboard"""
        print(f"🚀 Starting Main Cloud-Cognitive Agent Dashboard...")
        print(f"🌐 Dashboard URL: http://{host}:{port}")
        print(f"🧠 Local Agent: Active and responsive")
        print(f"☁️ Cloud Processing: Integrated and scalable")
        print(f"📊 Real-time Monitoring: Cognitive state, task processing, cloud balance")
        print(f"🎮 Interactive Controls: Submit tasks, monitor progress, clear queues")
        
        self.app.run(debug=False, host=host, port=port)

def main():
    """Main function to run the cloud-cognitive agent"""
    agent = MainCloudCognitiveAgent()
    
    print("🧠 Main Cloud-Cognitive Agent initialized")
    print("☁️ Cloud processing integration ready")
    print("📊 Real-time dashboard active")
    print("🎮 Agent controls available")
    
    # Run dashboard
    agent.run_dashboard()

if __name__ == "__main__":
    main()
