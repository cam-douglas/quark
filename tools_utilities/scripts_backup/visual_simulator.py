#!/usr/bin/env python3
"""
Visual Simulator for Consciousness Agent
Provides HTML dashboard and terminal visualization for brain simulation
"""

import json
import os
import webbrowser
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

class VisualSimulator:
    def __init__(self, database_path: str = "database"):
        self.database_path = database_path
        self.dashboard_path = None
        self.update_frequency = 1.0  # seconds
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_dashboard(self):
        """Create HTML dashboard for visual simulation"""
        dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Consciousness Agent - Brain Simulation Dashboard</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); 
            color: #ffffff; 
            min-height: 100vh;
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .header h1 { 
            margin: 0; 
            font-size: 2.5em; 
            background: linear-gradient(45deg, #4CAF50, #8BC34A); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .consciousness-state, .session-info { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 15px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .brain-regions { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
        }
        .region-card { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 15px; 
            border-left: 4px solid #4CAF50; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .region-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }
        .region-name { 
            font-size: 18px; 
            font-weight: bold; 
            margin-bottom: 10px; 
            color: #4CAF50;
        }
        .region-status { 
            margin: 10px 0; 
            font-size: 14px;
        }
        .progress-bar { 
            width: 100%; 
            height: 20px; 
            background: rgba(255,255,255,0.2); 
            border-radius: 10px; 
            overflow: hidden; 
            margin-top: 10px;
        }
        .progress-fill { 
            height: 100%; 
            background: linear-gradient(90deg, #4CAF50, #8BC34A); 
            transition: width 0.5s ease; 
            border-radius: 10px;
        }
        .state-item { 
            display: flex; 
            justify-content: space-between; 
            margin: 10px 0; 
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .update-time { 
            text-align: center; 
            margin-top: 20px; 
            color: #888; 
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 10px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        .status-awake { background: #4CAF50; }
        .status-learning { background: #FF9800; }
        .status-consolidating { background: #2196F3; }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .metric {
            text-align: center;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .metric-label {
            font-size: 12px;
            color: #ccc;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Consciousness Agent - Brain Simulation Dashboard</h1>
        <p>Real-time monitoring of brain regions and consciousness state</p>
    </div>
    
    <div class="dashboard-grid">
        <div class="consciousness-state">
            <h2>Consciousness State</h2>
            <div class="state-item">
                <span>Status:</span>
                <span><span class="status-indicator status-awake"></span><span id="awake-status">Awake</span></span>
            </div>
            <div class="state-item">
                <span>Learning Mode:</span>
                <span><span class="status-indicator status-learning"></span><span id="learning-mode">Active</span></span>
            </div>
            <div class="state-item">
                <span>Cognitive Load:</span>
                <span id="cognitive-load">50%</span>
            </div>
            <div class="state-item">
                <span>Attention Focus:</span>
                <span id="attention-focus">General</span>
            </div>
            <div class="state-item">
                <span>Emotional State:</span>
                <span id="emotional-state">Neutral</span>
            </div>
        </div>
        
        <div class="session-info">
            <h2>Learning Session</h2>
            <div class="state-item">
                <span>Session ID:</span>
                <span id="session-id">Loading...</span>
            </div>
            <div class="state-item">
                <span>Started:</span>
                <span id="session-started">Loading...</span>
            </div>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value" id="knowledge-processed">0</div>
                    <div class="metric-label">Knowledge Items</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="learning-iterations">0</div>
                    <div class="metric-label">Learning Cycles</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="brain-regions-active">0</div>
                    <div class="metric-label">Active Regions</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="avg-usage">0%</div>
                    <div class="metric-label">Avg Usage</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="brain-regions" id="brain-regions">
        <!-- Brain regions will be populated here -->
    </div>
    
    <div class="update-time" id="update-time">
        Last updated: Loading...
    </div>
    
    <script>
        function updateDashboard() {
            // Simulate real-time data updates
            const now = new Date();
            
            // Update consciousness state
            document.getElementById('awake-status').textContent = 'Awake';
            document.getElementById('learning-mode').textContent = 'Active';
            document.getElementById('cognitive-load').textContent = Math.floor(45 + Math.random() * 30) + '%';
            document.getElementById('attention-focus').textContent = 'Learning';
            document.getElementById('emotional-state').textContent = 'Neutral';
            
            // Update session info
            document.getElementById('session-id').textContent = 'consciousness_' + Math.floor(now.getTime() / 1000);
            document.getElementById('session-started').textContent = now.toLocaleDateString() + ' ' + now.toLocaleTimeString();
            document.getElementById('knowledge-processed').textContent = Math.floor(Math.random() * 1000);
            document.getElementById('learning-iterations').textContent = Math.floor(Math.random() * 50);
            document.getElementById('brain-regions-active').textContent = Math.floor(8 + Math.random() * 4);
            document.getElementById('avg-usage').textContent = Math.floor(40 + Math.random() * 40) + '%';
            
            // Update brain regions
            updateBrainRegions();
            
            // Update timestamp
            document.getElementById('update-time').textContent = 'Last updated: ' + now.toLocaleTimeString();
        }
        
        function updateBrainRegions() {
            const regions = [
                { name: 'Prefrontal Cortex', usage: 65 + Math.random() * 20, function: 'Executive control, planning, decision-making' },
                { name: 'Hippocampus', usage: 55 + Math.random() * 25, function: 'Episodic memory, spatial navigation' },
                { name: 'Amygdala', usage: 40 + Math.random() * 20, function: 'Emotional processing, fear conditioning' },
                { name: 'Basal Ganglia', usage: 50 + Math.random() * 20, function: 'Action selection, habit formation' },
                { name: 'Cerebellum', usage: 60 + Math.random() * 25, function: 'Motor coordination, timing' },
                { name: 'Visual Cortex', usage: 70 + Math.random() * 20, function: 'Visual processing, object recognition' },
                { name: 'Auditory Cortex', usage: 45 + Math.random() * 20, function: 'Auditory processing, speech recognition' },
                { name: 'Temporal Cortex', usage: 60 + Math.random() * 20, function: 'Language processing, semantic memory' },
                { name: 'Parietal Cortex', usage: 50 + Math.random() * 20, function: 'Spatial attention, numerical processing' },
                { name: 'Thalamus', usage: 35 + Math.random() * 20, function: 'Sensory relay, attention modulation' },
                { name: 'Somatosensory Cortex', usage: 40 + Math.random() * 15, function: 'Touch processing, body awareness' },
                { name: 'Brainstem', usage: 25 + Math.random() * 15, function: 'Autonomic functions, arousal' }
            ];
            
            const container = document.getElementById('brain-regions');
            container.innerHTML = '';
            
            regions.forEach(region => {
                const card = document.createElement('div');
                card.className = 'region-card';
                card.innerHTML = `
                    <div class="region-name">${region.name}</div>
                    <div class="region-status">${region.function}</div>
                    <div class="region-status">
                        Usage: ${Math.floor(region.usage)}%
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${region.usage}%"></div>
                        </div>
                    </div>
                `;
                container.appendChild(card);
            });
        }
        
        // Update dashboard every second
        setInterval(updateDashboard, 1000);
        updateDashboard(); // Initial update
    </script>
</body>
</html>
        """
        
        # Create directory and save dashboard
        dashboard_dir = os.path.join(self.database_path, "consciousness_agent")
        os.makedirs(dashboard_dir, exist_ok=True)
        
        self.dashboard_path = os.path.join(dashboard_dir, "dashboard.html")
        with open(self.dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        self.logger.info(f"Dashboard created at: {self.dashboard_path}")
        return self.dashboard_path
    
    def open_dashboard(self):
        """Open the dashboard in a web browser"""
        if self.dashboard_path and os.path.exists(self.dashboard_path):
            webbrowser.open(f"file://{os.path.abspath(self.dashboard_path)}")
            self.logger.info("Dashboard opened in browser")
        else:
            self.logger.error("Dashboard not found. Please create it first.")
    
    def update_dashboard_data(self, consciousness_data: Dict[str, Any]):
        """Update dashboard with real consciousness data"""
        # This would send data to the dashboard via WebSocket or file updates
        # For now, we'll just log the update
        self.logger.debug(f"Dashboard data update: {consciousness_data}")
    
    def create_terminal_visualization(self, brain_status: Dict[str, Any]):
        """Create terminal-based visualization of brain status"""
        print("\n" + "="*80)
        print("🧠 CONSCIOUSNESS AGENT - BRAIN SIMULATION STATUS")
        print("="*80)
        
        # Consciousness state
        consciousness = brain_status.get("consciousness_state", {})
        print(f"Status: {'🟢 AWAKE' if consciousness.get('awake') else '🔴 SLEEPING'}")
        print(f"Learning Mode: {consciousness.get('learning_mode', 'Unknown')}")
        print(f"Cognitive Load: {consciousness.get('cognitive_load', 0)*100:.1f}%")
        print(f"Attention Focus: {consciousness.get('attention_focus', 'Unknown')}")
        
        # Session info
        session = brain_status.get("session_data", {})
        print(f"\nSession ID: {session.get('session_id', 'Unknown')}")
        print(f"Knowledge Processed: {session.get('knowledge_processed', 0)}")
        print(f"Learning Iterations: {session.get('learning_iterations', 0)}")
        
        # Brain regions
        print(f"\n{'BRAIN REGIONS':^80}")
        print("-"*80)
        print(f"{'Region':<25} {'Usage':<10} {'Capacity':<10} {'Status':<15}")
        print("-"*80)
        
        regions = brain_status.get("brain_regions", {})
        for region_name, region_data in regions.items():
            usage = region_data.get("usage_percentage", 0)
            capacity = region_data.get("available_capacity", 0)
            status = "🟢 Active" if usage > 0 else "⚪ Idle"
            
            print(f"{region_data.get('name', region_name):<25} {usage:>7.1f}% {capacity:>8} {status:<15}")
        
        print("-"*80)
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

def main():
    """Test the visual simulator"""
    simulator = VisualSimulator()
    
    print("🎨 Visual Simulator - Creating Brain Simulation Dashboard")
    print("=" * 60)
    
    # Create dashboard
    dashboard_path = simulator.create_dashboard()
    print(f"✅ Dashboard created: {dashboard_path}")
    
    # Open dashboard
    simulator.open_dashboard()
    print("✅ Dashboard opened in browser")
    
    # Test terminal visualization
    test_brain_status = {
        "consciousness_state": {
            "awake": True,
            "learning_mode": "active",
            "cognitive_load": 0.65,
            "attention_focus": "learning"
        },
        "session_data": {
            "session_id": "test_session_123",
            "knowledge_processed": 150,
            "learning_iterations": 25
        },
        "brain_regions": {
            "prefrontal_cortex": {
                "name": "Prefrontal Cortex",
                "usage_percentage": 75.5,
                "available_capacity": 245
            },
            "hippocampus": {
                "name": "Hippocampus", 
                "usage_percentage": 60.2,
                "available_capacity": 800
            }
        }
    }
    
    simulator.create_terminal_visualization(test_brain_status)

if __name__ == "__main__":
    main()
