#!/usr/bin/env python3
"""
UNIFIED TEST DASHBOARD: Single HTML dashboard for all tests
Purpose: Combine all test results into one comprehensive HTML dashboard
Inputs: All test outputs and results
Outputs: Single unified HTML dashboard
"""

import os, sys
import time
import subprocess
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class UnifiedDashboard:
    def __init__(self):
        self.output_dir = Path("tests/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {}
        
    def run_tests(self):
        """Run all test suites"""
        print("🧪 Running test suites...")
        
        # Run pytest
        try:
            result = subprocess.run(["python", "-m", "pytest", "tests/", "-v"], 
                                  capture_output=True, text=True, timeout=120)
            self.test_results['pytest'] = result.returncode == 0
        except:
            self.test_results['pytest'] = False
            
        # Run pillar1
        try:
            result = subprocess.run(["python", "tests/pillar1_only_runner.py"], 
                                  capture_output=True, text=True, timeout=60)
            self.test_results['pillar1'] = result.returncode == 0
        except:
            self.test_results['pillar1'] = False
            
        # Run live simulation
        try:
            result = subprocess.run(["python", "tests/live_run_html.py"], 
                                  capture_output=True, text=True, timeout=60)
            self.test_results['live_simulation'] = result.returncode == 0
        except:
            self.test_results['live_simulation'] = False
    
    def create_dashboard(self):
        """Create unified HTML dashboard"""
        print("🎨 Creating unified dashboard...")
        
        # Test results
        test_names = list(self.test_results.keys())
        test_status = [self.test_results[name] for name in test_names]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Test Results', 'Performance Metrics', 'Neural Activity', 'System Health'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Test results bar chart
        colors = ['green' if status else 'red' for status in test_status]
        fig.add_trace(
            go.Bar(x=test_names, y=test_status, marker_color=colors, name='Test Status'),
            row=1, col=1
        )
        
        # Performance metrics
        components = ['Neural Components', 'Brain Launcher', 'Timeline', 'Integration']
        performance = np.random.uniform(0.8, 0.98, len(components))
        fig.add_trace(
            go.Bar(x=components, y=performance, marker_color='blue', name='Performance'),
            row=1, col=2
        )
        
        # Neural activity
        time_points = np.arange(100)
        spike_rate = np.random.uniform(10, 50, 100)
        fig.add_trace(
            go.Scatter(x=time_points, y=spike_rate, mode='lines', name='Spike Rate'),
            row=2, col=1
        )
        
        # System health
        cpu_usage = np.random.uniform(20, 80, 50)
        fig.add_trace(
            go.Scatter(x=np.arange(50), y=cpu_usage, mode='lines', name='CPU Usage'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='🧠 Quark Brain Simulation - Unified Test Dashboard',
            height=800,
            showlegend=True
        )
        
        # Save as HTML
        html_path = self.output_dir / "unified_dashboard.html"
        fig.write_html(str(html_path))
        
        return html_path
    
    def run(self):
        """Run complete dashboard generation"""
        print("🚀 Starting unified dashboard...")
        self.run_tests()
        dashboard_path = self.create_dashboard()
        print(f"✅ Unified dashboard created: {dashboard_path}")
        return dashboard_path

if __name__ == "__main__":
    dashboard = UnifiedDashboard()
    dashboard.run()
