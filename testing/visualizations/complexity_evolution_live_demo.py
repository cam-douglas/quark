#!/usr/bin/env python3
"""
Complexity Evolution Agent Live 3D Demo

This demo showcases the Complexity Evolution Agent working with live 3D streaming
to demonstrate how Quark evolves its complexity based on development stages.
"""

import sys
import os
import asyncio
import json
import time
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from brain_architecture.neural_core.complexity_evolution_agent.complexity_evolver import ComplexityEvolutionAgent
from brain_architecture.neural_core.complexity_evolution_agent.connectome_synchronizer import ConnectomeSynchronizer

class ComplexityEvolutionLiveDemo:
    """Live 3D demo of the Complexity Evolution Agent"""
    
    def __init__(self):
        self.agent = ComplexityEvolutionAgent()
        self.synchronizer = ConnectomeSynchronizer()
        self.demo_data = []
        self.evolution_history = []
        
    def create_3d_complexity_landscape(self):
        """Create a 3D landscape showing complexity evolution"""
        # Generate complexity landscape data
        stages = ['F', 'N0', 'N1', 'N2', 'N3']
        complexity_factors = [1.0, 2.5, 4.0, 6.0, 8.0]
        
        # Create 3D surface
        x = np.linspace(0, 4, 50)
        y = np.linspace(0, 4, 50)
        X, Y = np.meshgrid(x, y)
        
        # Create complexity surface with peaks at each stage
        Z = np.zeros_like(X)
        for i, (stage, factor) in enumerate(zip(stages, complexity_factors)):
            # Create peak at each stage
            peak_x, peak_y = i, i
            Z += factor * np.exp(-((X - peak_x)**2 + (Y - peak_y)**2) / 0.5)
        
        # Add current position marker
        current_stage_idx = stages.index(self.agent.current_stage)
        current_x = current_stage_idx
        current_y = current_stage_idx
        current_z = complexity_factors[current_stage_idx]
        
        # Create 3D plot
        fig = go.Figure()
        
        # Add complexity surface
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            name='Complexity Landscape',
            showscale=True,
            colorbar=dict(title='Complexity Factor')
        ))
        
        # Add current position
        fig.add_trace(go.Scatter3d(
            x=[current_x], y=[current_y], z=[current_z],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='diamond'
            ),
            name=f'Current: {self.agent.current_stage}',
            text=[f'Stage: {self.agent.current_stage}<br>Complexity: {current_z}x'],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add stage labels
        for i, (stage, factor) in enumerate(zip(stages, complexity_factors)):
            fig.add_trace(go.Scatter3d(
                x=[i], y=[i], z=[factor],
                mode='text',
                text=[stage],
                textposition='middle center',
                name=f'Stage {stage}',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title='🧠 Complexity Evolution Landscape - 3D View',
            scene=dict(
                xaxis_title='Development Dimension X',
                yaxis_title='Development Dimension Y',
                zaxis_title='Complexity Factor',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_evolution_timeline(self):
        """Create a timeline showing evolution progress"""
        # Get complexity analysis
        analysis = self.agent.get_complexity_analysis()
        
        # Create timeline data
        stages = ['F', 'N0', 'N1', 'N2', 'N3']
        stage_names = [
            'Basic Neural Dynamics',
            'Learning & Consolidation', 
            'Enhanced Control & Memory',
            'Meta-Control & Simulation',
            'Proto-Consciousness'
        ]
        
        # Create timeline plot
        fig = go.Figure()
        
        # Add stage markers
        for i, (stage, name) in enumerate(zip(stages, stage_names)):
            color = 'red' if stage == self.agent.current_stage else 'lightblue'
            size = 20 if stage == self.agent.current_stage else 15
            
            fig.add_trace(go.Scatter(
                x=[i], y=[0],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    symbol='circle'
                ),
                text=[stage],
                textposition='middle center',
                name=f'Stage {stage}',
                showlegend=False
            ))
            
            # Add stage labels
            fig.add_trace(go.Scatter(
                x=[i], y=[-0.5],
                mode='text',
                text=[name],
                textposition='middle center',
                name=f'Name {stage}',
                showlegend=False
            ))
        
        # Add current position indicator
        current_idx = stages.index(self.agent.current_stage)
        fig.add_trace(go.Scatter(
            x=[current_idx], y=[0.5],
            mode='markers+text',
            marker=dict(
                size=25,
                color='red',
                symbol='star'
            ),
            text=['CURRENT'],
            textposition='middle center',
            name='Current Position',
            showlegend=False
        ))
        
        # Add evolution path
        if analysis['evolution_ready']:
            next_idx = stages.index(analysis['next_stage'])
            fig.add_trace(go.Scatter(
                x=[current_idx, next_idx],
                y=[0.3, 0.3],
                mode='lines+markers',
                line=dict(color='green', width=3),
                marker=dict(size=8, color='green'),
                name='Evolution Path',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title='🔄 Complexity Evolution Timeline',
            xaxis=dict(
                title='Development Stage',
                range=[-0.5, len(stages) - 0.5],
                showticklabels=False
            ),
            yaxis=dict(
                title='',
                range=[-1, 1],
                showticklabels=False
            ),
            width=800,
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_complexity_metrics_dashboard(self):
        """Create a dashboard showing complexity metrics"""
        analysis = self.agent.get_complexity_analysis()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Complexity Factor', 'Document Depth', 'Technical Detail', 'Biological Accuracy'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Current vs Next stage data
        current = analysis['current_complexity']
        next_stage = analysis['next_complexity']
        
        # Complexity Factor
        fig.add_trace(
            go.Bar(
                x=['Current', 'Next'],
                y=[current['complexity_factor'], next_stage['complexity_factor']],
                name='Complexity Factor',
                marker_color=['lightblue', 'lightgreen']
            ),
            row=1, col=1
        )
        
        # Document Depth
        depth_mapping = {'foundational': 1, 'developmental': 2, 'advanced': 3, 'expert': 4, 'research_grade': 5}
        current_depth = depth_mapping.get(current['document_depth'], 1)
        next_depth = depth_mapping.get(next_stage['document_depth'], 2)
        
        fig.add_trace(
            go.Bar(
                x=['Current', 'Next'],
                y=[current_depth, next_depth],
                name='Document Depth',
                marker_color=['lightblue', 'lightgreen']
            ),
            row=1, col=2
        )
        
        # Technical Detail
        tech_mapping = {'basic': 1, 'intermediate': 2, 'advanced': 3, 'sophisticated': 4, 'research_frontier': 5}
        current_tech = tech_mapping.get(current['technical_detail'], 1)
        next_tech = tech_mapping.get(next_stage['technical_detail'], 2)
        
        fig.add_trace(
            go.Bar(
                x=['Current', 'Next'],
                y=[current_tech, next_tech],
                name='Technical Detail',
                marker_color=['lightblue', 'lightgreen']
            ),
            row=2, col=1
        )
        
        # Biological Accuracy
        bio_mapping = {'core_principles': 1, 'developmental_patterns': 2, 'sophisticated_models': 3, 'high_fidelity': 4, 'research_validation': 5}
        current_bio = bio_mapping.get(current['biological_accuracy'], 1)
        next_bio = bio_mapping.get(next_stage['biological_accuracy'], 2)
        
        fig.add_trace(
            go.Bar(
                x=['Current', 'Next'],
                y=[current_bio, next_bio],
                name='Biological Accuracy',
                marker_color=['lightblue', 'lightgreen']
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='📊 Complexity Metrics Dashboard',
            width=800,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_sync_status_visualization(self):
        """Create visualization of sync status"""
        sync_status = self.synchronizer.get_sync_status()
        
        # Create sync status indicators
        fig = go.Figure()
        
        # Add sync status indicators
        status_colors = {
            'no_syncs_performed': 'gray',
            'partial_sync': 'yellow', 
            'full_sync': 'green',
            'sync_error': 'red'
        }
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=sync_status.get('success_rate', 0) * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sync Success Rate (%)"},
            delta={'reference': 80},
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
        ))
        
        fig.update_layout(
            title='🔄 Connectome Synchronization Status',
            width=600,
            height=400
        )
        
        return fig
    
    def run_live_demo(self):
        """Run the live demo with real-time updates"""
        print("🚀 Starting Complexity Evolution Agent Live 3D Demo...")
        print("=" * 60)
        
        # Get initial analysis
        analysis = self.agent.get_complexity_analysis()
        
        print(f"🧠 Current Stage: {analysis['current_stage']} - {analysis['current_complexity']['name']}")
        print(f"📈 Complexity Factor: {analysis['current_complexity']['complexity_factor']}x")
        print(f"🔄 Evolution Ready: {analysis['evolution_ready']}")
        print(f"🎯 Next Stage: {analysis['next_stage']} - {analysis['next_complexity']['name']}")
        print(f"📊 Complexity Increase: {analysis['complexity_gap']['factor_increase']:.1f}x")
        
        print("\n" + "=" * 60)
        print("📊 Creating Live 3D Visualizations...")
        
        # Create all visualizations
        try:
            # 3D Complexity Landscape
            landscape_3d = self.create_3d_complexity_landscape()
            landscape_3d.write_html("testing/visualizations/complexity_evolution_3d_landscape.html")
            print("✅ 3D Complexity Landscape created")
            
            # Evolution Timeline
            timeline = self.create_evolution_timeline()
            timeline.write_html("testing/visualizations/complexity_evolution_timeline.html")
            print("✅ Evolution Timeline created")
            
            # Metrics Dashboard
            dashboard = self.create_complexity_metrics_dashboard()
            dashboard.write_html("testing/visualizations/complexity_metrics_dashboard.html")
            print("✅ Complexity Metrics Dashboard created")
            
            # Sync Status
            sync_viz = self.create_sync_status_visualization()
            sync_viz.write_html("testing/visualizations/sync_status_visualization.html")
            print("✅ Sync Status Visualization created")
            
            # Create comprehensive dashboard
            self.create_comprehensive_dashboard()
            print("✅ Comprehensive Dashboard created")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return
        
        print("\n" + "=" * 60)
        print("🎉 Demo Complete! Open the HTML files to see the visualizations:")
        print("   • 3D Complexity Landscape: complexity_evolution_3d_landscape.html")
        print("   • Evolution Timeline: complexity_evolution_timeline.html")
        print("   • Metrics Dashboard: complexity_metrics_dashboard.html")
        print("   • Sync Status: sync_status_visualization.html")
        print("   • Comprehensive Dashboard: complexity_evolution_dashboard.html")
        
        return analysis
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard combining all visualizations"""
        # Get analysis data
        analysis = self.agent.get_complexity_analysis()
        sync_status = self.synchronizer.get_sync_status()
        
        # Create comprehensive HTML dashboard
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🧠 Complexity Evolution Agent - Live Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .full-width {{ grid-column: 1 / -1; }}
        .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .status {{ padding: 5px 10px; border-radius: 5px; color: white; }}
        .status.ready {{ background-color: #28a745; }}
        .status.waiting {{ background-color: #ffc107; }}
        .status.error {{ background-color: #dc3545; }}
        iframe {{ width: 100%; height: 400px; border: none; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Complexity Evolution Agent - Live Dashboard</h1>
        <p>Real-time monitoring of Quark's complexity evolution and external synchronization</p>
        <p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="dashboard">
        <div class="card full-width">
            <h2>📊 Current Status</h2>
            <div class="metric">
                <span><strong>Current Stage:</strong></span>
                <span class="status ready">{analysis['current_stage']} - {analysis['current_complexity']['name']}</span>
            </div>
            <div class="metric">
                <span><strong>Complexity Factor:</strong></span>
                <span>{analysis['current_complexity']['complexity_factor']}x</span>
            </div>
            <div class="metric">
                <span><strong>Evolution Ready:</strong></span>
                <span class="status {'ready' if analysis['evolution_ready'] else 'waiting'}">
                    {'✅ Ready' if analysis['evolution_ready'] else '⏳ Waiting'}
                </span>
            </div>
            <div class="metric">
                <span><strong>Next Stage:</strong></span>
                <span>{analysis['next_stage']} - {analysis['next_complexity']['name']}</span>
            </div>
            <div class="metric">
                <span><strong>Complexity Increase:</strong></span>
                <span>{analysis['complexity_gap']['factor_increase']:.1f}x</span>
            </div>
        </div>
        
        <div class="card">
            <h2>🔄 Connectome Sync Status</h2>
            <div class="metric">
                <span><strong>Status:</strong></span>
                <span class="status {'ready' if sync_status['success_rate'] > 0.8 else 'waiting'}">
                    {sync_status['status'].replace('_', ' ').title()}
                </span>
            </div>
            <div class="metric">
                <span><strong>Success Rate:</strong></span>
                <span>{sync_status['success_rate']:.1%}</span>
            </div>
            <div class="metric">
                <span><strong>Last Sync:</strong></span>
                <span>{sync_status['last_sync'] or 'Never'}</span>
            </div>
            <div class="metric">
                <span><strong>Avg Validation:</strong></span>
                <span>{sync_status['avg_validation_score']:.2f}</span>
            </div>
        </div>
        
        <div class="card">
            <h2>📈 Complexity Metrics</h2>
            <div class="metric">
                <span><strong>Document Depth:</strong></span>
                <span>{analysis['current_complexity']['document_depth']} → {analysis['next_complexity']['document_depth']}</span>
            </div>
            <div class="metric">
                <span><strong>Technical Detail:</strong></span>
                <span>{analysis['current_complexity']['technical_detail']} → {analysis['next_complexity']['technical_detail']}</span>
            </div>
            <div class="metric">
                <span><strong>Biological Accuracy:</strong></span>
                <span>{analysis['current_complexity']['biological_accuracy']} → {analysis['next_complexity']['biological_accuracy']}</span>
            </div>
            <div class="metric">
                <span><strong>ML Sophistication:</strong></span>
                <span>{analysis['current_complexity']['ml_sophistication']} → {analysis['next_complexity']['ml_sophistication']}</span>
            </div>
        </div>
        
        <div class="card full-width">
            <h2>🎯 3D Complexity Landscape</h2>
            <iframe src="complexity_evolution_3d_landscape.html"></iframe>
        </div>
        
        <div class="card">
            <h2>🔄 Evolution Timeline</h2>
            <iframe src="complexity_evolution_timeline.html"></iframe>
        </div>
        
        <div class="card">
            <h2>📊 Metrics Dashboard</h2>
            <iframe src="complexity_metrics_dashboard.html"></iframe>
        </div>
        
        <div class="card full-width">
            <h2>🚀 Recommendations</h2>
            <ul>
                <li><strong>Current Focus:</strong> Continue development in {analysis['current_stage']} stage</li>
                <li><strong>Next Milestone:</strong> Prepare for transition to {analysis['next_stage']} stage</li>
                <li><strong>Complexity Target:</strong> Increase complexity by {analysis['complexity_gap']['factor_increase']:.1f}x</li>
                <li><strong>Document Enhancement:</strong> Upgrade from {analysis['current_complexity']['document_depth']} to {analysis['next_complexity']['document_depth']}</li>
                <li><strong>Technical Upgrade:</strong> Advance from {analysis['current_complexity']['technical_detail']} to {analysis['next_complexity']['technical_detail']}</li>
            </ul>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setInterval(() => {{
            location.reload();
        }}, 30000);
    </script>
</body>
</html>
        """
        
        # Save the dashboard
        with open("testing/visualizations/complexity_evolution_dashboard.html", "w") as f:
            f.write(dashboard_html)

def main():
    """Main demo function"""
    demo = ComplexityEvolutionLiveDemo()
    return demo.run_live_demo()

if __name__ == "__main__":
    main()
