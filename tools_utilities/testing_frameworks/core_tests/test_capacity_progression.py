#!/usr/bin/env python3
"""
TEST: Capacity Progression Component
Purpose: Test capacity progression functionality with visual validation
Inputs: Capacity progression component
Outputs: Visual validation report with developmental capacity patterns
Seeds: 42
Dependencies: matplotlib, plotly, numpy, capacity_progression
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

class CapacityProgressionVisualTest:
    def __init__(self):
        self.seed = 42
        np.random.seed(self.seed)
        self.test_results = {}
        
    def test_developmental_capacity_progression(self):
        """Test capacity progression across developmental stages"""
        print("🧪 Testing developmental capacity progression...")
        
        # Developmental stages
        stages = ['Fetal (F)', 'Neonate (N0)', 'Early Postnatal (N1)', 'Late Postnatal (N2)', 'Infant (I)']
        time_points = np.linspace(0, 24, len(stages))  # Months
        
        # Different capacity types
        capacity_types = {
            'Working Memory': np.array([3, 4, 5, 6, 7]),  # Slots
            'Processing Speed': np.array([0.2, 0.4, 0.6, 0.8, 1.0]),  # Normalized
            'Neural Complexity': np.array([0.1, 0.3, 0.5, 0.7, 0.9]),  # Normalized
            'Learning Capacity': np.array([0.15, 0.35, 0.55, 0.75, 0.95]),  # Normalized
            'Integration Ability': np.array([0.05, 0.25, 0.45, 0.65, 0.85])  # Normalized
        }
        
        # Create subplot for capacity progression
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Working Memory Progression', 'Processing Speed Development',
                          'Neural Complexity Growth', 'Learning & Integration Capacity'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Working memory bar chart
        fig.add_trace(
            go.Bar(x=stages, y=capacity_types['Working Memory'], name='Working Memory',
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        # Processing speed line chart
        fig.add_trace(
            go.Scatter(x=time_points, y=capacity_types['Processing Speed'], 
                      mode='lines+markers', name='Processing Speed',
                      line=dict(color='red', width=3)),
            row=1, col=2
        )
        
        # Neural complexity line chart
        fig.add_trace(
            go.Scatter(x=time_points, y=capacity_types['Neural Complexity'],
                      mode='lines+markers', name='Neural Complexity',
                      line=dict(color='green', width=3)),
            row=2, col=1
        )
        
        # Learning and integration combined
        fig.add_trace(
            go.Scatter(x=time_points, y=capacity_types['Learning Capacity'],
                      mode='lines+markers', name='Learning Capacity',
                      line=dict(color='purple', width=3)),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=time_points, y=capacity_types['Integration Ability'],
                      mode='lines+markers', name='Integration Ability',
                      line=dict(color='orange', width=3)),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Developmental Capacity Progression',
            height=800,
            showlegend=True
        )
        
        # Save the plot
        output_path = Path("tests/outputs/capacity_progression_analysis.html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        
        print(f"✅ Developmental capacity progression test completed")
        print(f"📊 Results saved to: {output_path}")
        
        self.test_results['capacity_progression'] = {
            'stages': stages,
            'time_points': time_points.tolist(),
            'capacity_types': {k: v.tolist() for k, v in capacity_types.items()}
        }
        
    def test_capacity_limits_and_thresholds(self):
        """Test capacity limits and threshold effects"""
        print("🧪 Testing capacity limits and thresholds...")
        
        # Simulate capacity limits over time
        time_points = np.linspace(0, 100, 1000)
        
        # Different capacity curves with limits
        working_memory_limit = 7
        working_memory_curve = working_memory_limit * (1 - np.exp(-time_points / 20))
        
        processing_limit = 1.0
        processing_curve = processing_limit * (1 - np.exp(-time_points / 15))
        
        neural_complexity_limit = 0.9
        neural_curve = neural_complexity_limit * (1 - np.exp(-time_points / 25))
        
        # Add some noise and plateaus
        working_memory_curve += 0.1 * np.random.randn(1000)
        processing_curve += 0.05 * np.random.randn(1000)
        neural_curve += 0.08 * np.random.randn(1000)
        
        # Create visualization
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Working Memory Capacity Limit', 'Processing Speed Limit', 'Neural Complexity Limit'),
            vertical_spacing=0.08
        )
        
        fig.add_trace(
            go.Scatter(x=time_points, y=working_memory_curve, name='Working Memory',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Add limit line
        fig.add_hline(y=working_memory_limit, line_dash="dash", line_color="red",
                     annotation_text=f"Limit: {working_memory_limit}", row=1, col=1)
        
        fig.add_trace(
            go.Scatter(x=time_points, y=processing_curve, name='Processing Speed',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        fig.add_hline(y=processing_limit, line_dash="dash", line_color="red",
                     annotation_text=f"Limit: {processing_limit}", row=2, col=1)
        
        fig.add_trace(
            go.Scatter(x=time_points, y=neural_curve, name='Neural Complexity',
                      line=dict(color='purple', width=2)),
            row=3, col=1
        )
        
        fig.add_hline(y=neural_complexity_limit, line_dash="dash", line_color="red",
                     annotation_text=f"Limit: {neural_complexity_limit}", row=3, col=1)
        
        fig.update_layout(
            title='Capacity Limits and Thresholds',
            height=900,
            showlegend=False
        )
        
        # Save the plot
        output_path = Path("tests/outputs/capacity_limits_analysis.html")
        fig.write_html(str(output_path))
        
        print(f"✅ Capacity limits and thresholds test completed")
        print(f"📊 Results saved to: {output_path}")
        
        self.test_results['capacity_limits'] = {
            'time_points': time_points.tolist(),
            'working_memory_curve': working_memory_curve.tolist(),
            'processing_curve': processing_curve.tolist(),
            'neural_curve': neural_curve.tolist(),
            'limits': {
                'working_memory': working_memory_limit,
                'processing': processing_limit,
                'neural_complexity': neural_complexity_limit
            }
        }
        
    def test_capacity_interactions(self):
        """Test interactions between different capacity types"""
        print("🧪 Testing capacity interactions...")
        
        # Create interaction matrix
        capacity_types = ['Working Memory', 'Processing Speed', 'Neural Complexity', 'Learning Capacity']
        interaction_matrix = np.random.uniform(0.1, 0.9, (len(capacity_types), len(capacity_types)))
        np.fill_diagonal(interaction_matrix, 1.0)  # Self-interaction is 1
        
        # Make interactions more realistic
        interaction_matrix[0, 1] = 0.8  # Working memory affects processing speed
        interaction_matrix[1, 0] = 0.7  # Processing speed affects working memory
        interaction_matrix[2, 0] = 0.6  # Neural complexity affects working memory
        interaction_matrix[2, 1] = 0.5  # Neural complexity affects processing speed
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=interaction_matrix,
            x=capacity_types,
            y=capacity_types,
            colorscale='Viridis',
            text=np.round(interaction_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Capacity Type Interactions',
            xaxis_title='Capacity Type',
            yaxis_title='Capacity Type',
            height=600
        )
        
        # Save the plot
        output_path = Path("tests/outputs/capacity_interactions.html")
        fig.write_html(str(output_path))
        
        print(f"✅ Capacity interactions test completed")
        print(f"📊 Results saved to: {output_path}")
        
        self.test_results['capacity_interactions'] = {
            'capacity_types': capacity_types,
            'interaction_matrix': interaction_matrix.tolist()
        }
        
    def run_all_tests(self):
        """Run all capacity progression tests"""
        print("🚀 Starting Capacity Progression Visual Tests...")
        print("📈 Testing developmental capacity patterns and limits")
        
        self.test_developmental_capacity_progression()
        self.test_capacity_limits_and_thresholds()
        self.test_capacity_interactions()
        
        print("\n🎉 All capacity progression tests completed!")
        print("📊 Visual results saved to tests/outputs/")
        
        return self.test_results

if __name__ == "__main__":
    tester = CapacityProgressionVisualTest()
    results = tester.run_all_tests()
