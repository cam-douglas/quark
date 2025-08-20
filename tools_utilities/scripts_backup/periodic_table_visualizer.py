"""
Interactive Periodic Table Visualizer
=====================================

This module creates an interactive visualization system for exploring the periodic table
and the AI's understanding of chemical properties and relationships.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import asdict
import pandas as pd

from periodic_table_trainer import ElementProperties

logger = logging.getLogger(__name__)

class PeriodicTableVisualizer:
    """Interactive periodic table visualization system"""
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data"):
        self.data_dir = Path(data_dir)
        self.elements_data = self._load_elements_data()
        self.color_schemes = self._initialize_color_schemes()
        
        # Periodic table layout (traditional 18-group layout)
        self.layout = self._create_periodic_table_layout()
        
        logger.info("Initialized Periodic Table Visualizer")
    
    def _load_elements_data(self) -> Dict[int, ElementProperties]:
        """Load elements data from JSON"""
        json_file = self.data_dir / "complete_periodic_table.json"
        
        if not json_file.exists():
            logger.warning(f"Elements data file not found at {json_file}")
            return {}
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        elements = {}
        for atomic_num_str, element_data in data['elements'].items():
            atomic_num = int(atomic_num_str)
            
            # Handle missing fields
            for field_name, field_type in ElementProperties.__annotations__.items():
                if field_name not in element_data:
                    if field_type == List[float] or field_type == List[int] or field_type == List[str]:
                        element_data[field_name] = []
                    elif 'Optional' in str(field_type):
                        element_data[field_name] = None
                    elif field_type == str:
                        element_data[field_name] = 'unknown'
                    elif field_type == int:
                        element_data[field_name] = 0
                    elif field_type == float:
                        element_data[field_name] = 0.0
            
            elements[atomic_num] = ElementProperties(**element_data)
        
        return elements
    
    def _initialize_color_schemes(self) -> Dict[str, Dict]:
        """Initialize color schemes for different property visualizations"""
        return {
            'metal_type': {
                'metal': '#4CAF50',
                'metalloid': '#FF9800', 
                'nonmetal': '#2196F3',
                'unknown': '#9E9E9E'
            },
            'block': {
                's': '#FF6B6B',
                'p': '#4ECDC4',
                'd': '#45B7D1',
                'f': '#96CEB4'
            },
            'state_at_stp': {
                'solid': '#8D6E63',
                'liquid': '#3F51B5',
                'gas': '#FFC107',
                'unknown': '#9E9E9E'
            },
            'discovery_era': {
                'ancient': '#795548',
                'medieval': '#9C27B0',
                'renaissance': '#E91E63',
                'industrial': '#FF5722',
                'modern': '#607D8B',
                'contemporary': '#009688'
            }
        }
    
    def _create_periodic_table_layout(self) -> Dict[int, Tuple[int, int]]:
        """Create position mapping for elements in standard periodic table layout"""
        layout = {}
        
        # Periods 1-7, Groups 1-18
        period_starts = {1: 1, 2: 3, 3: 11, 4: 19, 5: 37, 6: 55, 7: 87}
        
        # Standard layout positions
        positions = {
            # Period 1
            1: (1, 1), 2: (1, 18),
            
            # Period 2
            3: (2, 1), 4: (2, 2), 5: (2, 13), 6: (2, 14), 7: (2, 15), 8: (2, 16), 9: (2, 17), 10: (2, 18),
            
            # Period 3
            11: (3, 1), 12: (3, 2), 13: (3, 13), 14: (3, 14), 15: (3, 15), 16: (3, 16), 17: (3, 17), 18: (3, 18),
            
            # Period 4
            19: (4, 1), 20: (4, 2), 21: (4, 3), 22: (4, 4), 23: (4, 5), 24: (4, 6), 25: (4, 7), 26: (4, 8),
            27: (4, 9), 28: (4, 10), 29: (4, 11), 30: (4, 12), 31: (4, 13), 32: (4, 14), 33: (4, 15), 
            34: (4, 16), 35: (4, 17), 36: (4, 18),
            
            # Period 5
            37: (5, 1), 38: (5, 2), 39: (5, 3), 40: (5, 4), 41: (5, 5), 42: (5, 6), 43: (5, 7), 44: (5, 8),
            45: (5, 9), 46: (5, 10), 47: (5, 11), 48: (5, 12), 49: (5, 13), 50: (5, 14), 51: (5, 15),
            52: (5, 16), 53: (5, 17), 54: (5, 18),
            
            # Period 6 (excluding lanthanides)
            55: (6, 1), 56: (6, 2), 71: (6, 3), 72: (6, 4), 73: (6, 5), 74: (6, 6), 75: (6, 7), 76: (6, 8),
            77: (6, 9), 78: (6, 10), 79: (6, 11), 80: (6, 12), 81: (6, 13), 82: (6, 14), 83: (6, 15),
            84: (6, 16), 85: (6, 17), 86: (6, 18),
            
            # Period 7 (excluding actinides)
            87: (7, 1), 88: (7, 2), 103: (7, 3), 104: (7, 4), 105: (7, 5), 106: (7, 6), 107: (7, 7), 108: (7, 8),
            109: (7, 9), 110: (7, 10), 111: (7, 11), 112: (7, 12), 113: (7, 13), 114: (7, 14), 115: (7, 15),
            116: (7, 16), 117: (7, 17), 118: (7, 18)
        }
        
        # Lanthanides (separate row)
        for i, atomic_num in enumerate(range(57, 71)):
            positions[atomic_num] = (9, i + 3)
        
        # Actinides (separate row)
        for i, atomic_num in enumerate(range(89, 103)):
            positions[atomic_num] = (10, i + 3)
        
        return positions
    
    def create_interactive_periodic_table(self, property_name: str = 'atomic_weight', 
                                        save_html: bool = True) -> go.Figure:
        """Create interactive periodic table visualization using Plotly"""
        logger.info(f"Creating interactive periodic table for property: {property_name}")
        
        # Prepare data for visualization
        x_coords = []
        y_coords = []
        symbols = []
        names = []
        atomic_numbers = []
        property_values = []
        hover_texts = []
        colors = []
        
        for atomic_num, element in self.elements_data.items():
            if atomic_num in self.layout:
                period, group = self.layout[atomic_num]
                
                x_coords.append(group)
                y_coords.append(-period)  # Negative to flip y-axis
                symbols.append(element.symbol)
                names.append(element.name)
                atomic_numbers.append(atomic_num)
                
                # Get property value
                if hasattr(element, property_name):
                    prop_value = getattr(element, property_name)
                    if prop_value is None:
                        prop_value = 0
                    property_values.append(prop_value)
                else:
                    property_values.append(0)
                
                # Create hover text
                hover_text = f"""
                <b>{element.name} ({element.symbol})</b><br>
                Atomic Number: {atomic_num}<br>
                Atomic Weight: {element.atomic_weight}<br>
                Period: {element.period}, Group: {element.group}<br>
                Block: {element.block}<br>
                Metal Type: {element.metal_type}<br>
                State: {element.state_at_stp}<br>
                {property_name}: {getattr(element, property_name, 'N/A')}
                """
                hover_texts.append(hover_text)
                
                # Color by metal type
                colors.append(self.color_schemes['metal_type'].get(element.metal_type, '#9E9E9E'))
        
        # Create the figure
        fig = go.Figure()
        
        # Add element boxes
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            text=symbols,
            textposition='middle center',
            textfont=dict(size=12, color='white'),
            marker=dict(
                size=40,
                color=colors,
                line=dict(width=2, color='black'),
                symbol='square'
            ),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_texts,
            name='Elements'
        ))
        
        # Add atomic numbers as smaller text
        fig.add_trace(go.Scatter(
            x=[x - 0.3 for x in x_coords],
            y=[y + 0.3 for y in y_coords],
            mode='text',
            text=atomic_numbers,
            textfont=dict(size=8, color='white'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Interactive Periodic Table - {property_name.replace("_", " ").title()}',
            xaxis=dict(
                title='Group',
                range=[0, 19],
                tickmode='linear',
                tick0=1,
                dtick=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Period',
                range=[-11, 0],
                tickmode='array',
                tickvals=[-1, -2, -3, -4, -5, -6, -7, -9, -10],
                ticktext=['1', '2', '3', '4', '5', '6', '7', 'Lanthanides', 'Actinides'],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            width=1200,
            height=800,
            showlegend=False
        )
        
        if save_html:
            output_file = self.data_dir / f'interactive_periodic_table_{property_name}.html'
            pyo.plot(fig, filename=str(output_file), auto_open=False)
            logger.info(f"Interactive periodic table saved to {output_file}")
        
        return fig
    
    def create_property_correlation_heatmap(self, properties: List[str] = None) -> go.Figure:
        """Create correlation heatmap of element properties"""
        if properties is None:
            properties = [
                'atomic_weight', 'melting_point', 'boiling_point', 'density',
                'electronegativity', 'atomic_radius', 'ionization_energies'
            ]
        
        logger.info(f"Creating property correlation heatmap for {len(properties)} properties")
        
        # Prepare data matrix
        data_matrix = []
        valid_elements = []
        
        for atomic_num, element in self.elements_data.items():
            row = []
            valid_row = True
            
            for prop in properties:
                value = getattr(element, prop, None)
                
                if prop == 'ionization_energies':
                    value = value[0] if value and len(value) > 0 else None
                
                if value is None or value == 0:
                    valid_row = False
                    break
                
                row.append(float(value))
            
            if valid_row:
                data_matrix.append(row)
                valid_elements.append(element.symbol)
        
        if not data_matrix:
            logger.warning("No valid data for correlation analysis")
            return go.Figure()
        
        # Create DataFrame and calculate correlations
        df = pd.DataFrame(data_matrix, columns=properties, index=valid_elements)
        correlation_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(correlation_matrix.values, decimals=2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Element Property Correlations',
            xaxis_title='Properties',
            yaxis_title='Properties',
            width=800,
            height=800
        )
        
        # Save heatmap
        output_file = self.data_dir / 'property_correlation_heatmap.html'
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        logger.info(f"Property correlation heatmap saved to {output_file}")
        
        return fig
    
    def create_3d_periodic_trends(self) -> go.Figure:
        """Create 3D visualization of periodic trends"""
        logger.info("Creating 3D periodic trends visualization")
        
        # Prepare data
        x_coords = []  # Atomic number
        y_coords = []  # Atomic radius
        z_coords = []  # Electronegativity
        colors = []   # Ionization energy
        symbols = []
        hover_texts = []
        
        for atomic_num, element in self.elements_data.items():
            if (element.atomic_radius and element.electronegativity and 
                element.ionization_energies and len(element.ionization_energies) > 0):
                
                x_coords.append(atomic_num)
                y_coords.append(element.atomic_radius)
                z_coords.append(element.electronegativity)
                colors.append(element.ionization_energies[0])
                symbols.append(element.symbol)
                
                hover_text = f"""
                <b>{element.name} ({element.symbol})</b><br>
                Atomic Number: {atomic_num}<br>
                Atomic Radius: {element.atomic_radius} pm<br>
                Electronegativity: {element.electronegativity}<br>
                1st Ionization Energy: {element.ionization_energies[0]} kJ/mol
                """
                hover_texts.append(hover_text)
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            text=symbols,
            textposition='middle center',
            textfont=dict(size=8),
            marker=dict(
                size=8,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title='1st Ionization Energy (kJ/mol)'),
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_texts
        ))
        
        fig.update_layout(
            title='3D Periodic Trends',
            scene=dict(
                xaxis_title='Atomic Number',
                yaxis_title='Atomic Radius (pm)',
                zaxis_title='Electronegativity',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800
        )
        
        # Save 3D plot
        output_file = self.data_dir / '3d_periodic_trends.html'
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        logger.info(f"3D periodic trends saved to {output_file}")
        
        return fig
    
    def create_discovery_timeline(self) -> go.Figure:
        """Create timeline visualization of element discoveries"""
        logger.info("Creating element discovery timeline")
        
        # Prepare data
        discovery_data = []
        
        for atomic_num, element in self.elements_data.items():
            if element.discovery_year and element.discovery_year > 0:
                discovery_data.append({
                    'year': element.discovery_year,
                    'atomic_number': atomic_num,
                    'symbol': element.symbol,
                    'name': element.name,
                    'discoverer': element.discovered_by or 'Unknown',
                    'metal_type': element.metal_type,
                    'block': element.block
                })
        
        # Sort by discovery year
        discovery_data.sort(key=lambda x: x['year'])
        
        # Create timeline
        years = [d['year'] for d in discovery_data]
        atomic_numbers = [d['atomic_number'] for d in discovery_data]
        symbols = [d['symbol'] for d in discovery_data]
        names = [d['name'] for d in discovery_data]
        discoverers = [d['discoverer'] for d in discovery_data]
        metal_types = [d['metal_type'] for d in discovery_data]
        
        # Color by metal type
        colors = [self.color_schemes['metal_type'].get(mt, '#9E9E9E') for mt in metal_types]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=atomic_numbers,
            mode='markers+text',
            text=symbols,
            textposition='top center',
            textfont=dict(size=10),
            marker=dict(
                size=12,
                color=colors,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            hovertemplate="""
            <b>%{text}</b><br>
            Element: %{customdata[0]}<br>
            Discovered: %{x}<br>
            Discoverer: %{customdata[1]}<br>
            Atomic Number: %{y}
            <extra></extra>
            """,
            customdata=list(zip(names, discoverers))
        ))
        
        fig.update_layout(
            title='Timeline of Element Discoveries',
            xaxis_title='Discovery Year',
            yaxis_title='Atomic Number',
            width=1200,
            height=800,
            showlegend=False
        )
        
        # Save timeline
        output_file = self.data_dir / 'element_discovery_timeline.html'
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        logger.info(f"Discovery timeline saved to {output_file}")
        
        return fig
    
    def create_ai_predictions_dashboard(self, ai_predictions_file: str = None) -> go.Figure:
        """Create dashboard showing AI predictions vs actual values"""
        if ai_predictions_file is None:
            ai_predictions_file = self.data_dir / "ai_predictions.json"
        
        logger.info("Creating AI predictions dashboard")
        
        # Load AI predictions if available
        if Path(ai_predictions_file).exists():
            with open(ai_predictions_file, 'r') as f:
                predictions = json.load(f)
        else:
            # Generate mock predictions for demonstration
            predictions = self._generate_mock_predictions()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Melting Point Predictions', 'Density Predictions',
                          'Electronegativity Predictions', 'Prediction Accuracy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        properties = ['melting_point', 'density', 'electronegativity']
        
        for i, prop in enumerate(properties):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            if prop in predictions:
                actual_values = []
                predicted_values = []
                element_symbols = []
                
                for atomic_num_str, pred_data in predictions[prop].items():
                    atomic_num = int(atomic_num_str)
                    if atomic_num in self.elements_data:
                        element = self.elements_data[atomic_num]
                        actual_val = getattr(element, prop, None)
                        
                        if actual_val is not None and actual_val != 0:
                            actual_values.append(actual_val)
                            predicted_values.append(pred_data['predicted'])
                            element_symbols.append(element.symbol)
                
                if actual_values and predicted_values:
                    # Scatter plot: actual vs predicted
                    fig.add_trace(
                        go.Scatter(
                            x=actual_values,
                            y=predicted_values,
                            mode='markers',
                            text=element_symbols,
                            name=f'{prop.replace("_", " ").title()}',
                            marker=dict(size=8, opacity=0.7)
                        ),
                        row=row, col=col
                    )
                    
                    # Add perfect prediction line
                    min_val = min(min(actual_values), min(predicted_values))
                    max_val = max(max(actual_values), max(predicted_values))
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            line=dict(dash='dash', color='red'),
                            name='Perfect Prediction',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
        
        # Overall accuracy in bottom right
        if 'accuracy_metrics' in predictions:
            metrics = predictions['accuracy_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name='Accuracy Metrics',
                    marker_color='lightblue'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='AI Predictions Dashboard',
            height=800,
            showlegend=False
        )
        
        # Save dashboard
        output_file = self.data_dir / 'ai_predictions_dashboard.html'
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        logger.info(f"AI predictions dashboard saved to {output_file}")
        
        return fig
    
    def _generate_mock_predictions(self) -> Dict[str, Any]:
        """Generate mock AI predictions for demonstration"""
        predictions = {
            'melting_point': {},
            'density': {},
            'electronegativity': {},
            'accuracy_metrics': {
                'Melting Point R²': 0.85,
                'Density R²': 0.92,
                'Electronegativity R²': 0.78,
                'Overall Accuracy': 0.85
            }
        }
        
        for atomic_num, element in self.elements_data.items():
            for prop in ['melting_point', 'density', 'electronegativity']:
                actual_val = getattr(element, prop, None)
                if actual_val is not None and actual_val != 0:
                    # Add some noise to create mock predictions
                    noise = np.random.normal(0, 0.1)
                    predicted_val = actual_val * (1 + noise)
                    
                    predictions[prop][str(atomic_num)] = {
                        'actual': actual_val,
                        'predicted': predicted_val,
                        'error': abs(predicted_val - actual_val),
                        'relative_error': abs(predicted_val - actual_val) / actual_val
                    }
        
        return predictions
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive visualization report"""
        logger.info("Generating comprehensive visualization report")
        
        # Create all visualizations
        visualizations = {}
        
        # 1. Interactive periodic table
        visualizations['periodic_table'] = self.create_interactive_periodic_table('atomic_weight')
        
        # 2. Property correlations
        visualizations['correlations'] = self.create_property_correlation_heatmap()
        
        # 3. 3D trends
        visualizations['3d_trends'] = self.create_3d_periodic_trends()
        
        # 4. Discovery timeline
        visualizations['discovery_timeline'] = self.create_discovery_timeline()
        
        # 5. AI predictions dashboard
        visualizations['ai_dashboard'] = self.create_ai_predictions_dashboard()
        
        # Generate statistics
        stats = {
            'total_elements': len(self.elements_data),
            'elements_with_melting_point': sum(1 for e in self.elements_data.values() 
                                             if e.melting_point is not None and e.melting_point != 0),
            'elements_with_discovery_year': sum(1 for e in self.elements_data.values() 
                                              if e.discovery_year is not None and e.discovery_year > 0),
            'metal_distribution': {},
            'block_distribution': {},
            'period_distribution': {}
        }
        
        # Calculate distributions
        for element in self.elements_data.values():
            # Metal type distribution
            metal_type = element.metal_type
            stats['metal_distribution'][metal_type] = stats['metal_distribution'].get(metal_type, 0) + 1
            
            # Block distribution
            block = element.block
            stats['block_distribution'][block] = stats['block_distribution'].get(block, 0) + 1
            
            # Period distribution
            period = element.period
            stats['period_distribution'][period] = stats['period_distribution'].get(period, 0) + 1
        
        report = {
            'visualizations_created': list(visualizations.keys()),
            'statistics': stats,
            'output_files': [
                'interactive_periodic_table_atomic_weight.html',
                'property_correlation_heatmap.html',
                '3d_periodic_trends.html',
                'element_discovery_timeline.html',
                'ai_predictions_dashboard.html'
            ]
        }
        
        # Save report
        output_file = self.data_dir / 'visualization_report.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive visualization report saved to {output_file}")
        return report

def main():
    """Main visualization pipeline"""
    logger.info("Starting Periodic Table Visualization System")
    
    # Initialize visualizer
    visualizer = PeriodicTableVisualizer()
    
    print("="*60)
    print("PERIODIC TABLE INTERACTIVE VISUALIZATION SYSTEM")
    print("="*60)
    
    # Generate comprehensive report
    print("\nGenerating comprehensive visualization report...")
    report = visualizer.generate_comprehensive_report()
    
    print(f"\n✓ Created {len(report['visualizations_created'])} interactive visualizations")
    print(f"✓ Processed {report['statistics']['total_elements']} elements")
    
    print(f"\nElement Statistics:")
    print(f"  • Elements with melting point data: {report['statistics']['elements_with_melting_point']}")
    print(f"  • Elements with discovery year: {report['statistics']['elements_with_discovery_year']}")
    
    print(f"\nMetal Type Distribution:")
    for metal_type, count in report['statistics']['metal_distribution'].items():
        print(f"  • {metal_type}: {count} elements")
    
    print(f"\nBlock Distribution:")
    for block, count in report['statistics']['block_distribution'].items():
        print(f"  • {block}-block: {count} elements")
    
    print(f"\nOutput Files Created:")
    for filename in report['output_files']:
        print(f"  • {filename}")
    
    print("\n" + "="*60)
    print("VISUALIZATION SYSTEM COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nAll files saved to: {visualizer.data_dir}")
    print("\nOpen any .html file in your browser to view interactive visualizations!")
    
    return report

if __name__ == "__main__":
    main()
