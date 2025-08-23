"""Utility helpers to generate and persist live visualisations for tests.

Every figure is saved as both PNG and self-contained HTML in
`testing/visualizations/outputs/` so CI can upload them and developers
can open them locally.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
from pathlib import Path as _Path
from typing import Sequence, Dict

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import webbrowser
from datetime import datetime
import sys
import os

# Add the live stream server to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    # Import just to check availability
    import testing.visualizations.working_live_server
    LIVE_SERVER_AVAILABLE = True
except ImportError:
    LIVE_SERVER_AVAILABLE = False
    print("⚠️ Live streaming server not available")

def plot_series(x, y, title="Time Series", xlabel="Time", ylabel="Value"):
    """Create a Plotly line plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=title))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_dark"
    )
    return fig

def bar_chart(categories, values, title="Bar Chart", xlabel="Categories", ylabel="Values"):
    """Create a Plotly bar chart"""
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color='lightblue')
    ])
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_dark"
    )
    return fig

def save_fig(fig, stem, save_dir="testing/visualizations/outputs"):
    """Save figure as both PNG and interactive HTML"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as PNG
    png_path = os.path.join(save_dir, f"{stem}_{timestamp}.png")
    fig.write_image(png_path)
    print(f"📊 Saved PNG: {png_path}")
    
    # Save as interactive HTML
    html_path = os.path.join(save_dir, f"{stem}_{timestamp}.html")
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"🌐 Saved HTML: {html_path}")
    
    # Don't auto-open HTML to prevent multiple browser instances
    print(f"🌐 HTML available at: file://{os.path.abspath(html_path)}")
    print("📱 Open manually in browser to avoid multiple instances")
    
    return png_path, html_path

def live_series(series_id, value, step):
    """Stream data point to live dashboard"""
    if not LIVE_SERVER_AVAILABLE:
        return
    
    try:
        # Import here to avoid circular imports
        from testing.visualizations.working_live_server import working_live_series
        working_live_series(series_id, value, step)
    except Exception as e:
        print(f"⚠️ Live streaming error: {e}")

def start_live_server():
    """Start the live streaming server and open dashboard"""
    if not LIVE_SERVER_AVAILABLE:
        print("❌ Live streaming server not available")
        return None
    
    try:
        # Import here to avoid circular imports
        from testing.visualizations.working_live_server import start_working_live_server
        return start_working_live_server()
    except Exception as e:
        print(f"❌ Error starting live server: {e}")
        return None

# Legacy matplotlib functions for compatibility
def plot_matplotlib(x, y, title="Plot", save_path=None):
    """Create matplotlib plot (legacy)"""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    
    plt.show()
    return plt.gcf()
