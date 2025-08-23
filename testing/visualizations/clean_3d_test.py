#!/usr/bin/env python3
"""
Clean 3D Test - Single dashboard, guaranteed 3D visualization display.
"""

import time
import sys
import os
import webbrowser

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from testing.visualizations.live_3d_visualizer import get_3d_visualizer
from testing.visualizations.visual_utils import start_live_server

def clean_3d_test():
    """Clean 3D visualization test with single dashboard."""
    print("🧪 Clean 3D Test")
    print("=" * 30)
    
    # Start server
    print("🚀 Starting server...")
    start_live_server()
    time.sleep(2)
    
    # Open ONLY the clean dashboard
    dashboard_path = os.path.abspath("testing/visualizations/clean_3d_dashboard.html")
    print(f"🌐 Opening clean dashboard: {dashboard_path}")
    webbrowser.open(f"file://{dashboard_path}")
    
    # Wait for browser to open
    time.sleep(3)
    
    # Get visualizer
    visualizer = get_3d_visualizer()
    
    print("\n📊 Creating 3D visualizations...")
    
    # Create multiple 3D visualizations
    for i in range(5):
        print(f"\n🎯 Creating 3D visualization {i+1}/5...")
        
        # Create test data
        test_data = [
            {"name": f"test_{i+1}", "duration": 0.001 + i * 0.002, "status": "PASSED"},
            {"name": f"test_{i+1}_b", "duration": 0.002 + i * 0.001, "status": "FAILED" if i % 2 == 0 else "PASSED"}
        ]
        
        # Stream 3D visualization
        fig = visualizer.stream_3d_visualization("test_landscape", test_data)
        print(f"✅ 3D visualization {i+1} created and streamed!")
        
        # Wait between visualizations
        time.sleep(2)
    
    print("\n🎉 All 3D visualizations completed!")
    print("🌐 Check the clean dashboard in your browser!")
    print("📱 You should see 5 interactive 3D visualizations")
    
    # Keep server running
    print("\n⏳ Keeping server running for 20 seconds...")
    time.sleep(20)
    
    print("✅ Clean 3D test completed!")

if __name__ == "__main__":
    clean_3d_test()
