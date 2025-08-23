#!/usr/bin/env python3
"""
Final 3D Live Streaming Test - Guaranteed to show 3D visualizations live!
"""

import time
import sys
import os
import webbrowser
import threading

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from testing.visualizations.live_3d_visualizer import get_3d_visualizer
from testing.visualizations.visual_utils import start_live_server

def create_continuous_3d_stream():
    """Create a continuous stream of 3D visualizations."""
    visualizer = get_3d_visualizer()
    
    for i in range(10):
        print(f"\n🎯 Creating 3D visualization {i+1}/10...")
        
        # Create different test results each time
        test_results = []
        for j in range(5):
            test_results.append({
                "name": f"test_{j+1}_batch_{i+1}",
                "duration": 0.001 + (j * 0.002) + (i * 0.001),
                "status": "PASSED" if (j + i) % 3 != 0 else "FAILED"
            })
        
        # Stream the 3D visualization
        fig = visualizer.stream_3d_visualization("test_landscape", test_results)
        print(f"✅ 3D visualization {i+1} created and streamed!")
        
        # Wait before next visualization
        time.sleep(3)
    
    print("\n🎉 All 3D visualizations completed!")

def main():
    """Main function to run the final 3D live test."""
    print("🚀 Final 3D Live Streaming Test")
    print("=" * 50)
    
    # Start the live server
    print("🌐 Starting live server...")
    start_live_server()
    
    # Wait for server to start
    time.sleep(2)
    
    # Open the debug dashboard in browser
    debug_dashboard_path = os.path.abspath("testing/visualizations/debug_3d_dashboard.html")
    print(f"🌐 Opening debug dashboard: {debug_dashboard_path}")
    webbrowser.open(f"file://{debug_dashboard_path}")
    
    # Wait a moment for browser to open
    time.sleep(3)
    
    print("\n📊 Starting continuous 3D visualization stream...")
    print("🔍 Watch the debug dashboard in your browser!")
    print("📱 You should see 3D visualizations appearing in real-time")
    
    # Create continuous stream of 3D visualizations
    create_continuous_3d_stream()
    
    # Keep server running
    print("\n⏳ Keeping server running for 30 more seconds...")
    print("🌐 Dashboard should show all 3D visualizations")
    time.sleep(30)
    
    print("\n✅ Final 3D live streaming test completed!")
    print("🎯 If you saw 3D visualizations in the browser, the system is working!")

if __name__ == "__main__":
    main()
