#!/usr/bin/env python3
"""
Isolated 3D Test - Kill all servers and start fresh with single dashboard.
"""

import time
import sys
import os
import webbrowser
import subprocess

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def kill_existing_servers():
    """Kill any existing Python processes that might be running servers."""
    try:
        # Kill any Python processes that might be running servers
        subprocess.run(['pkill', '-f', 'working_live_server'], capture_output=True)
        subprocess.run(['pkill', '-f', 'live_server'], capture_output=True)
        time.sleep(1)
        print("🧹 Killed existing server processes")
    except Exception as e:
        print(f"⚠️ Could not kill processes: {e}")

def isolated_3d_test():
    """Isolated 3D visualization test."""
    print("🧪 Isolated 3D Test")
    print("=" * 40)
    
    # Kill existing servers
    kill_existing_servers()
    
    # Start fresh server
    print("🚀 Starting fresh server...")
    from testing.visualizations.working_live_server import WorkingLiveServer
    
    server = WorkingLiveServer()
    server.start()
    time.sleep(3)
    
    print(f"✅ Server running on port {server.port}")
    print(f"🔗 Clients connected: {len(server.clients)}")
    
    # Open ONLY the clean dashboard
    dashboard_path = os.path.abspath("testing/visualizations/clean_3d_dashboard.html")
    print(f"🌐 Opening clean dashboard: {dashboard_path}")
    webbrowser.open(f"file://{dashboard_path}")
    
    # Wait for browser to open
    time.sleep(3)
    
    print(f"🔗 Clients after dashboard: {len(server.clients)}")
    
    # Get visualizer
    from testing.visualizations.live_3d_visualizer import get_3d_visualizer
    visualizer = get_3d_visualizer()
    
    print("\n📊 Creating 3D visualizations...")
    
    # Create 3D visualizations
    for i in range(3):
        print(f"\n🎯 Creating 3D visualization {i+1}/3...")
        
        # Create test data
        test_data = [
            {"name": f"isolated_test_{i+1}", "duration": 0.001 + i * 0.002, "status": "PASSED"},
            {"name": f"isolated_test_{i+1}_b", "duration": 0.002 + i * 0.001, "status": "FAILED" if i % 2 == 0 else "PASSED"}
        ]
        
        # Stream 3D visualization
        fig = visualizer.stream_3d_visualization("test_landscape", test_data)
        print(f"✅ 3D visualization {i+1} created and streamed!")
        
        # Check client count
        print(f"🔗 Current clients: {len(server.clients)}")
        
        # Wait between visualizations
        time.sleep(3)
    
    print("\n🎉 All 3D visualizations completed!")
    print("🌐 Check the clean dashboard in your browser!")
    print("📱 You should see 3 interactive 3D visualizations")
    
    # Keep server running
    print("\n⏳ Keeping server running for 15 seconds...")
    time.sleep(15)
    
    # Stop server
    server.stop()
    print("🛑 Server stopped")
    print("✅ Isolated 3D test completed!")

if __name__ == "__main__":
    isolated_3d_test()
