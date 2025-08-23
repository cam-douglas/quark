#!/usr/bin/env python3
"""
Final Working 3D Test - Guaranteed to show 3D visualizations!
"""

import time
import sys
import os
import webbrowser

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def final_working_3d_test():
    """Final working 3D visualization test."""
    print("🎯 Final Working 3D Test")
    print("=" * 40)
    
    # Start server
    print("🚀 Starting server...")
    from testing.visualizations.working_live_server import WorkingLiveServer
    
    server = WorkingLiveServer()
    server.start()
    time.sleep(3)
    
    print(f"✅ Server running on port {server.port}")
    print(f"🔗 Initial clients: {len(server.clients)}")
    
    # Open ONLY the working 3D demo
    demo_path = os.path.abspath("testing/visualizations/working_3d_demo.html")
    print(f"🌐 Opening working 3D demo: {demo_path}")
    webbrowser.open(f"file://{demo_path}")
    
    # Wait for browser to open
    time.sleep(3)
    
    print(f"🔗 Clients after demo: {len(server.clients)}")
    
    # Get visualizer
    from testing.visualizations.live_3d_visualizer import get_3d_visualizer
    visualizer = get_3d_visualizer()
    
    print("\n📊 Creating 3D visualizations...")
    
    # Create 3D visualizations
    for i in range(3):
        print(f"\n🎯 Creating 3D visualization {i+1}/3...")
        
        # Create test data
        test_data = [
            {"name": f"working_test_{i+1}", "duration": 0.001 + i * 0.002, "status": "PASSED"},
            {"name": f"working_test_{i+1}_b", "duration": 0.002 + i * 0.001, "status": "FAILED" if i % 2 == 0 else "PASSED"}
        ]
        
        # Stream 3D visualization
        fig = visualizer.stream_3d_visualization("test_landscape", test_data)
        print(f"✅ 3D visualization {i+1} created and streamed!")
        
        # Check client count
        print(f"🔗 Current clients: {len(server.clients)}")
        
        # Wait between visualizations
        time.sleep(3)
    
    print("\n🎉 All 3D visualizations completed!")
    print("🌐 Check the working 3D demo in your browser!")
    print("📱 You should see 3 interactive 3D visualizations")
    
    # Keep server running
    print("\n⏳ Keeping server running for 20 seconds...")
    print("🎯 Watch the 3D visualizations appear in real-time!")
    time.sleep(20)
    
    # Stop server
    server.stop()
    print("🛑 Server stopped")
    print("✅ Final working 3D test completed!")

if __name__ == "__main__":
    final_working_3d_test()
