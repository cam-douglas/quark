#!/usr/bin/env python3
"""
Test script to verify 3D live streaming is working.
"""

import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from testing.visualizations.live_3d_visualizer import get_3d_visualizer, create_3d_test_demo
from testing.visualizations.visual_utils import start_live_server

def test_3d_live_streaming():
    """Test 3D live streaming functionality."""
    print("🧪 Testing 3D Live Streaming System")
    print("=" * 50)
    
    # Start live server
    print("🚀 Starting live server...")
    start_live_server()
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Get 3D visualizer
    visualizer = get_3d_visualizer()
    
    # Test individual 3D visualizations
    print("\n🎯 Testing individual 3D visualizations...")
    
    # Test 1: Test landscape
    test_results = [
        {"name": "test_1", "duration": 0.001, "status": "PASSED"},
        {"name": "test_2", "duration": 0.015, "status": "FAILED"},
        {"name": "test_3", "duration": 0.002, "status": "PASSED"}
    ]
    
    print("📊 Creating test landscape...")
    fig1 = visualizer.stream_3d_visualization("test_landscape", test_results)
    time.sleep(1)
    
    # Test 2: Performance cube
    performance_data = []
    for i in range(10):
        performance_data.append({
            "timestamp": i,
            "cpu_usage": 20 + 10 * (i % 3),
            "memory_usage": 45 + 5 * (i % 2),
            "message_count": i * 2
        })
    
    print("📈 Creating performance cube...")
    fig2 = visualizer.stream_3d_visualization("performance_cube", performance_data)
    time.sleep(1)
    
    # Test 3: Neural topology
    print("🧠 Creating neural topology...")
    fig3 = visualizer.stream_3d_visualization("neural_topology", [100, 50, 25, 10])
    time.sleep(1)
    
    print("\n✅ 3D live streaming tests completed!")
    print("🌐 Check your browser dashboard for live 3D visualizations!")
    
    # Keep server running to show results
    print("⏳ Keeping server running for 30 seconds...")
    print("🌐 Open your browser to: http://127.0.0.1:8000")
    print("📱 Check browser console (F12) for debug messages")
    time.sleep(30)
    
    return [fig1, fig2, fig3]

if __name__ == "__main__":
    figures = test_3d_live_streaming()
    print(f"\n📊 Created {len(figures)} 3D visualizations")
    print("🎉 3D live streaming test complete!")
