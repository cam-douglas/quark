#!/usr/bin/env python3
"""
Comprehensive 3D Live Streaming Demonstration for Quark.
Shows all 3D visualization types with live streaming to dashboard.
"""

import time
import sys
import os
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from testing.visualizations.live_3d_visualizer import get_3d_visualizer
from testing.visualizations.visual_utils import start_live_server, live_series

def create_comprehensive_3d_demo():
    """Create a comprehensive 3D live streaming demonstration."""
    print("🌟 Comprehensive 3D Live Streaming Demo")
    print("=" * 60)
    
    # Start live server
    print("🚀 Starting live server...")
    start_live_server()
    
    # Wait for server to start
    time.sleep(2)
    
    # Get 3D visualizer
    visualizer = get_3d_visualizer()
    
    print("\n🎯 Creating comprehensive 3D visualizations...")
    
    # 1. Test Results Landscape (Multiple tests)
    print("\n📊 1. Creating 3D Test Results Landscape...")
    test_results = []
    for i in range(8):
        test_results.append({
            "name": f"test_{i+1}",
            "duration": 0.001 + random.uniform(0, 0.02),
            "status": random.choice(["PASSED", "FAILED", "PASSED", "PASSED"])  # 75% pass rate
        })
    
    fig1 = visualizer.stream_3d_visualization("test_landscape", test_results)
    time.sleep(1)
    
    # 2. Performance Metrics Cube (Real-time simulation)
    print("📈 2. Creating 3D Performance Metrics Cube...")
    performance_data = []
    for i in range(25):
        performance_data.append({
            "timestamp": i,
            "cpu_usage": 20 + 15 * (i % 4) + random.uniform(-2, 2),
            "memory_usage": 45 + 8 * (i % 3) + random.uniform(-1, 1),
            "message_count": i * 3 + random.randint(0, 8)
        })
    
    fig2 = visualizer.stream_3d_visualization("performance_cube", performance_data)
    time.sleep(1)
    
    # 3. Experiment Parameter Space
    print("🧬 3. Creating 3D Experiment Parameter Space...")
    experiments = []
    for i in range(12):
        experiments.append({
            "name": f"experiment_{i+1}",
            "params": {
                "learning_rate": 0.0001 + (i * 0.0002),
                "batch_size": 16 + (i * 8),
                "epochs": 5 + (i % 10)
            },
            "duration": 0.001 + i * 0.003,
            "success": random.choice([True, True, True, False])  # 75% success rate
        })
    
    fig3 = visualizer.stream_3d_visualization("experiment_space", experiments)
    time.sleep(1)
    
    # 4. Neural Network Topology
    print("🧠 4. Creating 3D Neural Network Topology...")
    fig4 = visualizer.stream_3d_visualization("neural_topology", [784, 512, 256, 128, 64, 10])
    time.sleep(1)
    
    # 5. Live streaming of additional metrics
    print("\n📡 5. Streaming live metrics to dashboard...")
    for i in range(20):
        # Stream various metrics
        live_series("3d_demo_progress", {
            "step": i,
            "total_visualizations": 4,
            "current_step": i + 1,
            "status": "streaming"
        }, i)
        
        # Stream performance data
        live_series("3d_demo_performance", {
            "cpu": 20 + 10 * (i % 3),
            "memory": 45 + 5 * (i % 2),
            "active_connections": 1 + (i % 2)
        }, i)
        
        time.sleep(0.2)
    
    print("\n✅ Comprehensive 3D live streaming demonstration completed!")
    print("🌐 Check your browser dashboard for all 3D visualizations!")
    
    # Keep server running to show results
    print("⏳ Keeping server running for 15 seconds...")
    time.sleep(15)
    
    return [fig1, fig2, fig3, fig4]

def run_pytest_with_3d():
    """Run pytest with 3D visualizations enabled."""
    print("\n🧪 Running pytest with 3D visualizations...")
    
    import subprocess
    
    try:
        # Run a few tests to generate 3D visualizations
        result = subprocess.run([
            "python", "-m", "pytest", 
            "testing/testing_frameworks/tests/test_phase1_prototypes.py::test_thalamic_relay_gate",
            "-v"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Pytest with 3D visualizations completed successfully!")
        else:
            print(f"⚠️ Pytest completed with return code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Pytest timed out")
    except Exception as e:
        print(f"❌ Error running pytest: {e}")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive 3D Live Streaming Demo")
    print("=" * 60)
    
    # Run comprehensive demo
    figures = create_comprehensive_3d_demo()
    
    print(f"\n📊 Created {len(figures)} comprehensive 3D visualizations")
    
    # Run pytest with 3D visualizations
    run_pytest_with_3d()
    
    print("\n🎉 Comprehensive 3D live streaming demonstration complete!")
    print("🌟 All visualizations should now be visible in your live dashboard!")
