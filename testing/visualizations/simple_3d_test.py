#!/usr/bin/env python3
"""
Simple 3D Test - Debug 3D visualization issues.
"""

import time
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from testing.visualizations.live_3d_visualizer import get_3d_visualizer
from testing.visualizations.visual_utils import start_live_server

def simple_3d_test():
    """Simple 3D visualization test."""
    print("🧪 Simple 3D Test")
    print("=" * 30)
    
    # Start server
    print("🚀 Starting server...")
    start_live_server()
    time.sleep(2)
    
    # Get visualizer
    visualizer = get_3d_visualizer()
    
    # Create simple test data
    test_data = [
        {"name": "simple_test", "duration": 0.001, "status": "PASSED"}
    ]
    
    print("📊 Creating simple 3D visualization...")
    
    try:
        # Create and stream 3D visualization
        fig = visualizer.stream_3d_visualization("test_landscape", test_data)
        print("✅ 3D visualization created successfully!")
        
        # Check if figure was created
        if fig:
            print(f"📈 Figure type: {type(fig)}")
            print(f"📊 Figure data: {len(fig.data) if hasattr(fig, 'data') else 'No data'}")
        else:
            print("❌ No figure returned!")
            
    except Exception as e:
        print(f"❌ Error creating 3D visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # Keep server running
    print("\n⏳ Keeping server running for 10 seconds...")
    print("🌐 Check browser dashboard at: http://127.0.0.1:8000")
    time.sleep(10)
    
    print("✅ Simple 3D test completed!")

if __name__ == "__main__":
    simple_3d_test()
