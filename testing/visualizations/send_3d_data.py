#!/usr/bin/env python3
"""
Send 3D Data - Test script to send 3D visualizations to running server
"""

import asyncio
import websockets
import json
import time
import numpy as np
import plotly.graph_objects as go

async def send_3d_data():
    """Send 3D visualization data to the running debug server."""
    print("🎯 Sending 3D Data to Running Server")
    print("=" * 40)
    
    try:
        async with websockets.connect('ws://127.0.0.1:8005') as websocket:
            print("✅ Connected to debug server")
            
            # Wait for connection confirmation
            msg = await websocket.recv()
            print(f"📨 Server confirmed: {msg[:100]}...")
            
            # Create 3D visualization
            print("🎨 Creating 3D visualization...")
            
            # Simple 3D data
            x = [1, 2, 3, 4, 5]
            y = [1, 4, 9, 16, 25]
            z = [1, 8, 27, 64, 125]
            
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+lines',
                marker=dict(size=10, color='red', symbol='diamond'),
                line=dict(color='yellow', width=5)
            )])
            
            fig.update_layout(
                title='🎯 Test 3D Visualization',
                scene=dict(
                    xaxis_title='X Axis',
                    yaxis_title='Y Axis', 
                    zaxis_title='Z Axis'
                )
            )
            
            # Convert to dict
            fig_dict = fig.to_dict()
            
            # Create message
            message = {
                "series_id": "3d_test_landscape",
                "value": {
                    "plot_data": {
                        "figure": fig_dict,
                        "layout": fig_dict.get('layout', {}),
                        "data": fig_dict.get('data', [])
                    },
                    "type": "test_landscape",
                    "timestamp": time.time(),
                    "data_points": 5,
                    "interactive": True,
                    "dimensions": 3
                },
                "step": 0,
                "timestamp": time.time()
            }
            
            print(f"📨 Sending 3D data ({len(json.dumps(message))} characters)...")
            
            # Send the message
            await websocket.send(json.dumps(message))
            print("✅ 3D data sent!")
            
            # Wait for any response
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📨 Response received: {msg[:100]}...")
            except asyncio.TimeoutError:
                print("⏳ No response received (timeout)")
            
            print("🎉 3D data transmission complete!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(send_3d_data())
