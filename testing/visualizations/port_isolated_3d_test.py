#!/usr/bin/env python3
"""
Port Isolated 3D Test - Uses port 8001 to avoid conflicts.
"""

import time
import sys
import os
import webbrowser
import asyncio
import websockets
import json
import threading

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class Isolated3DServer:
    """Isolated 3D server on port 8001."""
    
    def __init__(self, port=8001):
        self.port = port
        self.clients = set()
        self.running = False
        self.server = None
        
    async def handler(self, websocket, path):
        """Handle WebSocket connections."""
        self.clients.add(websocket)
        print(f"🔗 New client connected. Total: {len(self.clients)}")
        
        try:
            async for message in websocket:
                # Echo back for testing
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"❌ Client disconnected. Total: {len(self.clients)}")
    
    async def start_server(self):
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.handler,
            "127.0.0.1",
            self.port
        )
        print(f"🚀 Isolated 3D server started on ws://127.0.0.1:{self.port}")
        
        await self.server.wait_closed()
    
    def start(self):
        """Start server in a separate thread."""
        def run_server():
            asyncio.run(self.start_server())
            
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(2)
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server:
            self.server.close()
    
    def send_data(self, series_id, value, step):
        """Send data to all connected clients."""
        if not self.clients:
            return
            
        message = json.dumps({
            "series_id": series_id,
            "value": value,
            "step": step,
            "timestamp": time.time()
        })
        
        # Broadcast to all clients
        async def broadcast():
            if self.clients:
                await asyncio.gather(
                    *[client.send(message) for client in self.clients],
                    return_exceptions=True
                )
        
        # Schedule broadcast
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast(), loop)
            else:
                asyncio.run(broadcast())
        except Exception as e:
            print(f"⚠️ Broadcast error: {e}")

def port_isolated_3d_test():
    """Port isolated 3D visualization test."""
    print("🧪 Port Isolated 3D Test")
    print("=" * 40)
    
    # Start isolated server on port 8001
    print("🚀 Starting isolated server on port 8001...")
    server = Isolated3DServer(port=8001)
    server.start()
    
    print(f"✅ Server running on port {server.port}")
    print(f"🔗 Initial clients: {len(server.clients)}")
    
    # Open ONLY the working 3D demo on port 8001
    demo_path = os.path.abspath("testing/visualizations/working_3d_demo.html")
    print(f"🌐 Opening working 3D demo: {demo_path}")
    
    # Create a modified version of the demo that connects to port 8001
    with open(demo_path, 'r') as f:
        demo_content = f.read()
    
    # Replace port 8000 with 8001
    demo_content = demo_content.replace('ws://127.0.0.1:8000', 'ws://127.0.0.1:8001')
    
    # Save modified demo
    modified_demo_path = "testing/visualizations/working_3d_demo_8001.html"
    with open(modified_demo_path, 'w') as f:
        f.write(demo_content)
    
    # Open modified demo
    webbrowser.open(f"file://{os.path.abspath(modified_demo_path)}")
    
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
    print("🌐 Check the working 3D demo in your browser!")
    print("📱 You should see 3 interactive 3D visualizations")
    
    # Keep server running
    print("\n⏳ Keeping server running for 20 seconds...")
    print("🎯 Watch the 3D visualizations appear in real-time!")
    time.sleep(20)
    
    # Stop server
    server.stop()
    print("🛑 Server stopped")
    print("✅ Port isolated 3D test completed!")

if __name__ == "__main__":
    port_isolated_3d_test()
