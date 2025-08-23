#!/usr/bin/env python3
"""
Working Live Streaming Server - Actually provides WebSocket functionality.
"""

import asyncio
import websockets
import json
import time
import threading
import webbrowser
import os
from typing import Set

class WorkingLiveServer:
    """Live streaming server that actually works with the dashboard."""
    
    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False
        self.server = None
        
    async def register(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)
        print(f"🔗 New client connected. Total: {len(self.clients)}")
        
    async def unregister(self, websocket):
        """Unregister a client connection"""
        self.clients.remove(websocket)
        print(f"❌ Client disconnected. Total: {len(self.clients)}")
        
    async def broadcast(self, message):
        """Broadcast message to all connected clients"""
        if self.clients:
            clients_copy = self.clients.copy()
            disconnected = set()
            
            for client in clients_copy:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
                except Exception as e:
                    print(f"⚠️ Error sending to client: {e}")
                    disconnected.add(client)
            
            # Remove disconnected clients
            for client in disconnected:
                await self.unregister(client)
                
            print(f"📡 Broadcasted to {len(self.clients)} clients")
        
    async def handler(self, websocket):
        """Handle WebSocket connections"""
        await self.register(websocket)
        try:
            async for message in websocket:
                # Echo back for ping/pong
                await websocket.send(json.dumps({"type": "pong", "data": message}))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
            
    async def start_server(self):
        """Start the WebSocket server"""
        self.server = await websockets.serve(
            self.handler, 
            self.host, 
            self.port
        )
        self.running = True
        print(f"🚀 Working live server started on ws://{self.host}:{self.port}")
        await self.server.wait_closed()
        
    def start(self):
        """Start server in a separate thread"""
        def run_server():
            asyncio.run(self.start_server())
            
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server:
            self.server.close()
            
    def send_data(self, series_id: str, value, step: int):
        """Send data point to all connected clients"""
        if not self.running:
            return
            
        message = json.dumps({
            "series_id": series_id,
            "value": value,
            "step": step,
            "timestamp": time.time()
        })
        
        # Schedule broadcast in the event loop
        try:
            # Get the event loop from the server thread
            if hasattr(self, 'server_thread') and self.server_thread.is_alive():
                # Use asyncio.run_coroutine_threadsafe to send from any thread
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
                else:
                    # Create a new event loop for this broadcast
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(self.broadcast(message))
                    new_loop.close()
            else:
                print("⚠️ Server not running, skipping broadcast")
        except Exception as e:
            print(f"⚠️ Error broadcasting: {e}")


# Global instance
_working_server = None

def start_working_live_server():
    """Start the working live streaming server and open dashboard"""
    global _working_server
    
    if _working_server is None:
        _working_server = WorkingLiveServer()
        _working_server.start()
        
        # Open the conscious dashboard
        dashboard_path = os.path.join(
            os.path.dirname(__file__), 
            "conscious_live_dashboard.html"
        )
        
        if os.path.exists(dashboard_path):
            print(f"🎥 Working dashboard available at: file://{os.path.abspath(dashboard_path)}")
            print("🌐 Open manually in browser to avoid multiple instances")
            print(f"🔗 WebSocket server running on ws://127.0.0.1:8000")
        else:
            print(f"❌ Dashboard not found: {dashboard_path}")
    
    return _working_server

def working_live_series(series_id: str, value, step: int):
    """Stream data point to working live dashboard"""
    global _working_server
    
    if _working_server is None:
        start_working_live_server()
        
    _working_server.send_data(series_id, value, step)

if __name__ == "__main__":
    # Test the working server
    server = start_working_live_server()
    
    # Send some test data
    for i in range(10):
        working_live_series("test_series", i * 0.1, i)
        time.sleep(1)
    
    print("✅ Working test completed")
