#!/usr/bin/env python3
"""
Simple Live Streaming Server for Quark Experiments
Uses asyncio and websockets for reliable real-time data streaming.
"""

import asyncio
import websockets
import json
import time
import threading
from typing import Set
import webbrowser
import os
import queue

class LiveStreamServer:
    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False
        self.server = None
        self.message_queue = queue.Queue()
        
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
            # Create a copy of the set to avoid modification during iteration
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
            
    async def process_message_queue(self):
        """Process messages from the queue"""
        while self.running:
            try:
                # Check for messages with a timeout
                message = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.message_queue.get(timeout=0.1)
                )
                await self.broadcast(message)
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Error processing message queue: {e}")

    async def start_server(self):
        """Start the WebSocket server"""
        self.server = await websockets.serve(
            self.handler, 
            self.host, 
            self.port
        )
        self.running = True
        print(f"🚀 Live stream server started on ws://{self.host}:{self.port}")
        
        # Start message queue processor
        queue_task = asyncio.create_task(self.process_message_queue())
        
        try:
            await self.server.wait_closed()
        finally:
            queue_task.cancel()
        
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
        # Clear the message queue
        try:
            while not self.message_queue.empty():
                self.message_queue.get_nowait()
        except:
            pass
            
    def send_data(self, series_id: str, value: float, step: int):
        """Send data point to all connected clients"""
        if not self.running:
            return
            
        message = json.dumps({
            "series_id": series_id,
            "value": value,
            "step": step,
            "timestamp": time.time()
        })
        
        # Simply add to queue - non-blocking
        try:
            self.message_queue.put_nowait(message)
        except queue.Full:
            print("⚠️ Message queue full, dropping message")
        except Exception as e:
            print(f"⚠️ Error queuing message: {e}")

# Global server instance
_live_server = None

def start_live_server():
    """Start the live streaming server and open dashboard"""
    global _live_server
    
    if _live_server is None:
        _live_server = LiveStreamServer()
        _live_server.start()
        
        # Open the conscious dashboard
        dashboard_path = os.path.join(
            os.path.dirname(__file__), 
            "conscious_live_dashboard.html"
        )
        
        if os.path.exists(dashboard_path):
            try:
                webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
                print("🎥 Live dashboard opened - server running on ws://127.0.0.1:8000")
            except Exception as e:
                print(f"⚠️ Could not open dashboard: {e}")
                print(f"📊 Manual access: file://{os.path.abspath(dashboard_path)}")
        else:
            print(f"❌ Dashboard not found: {dashboard_path}")
    
    return _live_server

def live_series(series_id: str, value: float, step: int):
    """Stream data point to live dashboard"""
    global _live_server
    
    if _live_server is None:
        start_live_server()
        
    _live_server.send_data(series_id, value, step)

if __name__ == "__main__":
    # Test the server
    server = start_live_server()
    
    # Send some test data
    for i in range(10):
        live_series("test_series", i * 0.1, i)
        time.sleep(1)
    
    print("✅ Test completed")
