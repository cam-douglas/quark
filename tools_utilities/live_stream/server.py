#!/usr/bin/env python3
"""
Live Stream Server - Real-time visualization broadcasting
Starts automatically when live_series() is called, broadcasts data to connected clients.
"""

import asyncio
import threading
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
import time
from typing import List

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🔗 New WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"❌ WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        if self.active_connections:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except:
                    # Remove dead connections
                    self.active_connections.remove(connection)
            print(f"📡 Broadcasted to {len(self.active_connections)} connections")

manager = ConnectionManager()

@app.get("/")
async def get():
    return HTMLResponse("""
    <html>
        <head>
            <title>Quark Live Stream Server</title>
        </head>
        <body>
            <h1>🧠 Quark Live Stream Server</h1>
            <p>Status: <span style="color: green;">Running</span></p>
            <p>Active connections: <span id="connections">0</span></p>
            <script>
                setInterval(() => {
                    fetch('/status').then(r => r.json()).then(data => {
                        document.getElementById('connections').textContent = data.connections;
                    });
                }, 1000);
            </script>
        </body>
    </html>
    """)

@app.get("/status")
async def status():
    return {"connections": len(manager.active_connections), "status": "running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def broadcast_data(series_id: str, value: float, step: int):
    """Send data to all connected WebSocket clients"""
    message = json.dumps({
        "series_id": series_id,
        "value": value,
        "step": step,
        "timestamp": time.time()
    })
    
    # Run broadcast in event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(manager.broadcast(message))
        else:
            # If no event loop, create one
            asyncio.run(manager.broadcast(message))
    except RuntimeError:
        # No event loop in this thread, create one
        asyncio.run(manager.broadcast(message))

def start_server(host="127.0.0.1", port=8000):
    """Start the FastAPI server in a daemon thread"""
    def run_server():
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            access_log=False
        )
        server = uvicorn.Server(config)
        server.run()
    
    # Create daemon thread so it doesn't block main program
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Give server time to start
    time.sleep(2)
    
    print(f"🎥 Live stream server started on {host}:{port}")
    print(f"📊 Dashboard available at http://{host}:{port}")
    print(f"🔌 WebSocket endpoint: ws://{host}:{port}/ws")
    
    return server_thread

if __name__ == "__main__":
    start_server()
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n�� Server stopped")
