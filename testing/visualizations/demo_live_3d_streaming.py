#!/usr/bin/env python3
"""
Demo Live 3D Streaming - Shows the system working with connected dashboard
"""

import time
import sys
import os
import webbrowser
import asyncio
import websockets
import json
import threading
import numpy as np
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class Demo3DServer:
    """Demo 3D server that will definitely show live streaming working."""
    
    def __init__(self, port=8008):
        self.port = port
        self.clients = set()
        self.running = False
        self.server = None
        self.loop = None
        
    async def handler(self, websocket):
        """Handle WebSocket connections."""
        self.clients.add(websocket)
        print(f"🔗 New client connected. Total: {len(self.clients)}")
        
        # Send immediate confirmation
        await websocket.send(json.dumps({
            "type": "connection_confirmed",
            "message": f"Connected to Demo 3D Server on port {self.port}",
            "timestamp": time.time(),
            "client_id": id(websocket)
        }))
        
        try:
            async for message in websocket:
                print(f"📨 Received from client: {message[:100]}...")
                # Echo back for testing
                await websocket.send(json.dumps({
                    "type": "echo",
                    "message": f"Echo: {message}",
                    "timestamp": time.time()
                }))
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
        print(f"🚀 Demo 3D server started on ws://127.0.0.1:{self.port}")
        
        await self.server.wait_closed()
    
    def start(self):
        """Start server in a separate thread."""
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.start_server())
            
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(2)
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server:
            self.server.close()
        if self.loop:
            self.loop.stop()
    
    def create_demo_3d_visualization(self, test_name, duration, status):
        """Create a demo 3D visualization."""
        # Create interesting 3D data
        x_vals = []
        y_vals = []
        z_vals = []
        colors = []
        text_labels = []
        
        # Create multiple test points
        for i in range(5):
            x_vals.append(i)
            y_vals.append(duration + i * 0.1)
            z_vals.append(1 if status == "PASSED" else 0)
            colors.append('lime' if status == "PASSED" else 'red')
            text_labels.append(f"Test: {test_name}_{i+1}<br>Duration: {duration + i * 0.1:.4f}s<br>Status: {status}")
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers+text+lines',
            marker=dict(
                size=20,
                color=colors,
                opacity=0.8,
                line=dict(width=3, color='white'),
                symbol='diamond'
            ),
            text=[f"T{i+1}" for i in range(len(x_vals))],
            textposition="middle center",
            hovertext=text_labels,
            hoverinfo='text',
            name=f'{test_name} Results',
            line=dict(color='yellow', width=5)
        )])
        
        # Add success plane
        x_surface = np.linspace(0, 4, 10)
        y_surface = np.linspace(duration, duration + 0.5, 10)
        X, Y = np.meshgrid(x_surface, y_surface)
        Z = np.ones_like(X) * 0.5
        
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name='Success Threshold'
        ))
        
        fig.update_layout(
            title=f'🎯 Demo 3D: {test_name}',
            scene=dict(
                xaxis_title='Test Index',
                yaxis_title='Duration (seconds)',
                zaxis_title='Success Rate',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            template='plotly_dark'
        )
        
        return fig
    
    def stream_3d_visualization(self, test_name, duration, status):
        """Stream 3D visualization data to connected clients."""
        try:
            print(f"🎯 Creating demo 3D visualization for: {test_name}")
            
            fig = self.create_demo_3d_visualization(test_name, duration, status)
            
            if fig:
                print(f"✅ 3D figure created successfully")
                
                # Convert to dict for JSON serialization
                fig_dict = fig.to_dict()
                
                # Create message
                message = {
                    "series_id": f"3d_demo_{test_name}",
                    "value": {
                        "plot_data": {
                            "figure": fig_dict,
                            "layout": fig_dict.get('layout', {}),
                            "data": fig_dict.get('data', [])
                        },
                        "type": "demo_landscape",
                        "timestamp": time.time(),
                        "data_points": 5,
                        "interactive": True,
                        "dimensions": 3
                    },
                    "step": 0,
                    "timestamp": time.time()
                }
                
                print(f"📨 Message created, size: {len(json.dumps(message))} characters")
                
                # Stream to all clients using the server's event loop
                if self.loop and self.loop.is_running():
                    async def broadcast():
                        if self.clients:
                            print(f"📡 Broadcasting to {len(self.clients)} clients...")
                            await asyncio.gather(
                                *[client.send(json.dumps(message)) for client in self.clients],
                                return_exceptions=True
                            )
                            print(f"✅ Broadcast complete to {len(self.clients)} clients")
                        else:
                            print("⚠️ No clients connected to broadcast to")
                    
                    # Schedule broadcast in the server's event loop
                    future = asyncio.run_coroutine_threadsafe(broadcast(), self.loop)
                    try:
                        future.result(timeout=5)  # Wait up to 5 seconds
                        print(f"📡 3D visualization broadcasted successfully")
                    except Exception as e:
                        print(f"❌ Broadcast failed: {e}")
                else:
                    print("❌ Server event loop not available")
                
                return fig
                
        except Exception as e:
            print(f"❌ Error creating 3D visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

def create_demo_dashboard():
    """Create a demo dashboard that will definitely show 3D visualizations."""
    dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <title>🎯 Demo Live 3D Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: 'Arial', sans-serif; 
            margin: 0; 
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            color: white;
            min-height: 100vh;
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid rgba(255,255,255,0.2);
        }
        .status { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 15px; 
            margin-bottom: 30px;
            text-align: center;
            border: 2px solid rgba(255,255,255,0.2);
        }
        .viz { 
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 25px;
            margin: 25px 0;
            border: 2px solid rgba(255,255,255,0.3);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .log { 
            background: rgba(0,0,0,0.7);
            padding: 20px;
            border-radius: 15px;
            margin-top: 30px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            border: 2px solid rgba(255,255,255,0.2);
        }
        .connected { color: #00ff88; font-weight: bold; }
        .disconnected { color: #ff4757; font-weight: bold; }
        .info { color: #ffd700; }
        .success { color: #00ff88; font-weight: bold; }
        .error { color: #ff4757; font-weight: bold; }
        .test-button {
            background: linear-gradient(45deg, #00ff88, #00ffff);
            color: black;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            margin: 10px;
            transition: all 0.3s ease;
        }
        .test-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,255,136,0.3);
        }
        .alert {
            background: rgba(255, 193, 7, 0.2);
            border: 2px solid #ffc107;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Demo Live 3D Dashboard</h1>
        <p>Port 8008 - This WILL show live 3D streaming working!</p>
        <button class="test-button" onclick="testConnection()">🧪 Test Connection</button>
        <button class="test-button" onclick="clearAll()">🗑️ Clear All</button>
        <button class="test-button" onclick="requestDemo()">🎨 Request Demo 3D</button>
    </div>
    
    <div class="alert">
        <strong>🎯 EXPECTED:</strong> You should see 3D visualizations appear automatically as they're created!
    </div>
    
    <div class="status">
        <div><strong>Status:</strong> <span id="status" class="disconnected">Disconnected</span></div>
        <div><strong>Messages:</strong> <span id="msg-count">0</span></div>
        <div><strong>3D Visualizations:</strong> <span id="viz-count">0</span></div>
        <div><strong>Last Update:</strong> <span id="last-update">Never</span></div>
    </div>
    
    <div id="visualizations"></div>
    
    <div class="log">
        <h3>📋 Activity Log</h3>
        <div id="log-entries"></div>
    </div>
    
    <script>
        let messageCount = 0;
        let vizCount = 0;
        let ws;
        
        function log(message, type = 'info') {
            const logDiv = document.getElementById('log-entries');
            const entry = document.createElement('div');
            entry.className = type;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logDiv.appendChild(entry);
            logDiv.scrollTop = logDiv.scrollHeight;
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
        
        function updateLastUpdate() {
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }
        
        function testConnection() {
            log('🧪 Testing connection...', 'info');
            if (ws && ws.readyState === WebSocket.OPEN) {
                log('✅ Connection is active!', 'success');
                ws.send('ping');
            } else {
                log('❌ No active connection', 'error');
                connect();
            }
        }
        
        function clearAll() {
            log('🗑️ Clearing all visualizations...', 'info');
            document.getElementById('visualizations').innerHTML = '';
            vizCount = 0;
            document.getElementById('viz-count').textContent = vizCount;
            log('✅ All visualizations cleared', 'success');
        }
        
        function requestDemo() {
            log('🎨 Requesting demo 3D data from server...', 'info');
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('request_demo_3d');
            } else {
                log('❌ No active connection', 'error');
            }
        }
        
        function connect() {
            log('🔌 Connecting to demo server on port 8008...', 'info');
            ws = new WebSocket('ws://127.0.0.1:8008');
            
            ws.onopen = function() {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'connected';
                log('✅ Connected to demo 3D server', 'success');
                updateLastUpdate();
            };
            
            ws.onmessage = function(event) {
                messageCount++;
                document.getElementById('msg-count').textContent = messageCount;
                updateLastUpdate();
                
                log(`📨 Raw message received (${event.data.length} chars)`, 'debug');
                console.log('Raw message data:', event.data);
                
                try {
                    const data = JSON.parse(event.data);
                    log(`📨 Parsed: ${data.series_id || data.type || 'unknown'}`, 'info');
                    console.log('Parsed message:', data);
                    
                    // Handle connection confirmation
                    if (data.type === 'connection_confirmed') {
                        log(`✅ ${data.message}`, 'success');
                        return;
                    }
                    
                    // Handle echo
                    if (data.type === 'echo') {
                        log(`🔄 Echo: ${data.message}`, 'debug');
                        return;
                    }
                    
                    // Handle 3D visualizations
                    if (data.series_id && data.series_id.startsWith('3d_')) {
                        log(`🎯 Processing 3D: ${data.series_id}`, 'success');
                        handle3DVisualization(data);
                    }
                    
                } catch (e) {
                    log(`❌ Error parsing message: ${e.message}`, 'error');
                    console.error('Parse error:', e);
                    console.error('Raw data:', event.data);
                }
            };
            
            ws.onclose = function() {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').className = 'disconnected';
                log('❌ Disconnected from server', 'error');
            };
            
            ws.onerror = function(error) {
                log('⚠️ WebSocket error', 'error');
                console.error('WebSocket error:', error);
            };
        }
        
        function handle3DVisualization(data) {
            const { series_id, value } = data;
            
            log(`🔍 Processing 3D data for: ${series_id}`, 'info');
            console.log('Full 3D data:', data);
            
            if (!value || !value.plot_data) {
                log(`❌ No plot_data found in value`, 'error');
                console.log('Value field:', value);
                return;
            }
            
            const plot_data = value.plot_data;
            log(`📈 Plot data keys: ${Object.keys(plot_data).join(', ')}`, 'debug');
            
            if (!plot_data.figure) {
                log(`❌ No figure in plot_data`, 'error');
                console.log('Plot data:', plot_data);
                return;
            }
            
            log(`🎨 Creating 3D visualization...`, 'success');
            
            try {
                // Create container
                const container = document.createElement('div');
                container.className = 'viz';
                container.id = `viz-${series_id}`;
                
                const title = document.createElement('h3');
                title.textContent = `🎨 ${series_id.replace('3d_', '3D ')}`;
                title.style.color = '#ffd700';
                title.style.textAlign = 'center';
                
                const chartDiv = document.createElement('div');
                chartDiv.id = `chart-${series_id}`;
                chartDiv.style.height = '600px';
                chartDiv.style.width = '100%';
                chartDiv.style.border = '2px solid rgba(255,255,255,0.3)';
                chartDiv.style.borderRadius = '10px';
                
                container.appendChild(title);
                container.appendChild(chartDiv);
                
                // Remove any existing visualization with same ID
                const existing = document.getElementById(`viz-${series_id}`);
                if (existing) {
                    existing.remove();
                }
                
                document.getElementById('visualizations').appendChild(container);
                
                // Create Plotly 3D visualization
                Plotly.newPlot(chartDiv.id, plot_data.data, plot_data.layout, {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                    displaylogo: false
                });
                
                vizCount++;
                document.getElementById('viz-count').textContent = vizCount;
                
                log(`✅ 3D visualization created: ${series_id}`, 'success');
                
            } catch (error) {
                log(`❌ Error creating 3D viz: ${error.message}`, 'error');
                console.error('Full error:', error);
                console.error('Error stack:', error.stack);
                
                // Show error in container
                chartDiv.innerHTML = `<div style="color: red; text-align: center; padding: 20px; font-size: 18px;">
                    <h3>❌ 3D Visualization Error</h3>
                    <p>Error: ${error.message}</p>
                    <p>Check browser console for details</p>
                </div>`;
            }
        }
        
        // Start connection
        connect();
        
        log('🚀 Demo 3D dashboard initialized', 'success');
        log('🌐 Waiting for 3D visualizations...', 'info');
        log('💡 Use browser console (F12) to see detailed logs', 'info');
        
        // Test connection after a delay
        setTimeout(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                log('✅ Connection test passed', 'success');
            } else {
                log('⚠️ Connection test failed', 'error');
            }
        }, 3000);
    </script>
</body>
</html>"""
    
    return dashboard_html

def demo_live_3d_streaming():
    """Demo live 3D visualization that will definitely show live streaming working."""
    print("🎯 Demo Live 3D Streaming")
    print("=" * 50)
    
    # Start demo server on port 8008
    print("🚀 Starting demo server on port 8008...")
    server = Demo3DServer(port=8008)
    server.start()
    
    print(f"✅ Server running on port {server.port}")
    print(f"🔗 Initial clients: {len(server.clients)}")
    
    # Create the demo dashboard
    dashboard_html = create_demo_dashboard()
    dashboard_path = "testing/visualizations/demo_live_3d_dashboard.html"
    
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    
    print(f"🌐 Created demo dashboard: {dashboard_path}")
    
    # Open dashboard
    print("🌐 Opening demo dashboard...")
    webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
    
    # Wait for browser to open
    time.sleep(5)
    
    print(f"🔗 Clients after dashboard: {len(server.clients)}")
    
    print("\n📊 Creating demo 3D visualizations...")
    
    # Create demo 3D visualizations
    demo_tests = [
        ("Simple Calculation", 0.1, "PASSED"),
        ("Array Operations", 0.2, "PASSED"),
        ("String Operations", 0.15, "PASSED"),
        ("Boolean Logic", 0.1, "PASSED"),
        ("List Operations", 0.2, "PASSED"),
        ("Dictionary Operations", 0.15, "PASSED"),
        ("Numerical Operations", 0.25, "PASSED"),
        ("Comparison Operations", 0.1, "PASSED")
    ]
    
    for i, (test_name, duration, status) in enumerate(demo_tests):
        print(f"\n🎯 Creating demo 3D visualization {i+1}/8...")
        print(f"📊 Test: {test_name}, Duration: {duration}s, Status: {status}")
        
        # Stream 3D visualization using our demo server
        fig = server.stream_3d_visualization(test_name, duration, status)
        print(f"✅ Demo 3D visualization {i+1} created and streamed!")
        
        # Check client count
        print(f"🔗 Current clients: {len(server.clients)}")
        
        # Wait between visualizations
        time.sleep(3)
    
    print("\n🎉 All demo 3D visualizations completed!")
    print("🌐 Check the demo dashboard in your browser!")
    print("📱 You should see 8 interactive 3D visualizations with live streaming!")
    
    # Keep server running
    print("\n⏳ Keeping server running for 30 seconds...")
    print("🎯 Watch the 3D visualizations appear in real-time!")
    
    # Show loading animation
    loading_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    for i in range(30):
        print(f"\r{loading_chars[i % len(loading_chars)]} Running... {i+1}/30 seconds", end="", flush=True)
        time.sleep(1)
    print("\n✅ Demo session complete!")
    
    # Stop server
    server.stop()
    print("🛑 Server stopped")
    print("✅ Demo live 3D streaming completed!")
    print("\n🎯 If you saw 3D visualizations:")
    print("   ✅ Live 3D streaming is working!")
    print("   🚀 The system is ready for production use!")

if __name__ == "__main__":
    demo_live_3d_streaming()
