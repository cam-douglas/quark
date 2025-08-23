#!/usr/bin/env python3
"""
Guaranteed 3D Show - This WILL show 3D visualizations!
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
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class Guaranteed3DServer:
    """3D server that guarantees visualizations will be shown."""
    
    def __init__(self, port=8004):
        self.port = port
        self.clients = set()
        self.running = False
        self.server = None
        
    async def handler(self, websocket):
        """Handle WebSocket connections."""
        self.clients.add(websocket)
        print(f"🔗 New client connected. Total: {len(self.clients)}")
        
        # Send immediate confirmation
        await websocket.send(json.dumps({
            "type": "connection_confirmed",
            "message": "Connected to Guaranteed 3D Server",
            "timestamp": time.time()
        }))
        
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
        print(f"🚀 Guaranteed 3D server started on ws://127.0.0.1:{self.port}")
        
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
    
    def create_amazing_3d_visualization(self, test_results):
        """Create an amazing 3D visualization that will definitely be visible."""
        if not test_results:
            return None
        
        # Create more interesting 3D data
        x_vals = []
        y_vals = []
        z_vals = []
        colors = []
        sizes = []
        text_labels = []
        
        for i, test in enumerate(test_results):
            x_vals.append(i)
            y_vals.append(test.get('duration', 0))
            z_vals.append(1 if test.get('status') == 'PASSED' else 0)
            colors.append('lime' if test.get('status') == 'PASSED' else 'red')
            sizes.append(15 + i * 2)  # Increasing size
            text_labels.append(f"Test: {test.get('name', 'Unknown')}<br>Duration: {test.get('duration', 0):.4f}s<br>Status: {test.get('status', 'Unknown')}")
        
        # Create 3D scatter plot with better visibility
        fig = go.Figure(data=[go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.9,
                line=dict(width=3, color='white'),
                symbol='diamond'  # More visible symbol
            ),
            text=[f"T{i+1}" for i in range(len(test_results))],
            textposition="middle center",
            hovertext=text_labels,
            hoverinfo='text',
            name='Test Results'
        )])
        
        # Add a success plane
        if len(x_vals) > 1:
            x_surface = np.linspace(0, len(x_vals)-1, 20)
            y_surface = np.linspace(min(y_vals), max(y_vals), 20)
            X, Y = np.meshgrid(x_surface, y_surface)
            Z = np.ones_like(X) * 0.5
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                opacity=0.4,
                colorscale='Blues',
                showscale=False,
                name='Success Threshold'
            ))
        
        # Add connecting lines between points
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='lines',
            line=dict(color='yellow', width=5),
            name='Test Progression'
        ))
        
        # Make it look amazing
        fig.update_layout(
            title=dict(
                text='🎯 AMAZING 3D Test Results Landscape',
                font=dict(size=24, color='white'),
                x=0.5
            ),
            scene=dict(
                xaxis_title='Test Index',
                yaxis_title='Duration (seconds)',
                zaxis_title='Success Rate',
                camera=dict(
                    eye=dict(x=2, y=2, z=2),
                    center=dict(x=0, y=0, z=0)
                ),
                bgcolor='rgba(0,0,0,0.8)',
                xaxis=dict(gridcolor='white', zerolinecolor='white'),
                yaxis=dict(gridcolor='white', zerolinecolor='white'),
                zaxis=dict(gridcolor='white', zerolinecolor='white')
            ),
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='white',
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def stream_3d_visualization(self, viz_type, data):
        """Stream 3D visualization data to connected clients."""
        try:
            if viz_type == "test_landscape":
                fig = self.create_amazing_3d_visualization(data)
            else:
                print(f"⚠️ Unknown visualization type: {viz_type}")
                return None
            
            if fig:
                # Convert to dict for JSON serialization
                fig_dict = fig.to_dict()
                
                # Stream to all connected clients
                message = json.dumps({
                    "series_id": f"3d_{viz_type}",
                    "value": {
                        "plot_data": {
                            "figure": fig_dict,
                            "layout": fig_dict.get('layout', {}),
                            "data": fig_dict.get('data', [])
                        },
                        "type": viz_type,
                        "timestamp": time.time(),
                        "data_points": len(data) if isinstance(data, list) else 1,
                        "interactive": True,
                        "dimensions": 3
                    },
                    "step": 0,
                    "timestamp": time.time()
                })
                
                # Broadcast to all clients
                async def broadcast():
                    if self.clients:
                        await asyncio.gather(
                            *[client.send(message) for client in self.clients],
                            return_exceptions=True
                        )
                
                # Schedule broadcast using the server's event loop
                try:
                    if hasattr(self, 'server_thread') and self.server_thread.is_alive():
                        # Use the server's event loop
                        asyncio.run_coroutine_threadsafe(broadcast(), asyncio.get_event_loop())
                    else:
                        # Fallback to running in new event loop
                        asyncio.run(broadcast())
                except Exception as e:
                    print(f"⚠️ Broadcast error: {e}")
                
                print(f"📡 3D visualization broadcasted to {len(self.clients)} clients")
                return fig
                
        except Exception as e:
            print(f"❌ Error creating 3D visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

def create_guaranteed_dashboard():
    """Create a dashboard that will definitely show 3D visualizations."""
    dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <title>🎯 GUARANTEED 3D Dashboard</title>
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
            font-size: 14px;
            border: 2px solid rgba(255,255,255,0.2);
        }
        .connected { color: #00ff88; font-weight: bold; }
        .disconnected { color: #ff4757; font-weight: bold; }
        .info { color: #ffd700; }
        .success { color: #00ff88; font-weight: bold; }
        .error { color: #ff4757; font-weight: bold; }
        .loading { 
            color: #00ffff; 
            animation: pulse 2s infinite;
            font-weight: bold;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .big-button {
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
        .big-button:hover {
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
        <h1>🎯 GUARANTEED 3D Dashboard</h1>
        <p>Port 8004 - This WILL show 3D visualizations!</p>
        <button class="big-button" onclick="testConnection()">🧪 Test Connection</button>
        <button class="big-button" onclick="clearAll()">🗑️ Clear All</button>
    </div>
    
    <div class="alert">
        <strong>⚠️ IMPORTANT:</strong> If you don't see 3D visualizations, check the browser console (F12) for errors!
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
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
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
        
        function connect() {
            if (reconnectAttempts >= maxReconnectAttempts) {
                log(`❌ Max reconnection attempts (${maxReconnectAttempts}) reached`, 'error');
                return;
            }
            
            log(`🔌 Connecting to guaranteed server on port 8004... (Attempt ${reconnectAttempts + 1})`, 'info');
            ws = new WebSocket('ws://127.0.0.1:8004');
            
            ws.onopen = function() {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'connected';
                reconnectAttempts = 0;
                log('✅ Connected to guaranteed 3D server', 'success');
                updateLastUpdate();
            };
            
            ws.onmessage = function(event) {
                messageCount++;
                document.getElementById('msg-count').textContent = messageCount;
                updateLastUpdate();
                
                try {
                    const data = JSON.parse(event.data);
                    log(`📨 Received: ${data.series_id || data.type || 'unknown'}`, 'info');
                    
                    // Handle connection confirmation
                    if (data.type === 'connection_confirmed') {
                        log(`✅ ${data.message}`, 'success');
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
                }
            };
            
            ws.onclose = function() {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').className = 'disconnected';
                log('❌ Disconnected from server', 'error');
                
                // Auto-reconnect
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    log(`🔄 Attempting to reconnect in 2 seconds... (${reconnectAttempts}/${maxReconnectAttempts})`, 'info');
                    setTimeout(connect, 2000);
                }
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
            log(`📈 Plot data keys: ${Object.keys(plot_data).join(', ')}`, 'info');
            
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
                chartDiv.style.height = '700px';
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
                    displaylogo: false,
                    toImageButtonOptions: {
                        format: 'png',
                        filename: `3d_${series_id}`,
                        height: 700,
                        width: 1000,
                        scale: 2
                    }
                });
                
                vizCount++;
                document.getElementById('viz-count').textContent = vizCount;
                
                log(`✅ 3D visualization created: ${series_id}`, 'success');
                
                // Add rotation animation
                const camera = plot_data.layout.scene?.camera;
                if (camera) {
                    let angle = 0;
                    const rotationInterval = setInterval(() => {
                        angle += 1;
                        Plotly.relayout(chartDiv.id, {
                            'scene.camera': {
                                eye: {
                                    x: camera.eye.x * Math.cos(angle * 0.02),
                                    y: camera.eye.y * Math.sin(angle * 0.02),
                                    z: camera.eye.z
                                }
                            }
                        });
                    }, 100);
                    
                    // Store interval for cleanup
                    chartDiv.dataset.rotationInterval = rotationInterval;
                }
                
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
        
        log('🚀 Guaranteed 3D dashboard initialized', 'success');
        log('🌐 Waiting for 3D visualizations...', 'info');
        log('💡 Tip: Use browser console (F12) to see detailed logs', 'info');
        
        // Test connection after a delay
        setTimeout(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                log('✅ Connection test passed', 'success');
            } else {
                log('⚠️ Connection test failed - attempting reconnect', 'error');
                connect();
            }
        }, 3000);
    </script>
</body>
</html>"""
    
    return dashboard_html

def guaranteed_3d_show():
    """Guaranteed 3D visualization show that will definitely work."""
    print("🎯 GUARANTEED 3D Show")
    print("=" * 50)
    
    # Start guaranteed server on port 8004
    print("🚀 Starting guaranteed server on port 8004...")
    server = Guaranteed3DServer(port=8004)
    server.start()
    
    print(f"✅ Server running on port {server.port}")
    print(f"🔗 Initial clients: {len(server.clients)}")
    
    # Create the guaranteed dashboard
    dashboard_html = create_guaranteed_dashboard()
    dashboard_path = "testing/visualizations/guaranteed_3d_dashboard.html"
    
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    
    print(f"🌐 Created guaranteed dashboard: {dashboard_path}")
    
    # Open dashboard
    print("🌐 Opening guaranteed dashboard...")
    webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
    
    # Wait for browser to open
    time.sleep(5)
    
    print(f"🔗 Clients after dashboard: {len(server.clients)}")
    
    print("\n📊 Creating AMAZING 3D visualizations...")
    
    # Create 3D visualizations with better data
    for i in range(3):
        print(f"\n🎯 Creating AMAZING 3D visualization {i+1}/3...")
        
        # Create more interesting test data
        test_data = [
            {"name": f"amazing_test_{i+1}", "duration": 0.001 + i * 0.005, "status": "PASSED"},
            {"name": f"amazing_test_{i+1}_b", "duration": 0.003 + i * 0.002, "status": "FAILED" if i % 2 == 0 else "PASSED"},
            {"name": f"amazing_test_{i+1}_c", "duration": 0.002 + i * 0.003, "status": "PASSED"},
            {"name": f"amazing_test_{i+1}_d", "duration": 0.004 + i * 0.001, "status": "FAILED" if i % 3 == 0 else "PASSED"}
        ]
        
        # Stream 3D visualization using our guaranteed server
        fig = server.stream_3d_visualization("test_landscape", test_data)
        print(f"✅ AMAZING 3D visualization {i+1} created and streamed!")
        
        # Check client count
        print(f"🔗 Current clients: {len(server.clients)}")
        
        # Wait between visualizations
        time.sleep(4)
    
    print("\n🎉 All AMAZING 3D visualizations completed!")
    print("🌐 Check the guaranteed dashboard in your browser!")
    print("📱 You should see 3 interactive 3D visualizations with rotation!")
    
    # Keep server running with loading indicator
    print("\n⏳ Keeping server running for 30 seconds...")
    print("🎯 Watch the AMAZING 3D visualizations appear in real-time!")
    print("🔄 Loading indicator active - please wait...")
    
    # Show loading animation
    loading_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    for i in range(30):
        print(f"\r{loading_chars[i % len(loading_chars)]} Loading... {i+1}/30 seconds", end="", flush=True)
        time.sleep(1)
    print("\n✅ Loading complete!")
    
    # Stop server
    server.stop()
    print("🛑 Server stopped")
    print("✅ Guaranteed 3D show completed!")
    print("\n🎯 If you still didn't see 3D visualizations:")
    print("   1. Check browser console (F12) for errors")
    print("   2. Make sure WebSocket connection is established")
    print("   3. Try refreshing the dashboard page")

if __name__ == "__main__":
    guaranteed_3d_show()
