#!/usr/bin/env python3
"""
Debug Live 3D - Shows exactly what's happening with live streaming
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

class Debug3DServer:
    """Debug 3D server that shows exactly what's happening."""
    
    def __init__(self, port=8005):
        self.port = port
        self.clients = set()
        self.running = False
        self.server = None
        self.message_count = 0
        
    async def handler(self, websocket):
        """Handle WebSocket connections."""
        self.clients.add(websocket)
        print(f"🔗 New client connected. Total: {len(self.clients)}")
        
        # Send immediate confirmation
        await websocket.send(json.dumps({
            "type": "connection_confirmed",
            "message": f"Connected to Debug 3D Server on port {self.port}",
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
        print(f"🚀 Debug 3D server started on ws://127.0.0.1:{self.port}")
        
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
    
    def create_debug_3d_visualization(self, test_results):
        """Create a debug 3D visualization."""
        if not test_results:
            return None
        
        # Create simple 3D data
        x_vals = []
        y_vals = []
        z_vals = []
        colors = []
        text_labels = []
        
        for i, test in enumerate(test_results):
            x_vals.append(i)
            y_vals.append(test.get('duration', 0))
            z_vals.append(1 if test.get('status') == 'PASSED' else 0)
            colors.append('lime' if test.get('status') == 'PASSED' else 'red')
            text_labels.append(f"Test: {test.get('name', 'Unknown')}<br>Duration: {test.get('duration', 0):.4f}s<br>Status: {test.get('status', 'Unknown')}")
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers+text',
            marker=dict(
                size=15,
                color=colors,
                opacity=0.8,
                line=dict(width=3, color='white'),
                symbol='diamond'
            ),
            text=[f"T{i+1}" for i in range(len(test_results))],
            textposition="middle center",
            hovertext=text_labels,
            hoverinfo='text',
            name='Debug Test Results'
        )])
        
        fig.update_layout(
            title='🐛 Debug 3D Test Results',
            scene=dict(
                xaxis_title='Test Index',
                yaxis_title='Duration (seconds)',
                zaxis_title='Success Rate',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            template='plotly_dark'
        )
        
        return fig
    
    def stream_3d_visualization(self, viz_type, data):
        """Stream 3D visualization data to connected clients."""
        try:
            print(f"🎯 Attempting to create 3D visualization: {viz_type}")
            print(f"📊 Data: {data}")
            
            if viz_type == "test_landscape":
                fig = self.create_debug_3d_visualization(data)
            else:
                print(f"⚠️ Unknown visualization type: {viz_type}")
                return None
            
            if fig:
                print(f"✅ 3D figure created successfully")
                
                # Convert to dict for JSON serialization
                fig_dict = fig.to_dict()
                print(f"📝 Figure converted to dict, keys: {list(fig_dict.keys())}")
                
                # Create message
                message = {
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
                }
                
                print(f"📨 Message created, size: {len(json.dumps(message))} characters")
                
                # Stream to all clients
                async def broadcast():
                    if self.clients:
                        print(f"📡 Broadcasting to {len(self.clients)} clients...")
                        results = await asyncio.gather(
                            *[client.send(json.dumps(message)) for client in self.clients],
                            return_exceptions=True
                        )
                        
                        # Check results
                        for i, result in enumerate(results):
                            if isinstance(result, Exception):
                                print(f"❌ Client {i} broadcast failed: {result}")
                            else:
                                print(f"✅ Client {i} broadcast successful")
                    else:
                        print("⚠️ No clients connected to broadcast to")
                
                # Schedule broadcast
                try:
                    if hasattr(self, 'server_thread') and self.server_thread.is_alive():
                        print("🔄 Using server's event loop for broadcast")
                        asyncio.run_coroutine_threadsafe(broadcast(), asyncio.get_event_loop())
                    else:
                        print("🔄 Creating new event loop for broadcast")
                        asyncio.run(broadcast())
                except Exception as e:
                    print(f"❌ Broadcast error: {e}")
                    import traceback
                    traceback.print_exc()
                
                self.message_count += 1
                print(f"📡 3D visualization broadcasted to {len(self.clients)} clients (Message #{self.message_count})")
                return fig
            else:
                print("❌ No figure created")
                
        except Exception as e:
            print(f"❌ Error creating 3D visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

def create_debug_dashboard():
    """Create a debug dashboard that shows exactly what's happening."""
    dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <title>🐛 Debug Live 3D Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: 'Arial', sans-serif; 
            margin: 0; 
            padding: 20px;
            background: #1a1a1a;
            color: white;
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
        }
        .status { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 15px; 
            margin-bottom: 30px;
            text-align: center;
        }
        .viz { 
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 25px;
            margin: 25px 0;
            border: 2px solid rgba(255,255,255,0.3);
        }
        .log { 
            background: rgba(0,0,0,0.7);
            padding: 20px;
            border-radius: 15px;
            margin-top: 30px;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        .connected { color: #00ff88; font-weight: bold; }
        .disconnected { color: #ff4757; font-weight: bold; }
        .info { color: #ffd700; }
        .success { color: #00ff88; font-weight: bold; }
        .error { color: #ff4757; font-weight: bold; }
        .debug { color: #00ffff; }
        .test-button {
            background: #ff6b35;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐛 Debug Live 3D Dashboard</h1>
        <p>Port 8005 - Shows exactly what's happening with live streaming</p>
        <button class="test-button" onclick="testConnection()">🧪 Test Connection</button>
        <button class="test-button" onclick="clearAll()">🗑️ Clear All</button>
        <button class="test-button" onclick="showDebugInfo()">🔍 Show Debug Info</button>
    </div>
    
    <div class="status">
        <div><strong>Status:</strong> <span id="status" class="disconnected">Disconnected</span></div>
        <div><strong>Messages:</strong> <span id="msg-count">0</span></div>
        <div><strong>3D Visualizations:</strong> <span id="viz-count">0</span></div>
        <div><strong>Last Update:</strong> <span id="last-update">Never</span></div>
        <div><strong>WebSocket State:</strong> <span id="ws-state">Unknown</span></div>
    </div>
    
    <div id="visualizations"></div>
    
    <div class="log">
        <h3>📋 Debug Log</h3>
        <div id="log-entries"></div>
    </div>
    
    <script>
        let messageCount = 0;
        let vizCount = 0;
        let ws;
        let debugMode = true;
        
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
        
        function updateWebSocketState() {
            if (ws) {
                const states = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'];
                document.getElementById('ws-state').textContent = states[ws.readyState];
            } else {
                document.getElementById('ws-state').textContent = 'NULL';
            }
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
        
        function showDebugInfo() {
            log('🔍 Debug Information:', 'debug');
            log(`WebSocket: ${ws ? 'exists' : 'null'}`, 'debug');
            if (ws) {
                log(`Ready State: ${ws.readyState}`, 'debug');
                log(`URL: ${ws.url}`, 'debug');
            }
            log(`Message Count: ${messageCount}`, 'debug');
            log(`Viz Count: ${vizCount}`, 'debug');
            log(`Debug Mode: ${debugMode}`, 'debug');
        }
        
        function connect() {
            log('🔌 Connecting to debug server on port 8005...', 'info');
            ws = new WebSocket('ws://127.0.0.1:8005');
            
            ws.onopen = function() {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'connected';
                log('✅ Connected to debug 3D server', 'success');
                updateLastUpdate();
                updateWebSocketState();
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
                updateWebSocketState();
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
            
            if (!value) {
                log(`❌ No value field`, 'error');
                return;
            }
            
            if (!value.plot_data) {
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
        
        log('🚀 Debug 3D dashboard initialized', 'success');
        log('🌐 Waiting for 3D visualizations...', 'info');
        log('💡 Use browser console (F12) to see detailed logs', 'info');
        
        // Update WebSocket state periodically
        setInterval(updateWebSocketState, 1000);
        
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

def debug_live_3d():
    """Debug live 3D visualization to see exactly what's happening."""
    print("🐛 Debug Live 3D")
    print("=" * 40)
    
    # Start debug server on port 8005
    print("🚀 Starting debug server on port 8005...")
    server = Debug3DServer(port=8005)
    server.start()
    
    print(f"✅ Server running on port {server.port}")
    print(f"🔗 Initial clients: {len(server.clients)}")
    
    # Create the debug dashboard
    dashboard_html = create_debug_dashboard()
    dashboard_path = "testing/visualizations/debug_live_3d_dashboard.html"
    
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    
    print(f"🌐 Created debug dashboard: {dashboard_path}")
    
    # Open dashboard
    print("🌐 Opening debug dashboard...")
    webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
    
    # Wait for browser to open
    time.sleep(5)
    
    print(f"🔗 Clients after dashboard: {len(server.clients)}")
    
    print("\n📊 Creating debug 3D visualizations...")
    
    # Create 3D visualizations with detailed logging
    for i in range(3):
        print(f"\n🎯 Creating debug 3D visualization {i+1}/3...")
        
        # Create test data
        test_data = [
            {"name": f"debug_test_{i+1}", "duration": 0.001 + i * 0.002, "status": "PASSED"},
            {"name": f"debug_test_{i+1}_b", "duration": 0.002 + i * 0.001, "status": "FAILED" if i % 2 == 0 else "PASSED"}
        ]
        
        print(f"📊 Test data: {test_data}")
        
        # Stream 3D visualization using our debug server
        fig = server.stream_3d_visualization("test_landscape", test_data)
        print(f"✅ Debug 3D visualization {i+1} created and streamed!")
        
        # Check client count
        print(f"🔗 Current clients: {len(server.clients)}")
        
        # Wait between visualizations
        time.sleep(4)
    
    print("\n🎉 All debug 3D visualizations completed!")
    print("🌐 Check the debug dashboard in your browser!")
    print("📱 You should see 3 interactive 3D visualizations with detailed logging!")
    
    # Keep server running
    print("\n⏳ Keeping server running for 30 seconds...")
    print("🎯 Watch the debug dashboard for detailed information!")
    
    # Show loading animation
    loading_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    for i in range(30):
        print(f"\r{loading_chars[i % len(loading_chars)]} Running... {i+1}/30 seconds", end="", flush=True)
        time.sleep(1)
    print("\n✅ Debug session complete!")
    
    # Stop server
    server.stop()
    print("🛑 Server stopped")
    print("✅ Debug live 3D completed!")
    print("\n🐛 Debug Information:")
    print("   - Check browser console (F12) for detailed logs")
    print("   - Look at the debug log in the dashboard")
    print("   - Check WebSocket connection status")

if __name__ == "__main__":
    debug_live_3d()
