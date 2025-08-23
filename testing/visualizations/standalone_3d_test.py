#!/usr/bin/env python3
"""
Standalone 3D Test - Completely self-contained, no external dependencies.
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

class Standalone3DServer:
    """Standalone 3D server with integrated 3D visualization."""
    
    def __init__(self, port=8002):
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
        print(f"🚀 Standalone 3D server started on ws://127.0.0.1:{self.port}")
        
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
    
    def create_3d_visualization(self, test_results):
        """Create a 3D visualization of test results."""
        if not test_results:
            return None
        
        # Extract data for 3D plotting
        x_vals = []  # Test index
        y_vals = []  # Duration
        z_vals = []  # Success rate (1 for pass, 0 for fail)
        colors = []
        text_labels = []
        
        for i, test in enumerate(test_results):
            x_vals.append(i)
            y_vals.append(test.get('duration', 0))
            z_vals.append(1 if test.get('status') == 'PASSED' else 0)
            colors.append('green' if test.get('status') == 'PASSED' else 'red')
            text_labels.append(f"Test: {test.get('name', 'Unknown')}<br>Duration: {test.get('duration', 0):.4f}s<br>Status: {test.get('status', 'Unknown')}")
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers+text',
            marker=dict(
                size=12,
                color=colors,
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=[f"T{i+1}" for i in range(len(test_results))],
            textposition="middle center",
            hovertext=text_labels,
            hoverinfo='text',
            name='Test Results'
        )])
        
        # Add surface for success rate
        if len(x_vals) > 1:
            x_surface = np.linspace(0, len(x_vals)-1, 10)
            y_surface = np.linspace(min(y_vals), max(y_vals), 10)
            X, Y = np.meshgrid(x_surface, y_surface)
            Z = np.ones_like(X) * 0.5  # Success threshold plane
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                opacity=0.3,
                colorscale='Viridis',
                showscale=False,
                name='Success Threshold'
            ))
        
        fig.update_layout(
            title='🧪 3D Test Results Landscape',
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
            if viz_type == "test_landscape":
                fig = self.create_3d_visualization(data)
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

def standalone_3d_test():
    """Standalone 3D visualization test."""
    print("🧪 Standalone 3D Test")
    print("=" * 40)
    
    # Start standalone server on port 8002
    print("🚀 Starting standalone server on port 8002...")
    server = Standalone3DServer(port=8002)
    server.start()
    
    print(f"✅ Server running on port {server.port}")
    print(f"🔗 Initial clients: {len(server.clients)}")
    
    # Create a simple HTML dashboard that connects to port 8002
    dashboard_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>🎯 Standalone 3D Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #1a1a1a;
            color: white;
        }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .status {{ 
            background: rgba(255,255,255,0.1); 
            padding: 15px; 
            border-radius: 10px; 
            margin-bottom: 20px;
            text-align: center;
        }}
        .viz {{ 
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .log {{ 
            background: rgba(0,0,0,0.5);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }}
        .connected {{ color: #00ff88; }}
        .disconnected {{ color: #ff4757; }}
        .info {{ color: #ffd700; }}
        .success {{ color: #00ff88; }}
        .error {{ color: #ff4757; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Standalone 3D Dashboard</h1>
        <p>Port 8002 - No external dependencies</p>
    </div>
    
    <div class="status">
        <div><strong>Status:</strong> <span id="status" class="disconnected">Disconnected</span></div>
        <div><strong>Messages:</strong> <span id="msg-count">0</span></div>
        <div><strong>3D Visualizations:</strong> <span id="viz-count">0</span></div>
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
        
        function log(message, type = 'info') {{
            const logDiv = document.getElementById('log-entries');
            const entry = document.createElement('div');
            entry.className = type;
            entry.textContent = `[${{new Date().toLocaleTimeString()}}] ${{message}}`;
            logDiv.appendChild(entry);
            logDiv.scrollTop = logDiv.scrollHeight;
            console.log(message);
        }}
        
        function connect() {{
            log('🔌 Connecting to standalone server on port 8002...', 'info');
            ws = new WebSocket('ws://127.0.0.1:8002');
            
            ws.onopen = function() {{
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'connected';
                log('✅ Connected to standalone server', 'success');
            }};
            
            ws.onmessage = function(event) {{
                messageCount++;
                document.getElementById('msg-count').textContent = messageCount;
                
                try {{
                    const data = JSON.parse(event.data);
                    log(`📨 Received: ${{data.series_id}}`, 'info');
                    
                    // Handle 3D visualizations
                    if (data.series_id && data.series_id.startsWith('3d_')) {{
                        log(`🎯 Processing 3D: ${{data.series_id}}`, 'success');
                        handle3DVisualization(data);
                    }}
                    
                }} catch (e) {{
                    log(`❌ Error parsing message: ${{e.message}}`, 'error');
                }}
            }};
            
            ws.onclose = function() {{
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').className = 'disconnected';
                log('❌ Disconnected from server', 'error');
            }};
            
            ws.onerror = function() {{
                log('⚠️ WebSocket error', 'error');
            }};
        }}
        
        function handle3DVisualization(data) {{
            const {{ series_id, value }} = data;
            
            log(`🔍 Processing 3D data for: ${{series_id}}`, 'info');
            
            if (!value || !value.plot_data) {{
                log(`❌ No plot_data found`, 'error');
                return;
            }}
            
            const plot_data = value.plot_data;
            log(`📈 Plot data keys: ${{Object.keys(plot_data).join(', ')}}`, 'info');
            
            if (!plot_data.figure) {{
                log(`❌ No figure in plot_data`, 'error');
                return;
            }}
            
            log(`🎨 Creating 3D visualization...`, 'success');
            
            try {{
                // Create container
                const container = document.createElement('div');
                container.className = 'viz';
                container.id = `viz-${{series_id}}`;
                
                const title = document.createElement('h3');
                title.textContent = `🎨 ${{series_id.replace('3d_', '3D ')}}`;
                title.style.color = '#ffd700';
                
                const chartDiv = document.createElement('div');
                chartDiv.id = `chart-${{series_id}}`;
                chartDiv.style.height = '600px';
                chartDiv.style.width = '100%';
                
                container.appendChild(title);
                container.appendChild(chartDiv);
                
                // Remove any existing visualization with same ID
                const existing = document.getElementById(`viz-${{series_id}}`);
                if (existing) {{
                    existing.remove();
                }}
                
                document.getElementById('visualizations').appendChild(container);
                
                // Create Plotly 3D visualization
                Plotly.newPlot(chartDiv.id, plot_data.data, plot_data.layout, {{
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                    displaylogo: false
                }});
                
                vizCount++;
                document.getElementById('viz-count').textContent = vizCount;
                
                log(`✅ 3D visualization created: ${{series_id}}`, 'success');
                
            }} catch (error) {{
                log(`❌ Error creating 3D viz: ${{error.message}}`, 'error');
                console.error('Full error:', error);
            }}
        }}
        
        // Start connection
        connect();
        
        log('🚀 Standalone 3D dashboard initialized', 'info');
        log('🌐 Waiting for 3D visualizations...', 'info');
    </script>
</body>
</html>"""
    
    # Save dashboard
    dashboard_path = "testing/visualizations/standalone_3d_dashboard.html"
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    
    # Open dashboard
    print(f"🌐 Opening standalone dashboard: {dashboard_path}")
    webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
    
    # Wait for browser to open
    time.sleep(3)
    
    print(f"🔗 Clients after dashboard: {len(server.clients)}")
    
    print("\n📊 Creating 3D visualizations...")
    
    # Create 3D visualizations
    for i in range(3):
        print(f"\n🎯 Creating 3D visualization {i+1}/3...")
        
        # Create test data
        test_data = [
            {"name": f"standalone_test_{i+1}", "duration": 0.001 + i * 0.002, "status": "PASSED"},
            {"name": f"standalone_test_{i+1}_b", "duration": 0.002 + i * 0.001, "status": "FAILED" if i % 2 == 0 else "PASSED"}
        ]
        
        # Stream 3D visualization using our standalone server
        fig = server.stream_3d_visualization("test_landscape", test_data)
        print(f"✅ 3D visualization {i+1} created and streamed!")
        
        # Check client count
        print(f"🔗 Current clients: {len(server.clients)}")
        
        # Wait between visualizations
        time.sleep(3)
    
    print("\n🎉 All 3D visualizations completed!")
    print("🌐 Check the standalone dashboard in your browser!")
    print("📱 You should see 3 interactive 3D visualizations")
    
    # Keep server running
    print("\n⏳ Keeping server running for 20 seconds...")
    print("🎯 Watch the 3D visualizations appear in real-time!")
    time.sleep(20)
    
    # Stop server
    server.stop()
    print("🛑 Server stopped")
    print("✅ Standalone 3D test completed!")

if __name__ == "__main__":
    standalone_3d_test()
