#!/usr/bin/env python3
"""
Simple Web Visualization for Quark's Balance Learning

This creates a basic but reliable web interface to visualize Quark's learning.
"""

import asyncio
import websockets
import json
import logging
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleWebViz:
    def __init__(self, host="127.0.0.1", port=8000, web_port=8080):
        self.host = host
        self.port = port
        self.web_port = web_port
        self.websocket = None
        self.data_queue = queue.Queue(maxsize=100)
        self.running = False
        
    async def connect_and_run(self):
        """Connect to brain and run visualization."""
        try:
            uri = f"ws://{self.host}:{self.port}/ws/simulation"
            self.websocket = await websockets.connect(uri)
            logger.info(f"✅ Connected to brain at {uri}")
            
            self.running = True
            await self.run_visualization()
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
    
    async def run_visualization(self):
        """Run the main visualization loop."""
        logger.info("🔄 Starting balance learning visualization...")
        
        episode = 1
        step = 0
        
        while self.running:
            try:
                # Send simulated sensory data
                sensory_data = {
                    "timestamp": time.time(),
                    "state_vector": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0] + [0.0] * 8,
                    "model_info": {"num_joints": 8, "num_actuators": 2, "num_bodies": 3}
                }
                
                await self.websocket.send(json.dumps(sensory_data))
                
                # Try to receive motor commands
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
                    motor_data = json.loads(message)
                    
                    if "actuators" in motor_data and "controls" in motor_data["actuators"]:
                        controls = motor_data["actuators"]["controls"]
                        
                        # Store data for visualization
                        viz_data = {
                            "timestamp": time.time(),
                            "step": step,
                            "episode": episode,
                            "pendulum_control": controls[0] if len(controls) > 0 else 0.0,
                            "counterweight_control": controls[1] if len(controls) > 1 else 0.0,
                            "reward": -0.1,  # Simulated reward
                            "status": "Learning"
                        }
                        
                        try:
                            self.data_queue.put_nowait(viz_data)
                        except queue.Full:
                            # Remove oldest data
                            try:
                                self.data_queue.get_nowait()
                                self.data_queue.put_nowait(viz_data)
                            except:
                                pass
                        
                        logger.info(f"Episode {episode}, Step {step}: Pendulum={controls[0]:.3f}, Counterweight={controls[1]:.3f}")
                
                except asyncio.TimeoutError:
                    # No motor command received, continue
                    pass
                
                step += 1
                
                # Simulate episode progression
                if step > 30:
                    episode += 1
                    step = 0
                    logger.info(f"💥 Episode {episode-1} finished, starting Episode {episode}")
                    
                    if episode > 20:
                        logger.info("Simulation finished.")
                        break
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Visualization error: {e}")
                break
        
        if self.websocket:
            await self.websocket.close()

class WebHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, viz=None, **kwargs):
        self.viz = viz
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = self.generate_html()
            self.wfile.write(html.encode())
            
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Get data from queue
            data = []
            while not self.viz.data_queue.empty():
                try:
                    data.append(self.viz.data_queue.get_nowait())
                except:
                    break
            
            response = {"data": data}
            self.wfile.write(json.dumps(response).encode())
    
    def generate_html(self):
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Quark Balance Learning</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .status { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; text-align: center; }
        .chart { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .stats { display: flex; justify-content: space-around; margin-bottom: 20px; }
        .stat-box { background: white; padding: 15px; border-radius: 8px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Quark Balance Learning</h1>
            <p>Real-time visualization of Quark learning to balance</p>
        </div>
        
        <div class="status">
            <h2 id="status">🔄 Initializing...</h2>
            <p id="details">Connecting to Quark's brain...</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>Episode</h3>
                <div id="episode">-</div>
            </div>
            <div class="stat-box">
                <h3>Step</h3>
                <div id="step">-</div>
            </div>
            <div class="stat-box">
                <h3>Pendulum Control</h3>
                <div id="pendulum">-</div>
            </div>
            <div class="stat-box">
                <h3>Counterweight Control</h3>
                <div id="counterweight">-</div>
            </div>
        </div>
        
        <div class="chart">
            <h3>Motor Controls Over Time</h3>
            <div id="motor-chart"></div>
        </div>
        
        <div class="chart">
            <h3>Learning Progress</h3>
            <div id="progress-chart"></div>
        </div>
    </div>
    
    <script>
        let motorData = [];
        
        function updateStatus(data) {
            if (data.length > 0) {
                const latest = data[data.length - 1];
                
                document.getElementById('episode').textContent = latest.episode || 0;
                document.getElementById('step').textContent = latest.step || 0;
                document.getElementById('pendulum').textContent = (latest.pendulum_control || 0).toFixed(3);
                document.getElementById('counterweight').textContent = (latest.counterweight_control || 0).toFixed(3);
                
                // Update status
                const statusDiv = document.getElementById('status');
                const detailsDiv = document.getElementById('details');
                
                if (Math.abs(latest.pendulum_control) < 0.1 && Math.abs(latest.counterweight_control) < 0.1) {
                    statusDiv.innerHTML = '✅ Balanced!';
                    detailsDiv.innerHTML = 'Quark has learned to maintain balance';
                } else if (Math.abs(latest.pendulum_control) > 0.5 || Math.abs(latest.counterweight_control) > 0.5) {
                    statusDiv.innerHTML = '⚠️ Active Learning';
                    detailsDiv.innerHTML = 'Quark is actively correcting balance';
                } else {
                    statusDiv.innerHTML = '🔄 Learning...';
                    detailsDiv.innerHTML = 'Quark is learning to balance';
                }
            }
        }
        
        function updateCharts(data) {
            if (data.length === 0) return;
            
            const steps = data.map(d => d.step);
            const pendulum = data.map(d => d.pendulum_control || 0);
            const counterweight = data.map(d => d.counterweight_control || 0);
            const rewards = data.map(d => d.reward || 0);
            
            // Motor controls chart
            const motorTrace = {
                x: steps,
                y: pendulum,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Pendulum',
                line: {color: 'red'}
            };
            
            const counterweightTrace = {
                x: steps,
                y: counterweight,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Counterweight',
                line: {color: 'blue'}
            };
            
            Plotly.newPlot('motor-chart', [motorTrace, counterweightTrace], {
                title: 'Motor Control Values',
                xaxis: {title: 'Step'},
                yaxis: {title: 'Control Value', range: [-1, 1]}
            });
            
            // Progress chart
            const progressTrace = {
                x: steps,
                y: rewards,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Reward',
                line: {color: 'green'}
            };
            
            Plotly.newPlot('progress-chart', [progressTrace], {
                title: 'Learning Progress',
                xaxis: {title: 'Step'},
                yaxis: {title: 'Reward'}
            });
        }
        
        function fetchData() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    if (data.data && data.data.length > 0) {
                        motorData = data.data;
                        updateStatus(data.data);
                        updateCharts(data.data);
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        // Update every 200ms
        setInterval(fetchData, 200);
        
        // Initial fetch
        fetchData();
    </script>
</body>
</html>
        """

def start_web_server(viz, port=8080):
    """Start web server in background thread."""
    class Handler(WebHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, viz=viz, **kwargs)
    
    server = HTTPServer(('localhost', port), Handler)
    logger.info(f"🌐 Web server started at http://localhost:{port}")
    server.serve_forever()

async def main():
    viz = SimpleWebViz()
    
    # Start web server in background
    web_thread = threading.Thread(target=start_web_server, args=(viz, 8080), daemon=True)
    web_thread.start()
    
    # Wait for web server to start
    await asyncio.sleep(1)
    
    logger.info("🚀 Open your browser to http://localhost:8080 to see Quark learning!")
    
    # Run visualization
    await viz.connect_and_run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Visualization stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")
