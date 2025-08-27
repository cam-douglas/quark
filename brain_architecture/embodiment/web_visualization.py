#!/usr/bin/env python3
"""
Web-based Visualization for Quark's Balance Learning

This creates a real-time web interface to visualize Quark's balance learning
since the MuJoCo viewer isn't working on macOS.
"""

import asyncio
import websockets
import json
import logging
import argparse
import time
import sys
import os
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebVisualization:
    def __init__(self, host="127.0.0.1", port=8000, web_port=8080):
        self.host = host
        self.port = port
        self.web_port = web_port
        self.websocket = None
        self.simulation_data = queue.Queue(maxsize=1000)
        self.learning_stats = {
            "episode": 0,
            "step_count": 0,
            "episode_reward": 0.0,
            "exploration_rate": 0.0,
            "safety_score": 0.0
        }
        
    async def connect_to_brain(self):
        """Connect to the Brain-Body Interface via WebSocket."""
        try:
            uri = f"ws://{self.host}:{self.port}/ws/simulation"
            self.websocket = await websockets.connect(uri)
            logger.info(f"✅ Successfully connected to Brain-Body Interface at {uri}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Brain-Body Interface: {e}")
            return False
    
    async def send_sensory_data(self):
        """Send current simulation state as sensory data to the brain."""
        if self.websocket is None:
            return
        
        try:
            # Generate simulated sensory data for visualization
            # In a real scenario, this would come from MuJoCo
            simulated_state = {
                "timestamp": time.time(),
                "state_vector": [
                    0.5, 0.0, 0.0, 0.0, 0.0, 0.0,  # base position and rotation
                    0.1, 0.0,  # pendulum and counterweight angles
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # base velocity
                    0.0, 0.0   # pendulum and counterweight velocities
                ],
                "model_info": {
                    "num_joints": 8,
                    "num_actuators": 2,
                    "num_bodies": 3
                }
            }
            
            await self.websocket.send(json.dumps(simulated_state))
            
        except Exception as e:
            logger.error(f"Error sending sensory data: {e}")
    
    async def receive_motor_command(self):
        """Receive motor commands from the brain."""
        if self.websocket is None:
            return None
        
        try:
            # Wait for motor command with timeout
            message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
            motor_data = json.loads(message)
            
            if "actuators" in motor_data and "controls" in motor_data["actuators"]:
                return motor_data["actuators"]["controls"]
            else:
                logger.warning(f"Invalid motor command format: {motor_data}")
                return None
                
        except asyncio.TimeoutError:
            # No command received, use zero command
            return None
        except Exception as e:
            logger.error(f"Error receiving motor command: {e}")
            return None
    
    def update_learning_stats(self, controls, reward=None):
        """Update learning statistics for visualization."""
        self.learning_stats["step_count"] += 1
        
        if controls:
            # Extract motor control values
            pendulum_control = controls[0] if len(controls) > 0 else 0.0
            counterweight_control = controls[1] if len(controls) > 1 else 0.0
            
            # Store data for visualization
            data_point = {
                "timestamp": time.time(),
                "step": self.learning_stats["step_count"],
                "pendulum_control": pendulum_control,
                "counterweight_control": counterweight_control,
                "reward": reward or 0.0,
                "episode": self.learning_stats["episode"]
            }
            
            try:
                self.simulation_data.put_nowait(data_point)
            except queue.Full:
                # Remove oldest data point
                try:
                    self.simulation_data.get_nowait()
                    self.simulation_data.put_nowait(data_point)
                except:
                    pass
    
    async def run_simulation(self):
        """Run the main simulation loop."""
        logger.info("🔄 Starting Web Visualization of Balance Learning")
        episode = 1
        episode_steps = 0
        
        while True:
            try:
                # Send current sensory data to brain
                await self.send_sensory_data()
                
                # Receive motor commands from brain
                controls = await self.receive_motor_command()
                
                # Update learning statistics
                self.update_learning_stats(controls)
                
                episode_steps += 1
                
                # Simulate episode progression
                if episode_steps > 30:  # Simulate episode end
                    logger.info(f"💥 Episode {episode} finished after {episode_steps} steps.")
                    episode += 1
                    episode_steps = 0
                    self.learning_stats["episode"] = episode
                    
                    # Limit episodes for demo
                    if episode > 20:
                        logger.info("Simulation finished.")
                        break
                
                # Small delay for visualization
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                break
    
    async def run(self):
        """Main run method."""
        # Connect to brain
        if not await self.connect_to_brain():
            return
        
        try:
            await self.run_simulation()
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            if self.websocket:
                await self.websocket.close()

class WebServer(BaseHTTPRequestHandler):
    def __init__(self, *args, visualization=None, **kwargs):
        self.visualization = visualization
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html_content = self.generate_html()
            self.wfile.write(html_content.encode())
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Get latest data from visualization
            data = []
            while not self.visualization.simulation_data.empty():
                try:
                    data.append(self.visualization.simulation_data.get_nowait())
                except:
                    break
            
            response = {
                "data": data,
                "stats": self.visualization.learning_stats
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def generate_html(self):
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Quark Balance Learning Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .stats { display: flex; justify-content: space-around; margin-bottom: 20px; }
        .stat-box { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .balance-visual { text-align: center; font-size: 24px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Quark Balance Learning Visualization</h1>
            <p>Real-time visualization of Quark learning to balance a pendulum</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>Episode</h3>
                <div id="episode">-</div>
            </div>
            <div class="stat-box">
                <h3>Step Count</h3>
                <div id="step-count">-</div>
            </div>
            <div class="stat-box">
                <h3>Episode Reward</h3>
                <div id="episode-reward">-</div>
            </div>
            <div class="stat-box">
                <h3>Safety Score</h3>
                <div id="safety-score">-</div>
            </div>
        </div>
        
        <div class="balance-visual">
            <div id="balance-status">🔄 Initializing...</div>
        </div>
        
        <div class="chart-container">
            <h3>Motor Controls Over Time</h3>
            <div id="motor-controls-chart"></div>
        </div>
        
        <div class="chart-container">
            <h3>Learning Progress</h3>
            <div id="learning-progress-chart"></div>
        </div>
    </div>
    
    <script>
        let motorData = [];
        let learningData = [];
        
        function updateStats(stats) {
            document.getElementById('episode').textContent = stats.episode || 0;
            document.getElementById('step-count').textContent = stats.step_count || 0;
            document.getElementById('episode-reward').textContent = (stats.episode_reward || 0).toFixed(2);
            document.getElementById('safety-score').textContent = (stats.safety_score || 0).toFixed(2);
        }
        
        function updateBalanceStatus(data) {
            const statusDiv = document.getElementById('balance-status');
            if (data.length > 0) {
                const latest = data[data.length - 1];
                const pendulum = latest.pendulum_control || 0;
                const counterweight = latest.counterweight_control || 0;
                
                let status = '🔄 Learning to balance...';
                if (Math.abs(pendulum) < 0.1 && Math.abs(counterweight) < 0.1) {
                    status = '✅ Balanced!';
                } else if (Math.abs(pendulum) > 0.5 || Math.abs(counterweight) > 0.5) {
                    status = '⚠️ Active correction';
                }
                
                statusDiv.innerHTML = status + '<br><small>Pendulum: ' + pendulum.toFixed(2) + ' | Counterweight: ' + counterweight.toFixed(2) + '</small>';
            }
        }
        
        function updateCharts(data) {
            if (data.length === 0) return;
            
            // Motor Controls Chart
            const steps = data.map(d => d.step);
            const pendulumControls = data.map(d => d.pendulum_control || 0);
            const counterweightControls = data.map(d => d.counterweight_control || 0);
            
            const motorTrace = {
                x: steps,
                y: pendulumControls,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Pendulum Control',
                line: {color: 'red'}
            };
            
            const counterweightTrace = {
                x: steps,
                y: counterweightControls,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Counterweight Control',
                line: {color: 'blue'}
            };
            
            Plotly.newPlot('motor-controls-chart', [motorTrace, counterweightTrace], {
                title: 'Motor Control Values Over Time',
                xaxis: {title: 'Step'},
                yaxis: {title: 'Control Value', range: [-1, 1]}
            });
            
            // Learning Progress Chart
            const rewards = data.map(d => d.reward || 0);
            const learningTrace = {
                x: steps,
                y: rewards,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Reward',
                line: {color: 'green'}
            };
            
            Plotly.newPlot('learning-progress-chart', [learningTrace], {
                title: 'Learning Progress (Reward Over Time)',
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
                        updateBalanceStatus(data.data);
                        updateCharts(data.data);
                    }
                    if (data.stats) {
                        updateStats(data.stats);
                    }
                })
                .catch(error => console.error('Error fetching data:', error));
        }
        
        // Update data every 100ms
        setInterval(fetchData, 100);
        
        // Initial fetch
        fetchData();
    </script>
</body>
</html>
        """

def start_web_server(visualization, port=8080):
    """Start the web server in a separate thread."""
    class Handler(WebServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, visualization=visualization, **kwargs)
    
    server = HTTPServer(('localhost', port), Handler)
    logger.info(f"🌐 Web visualization server started at http://localhost:{port}")
    server.serve_forever()

async def main():
    parser = argparse.ArgumentParser(description="Web-based Balance Learning Visualization")
    parser.add_argument("--host", default="127.0.0.1", help="Brain interface host")
    parser.add_argument("--port", type=int, default=8000, help="Brain interface port")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port")
    
    args = parser.parse_args()
    
    visualization = WebVisualization(args.host, args.port, args.web_port)
    
    # Start web server in background thread
    web_thread = threading.Thread(target=start_web_server, args=(visualization, args.web_port), daemon=True)
    web_thread.start()
    
    # Wait a moment for web server to start
    await asyncio.sleep(1)
    
    logger.info(f"🚀 Open your browser to http://localhost:{args.web_port} to see Quark learning!")
    
    # Run the visualization
    await visualization.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Visualization interrupted by user")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
