#!/usr/bin/env python3
"""
Live 3D Visualization Server for Quark Stage N0 Benchmark

This script serves a web page with a live, 3D force-directed graph
representing the health and status of Quark's core Stage N0 modules.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any

import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path to allow module imports
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import Quark's brain modules
from brain_architecture.neural_core.consciousness_agent.proto_consciousness_mechanisms import ProtoConsciousnessMechanisms
from brain_architecture.neural_core.learning_agent.enhanced_neural_plasticity import EnhancedNeuralPlasticity
from brain_architecture.neural_core.self_organization_agent.advanced_self_organization import AdvancedSelfOrganization
from brain_architecture.neural_core.learning_agent.advanced_learning_integration import AdvancedLearningIntegration
from brain_architecture.neural_core.knowledge_agent.knowledge_graph_framework import KnowledgeGraphFramework
from brain_architecture.neural_core.safety_agent.enhanced_safety_protocols import EnhancedSafetyProtocols
from brain_architecture.neural_core.safety_agent.overconfidence_monitor import OverconfidenceMonitor
from brain_architecture.neural_core.monitoring.neural_activity_monitor import NeuralActivityMonitor
from brain_architecture.neural_core.analytics_agent.predictive_analytics import PredictiveAnalytics
from brain_architecture.neural_core.research_agent.external_research_connectors import ExternalResearchConnectors

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI and Socket.IO Setup ---
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Socket.IO server with proper configuration
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)

# Create Socket.IO app
socket_app = socketio.ASGIApp(sio, app)

# Serve static files (HTML)
web_dir = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open(web_dir / "index.html") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)

# --- Quark's Brain Initialization ---
logger.info("🧠 Initializing Quark's Stage N0 modules...")
brain_modules = {
    "consciousness": ProtoConsciousnessMechanisms(),
    "plasticity": EnhancedNeuralPlasticity(),
    "self_organization": AdvancedSelfOrganization(),
    "learning_integration": AdvancedLearningIntegration(),
    "knowledge_graph": KnowledgeGraphFramework(),
    "safety": EnhancedSafetyProtocols(),
    "overconfidence": OverconfidenceMonitor(),
    "activity_monitor": NeuralActivityMonitor(),
    "analytics": PredictiveAnalytics(),
    "research": ExternalResearchConnectors(),
}
logger.info("✅ All modules initialized.")

# --- Graph Data Structure ---
def get_graph_data() -> Dict[str, Any]:
    nodes = [
        {"id": "consciousness", "name": "Consciousness", "group": "Core"},
        {"id": "plasticity", "name": "Neural Plasticity", "group": "Learning"},
        {"id": "self_organization", "name": "Self-Organization", "group": "Learning"},
        {"id": "learning_integration", "name": "Learning Integration", "group": "Learning"},
        {"id": "knowledge_graph", "name": "Knowledge Graph", "group": "Knowledge"},
        {"id": "safety", "name": "Safety Protocols", "group": "Control"},
        {"id": "overconfidence", "name": "Overconfidence Monitor", "group": "Control"},
        {"id": "activity_monitor", "name": "Activity Monitor", "group": "Monitoring"},
        {"id": "analytics", "name": "Predictive Analytics", "group": "Monitoring"},
        {"id": "research", "name": "Research Connectors", "group": "Knowledge"},
    ]
    
    links = [
        {"source": "consciousness", "target": "activity_monitor"},
        {"source": "consciousness", "target": "plasticity"},
        {"source": "consciousness", "target": "learning_integration"},
        {"source": "learning_integration", "target": "plasticity"},
        {"source": "learning_integration", "target": "self_organization"},
        {"source": "learning_integration", "target": "knowledge_graph"},
        {"source": "knowledge_graph", "target": "research"},
        {"source": "safety", "target": "consciousness"},
        {"source": "safety", "target": "overconfidence"},
        {"source": "analytics", "target": "activity_monitor"},
        {"source": "analytics", "target": "learning_integration"},
    ]

    # Update nodes with live data
    for node in nodes:
        module = brain_modules.get(node["id"])
        if hasattr(module, 'get_system_health'):
            health = module.get_system_health()
            node["health_status"] = "Healthy" if health.get("healthy") else "Unhealthy"
            node["color"] = "#00ff00" if health.get("healthy") else "#ff0000"
        else:
            node["health_status"] = "Unknown"
            node["color"] = "#808080"
        
        # Add specific metrics
        metrics = {}
        if node["id"] == "consciousness" and hasattr(module, 'get_consciousness_metrics'):
            metrics = module.get_consciousness_metrics()
        elif node["id"] == "plasticity" and hasattr(module, 'get_plasticity_metrics'):
            metrics = module.get_plasticity_metrics()
        
        node["metrics"] = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if metrics.get("consciousness_level"):
             node["val"] = metrics["consciousness_level"] * 50 # val controls size

    return {"nodes": nodes, "links": links}

# --- WebSocket Event Handlers ---
@sio.event
async def connect(sid, environ):
    logger.info(f"🔗 Client connected: {sid}")
    initial_data = get_graph_data()
    await sio.emit('update_data', initial_data, to=sid)

@sio.event
def disconnect(sid):
    logger.info(f"👋 Client disconnected: {sid}")

# --- Background Task for Live Updates ---
def update_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        try:
            data = get_graph_data()
            loop.run_until_complete(sio.emit('update_data', data))
            time.sleep(2)  # Update interval
        except Exception as e:
            logger.error(f"Error in update loop: {e}")
            time.sleep(5)

if __name__ == '__main__':
    logger.info("🚀 Starting Quark Benchmark Visualization Server...")
    
    # Start the background thread
    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()
    
    # Run the web server with Socket.IO app
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
