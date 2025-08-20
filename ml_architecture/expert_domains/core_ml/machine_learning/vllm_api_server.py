#!/usr/bin/env python3
"""
vLLM Brain Simulation API Server
===============================

FastAPI-based REST API server that provides brain simulation services 
powered by vLLM high-throughput inference engine.

Features:
- RESTful API for brain simulation inference
- Real-time consciousness analysis endpoints
- Neural activity prediction services
- Training progress monitoring
- WebSocket support for real-time updates
- OpenAPI documentation and monitoring

Author: Quark Brain Team
Date: 2025-01-20
"""

import os, sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
import uuid
import numpy as np

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# vLLM integration
from ...........................................................vllm_brain_integration import (
    VLLMBrainEngine, VLLMConfig, BrainInferenceRequest, 
    BrainInferenceResponse, VLLMBrainTrainer
)
from ...........................................................vllm_training_pipeline import VLLMBrainTrainingPipeline, TrainingConfig, TrainingEpisode
from ...........................................................vllm_distributed_config import DistributedBrainTrainer, DistributedConfig

# Brain simulation imports
from ...........................................................brain_launcher_v3 import Brain, load_connectome

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models for API
class BrainStateRequest(BaseModel):
    """Request model for brain state analysis."""
    brain_modules: Dict[str, float] = Field(..., description="Brain module activation levels")
    consciousness_level: float = Field(..., ge=0.0, le=1.0, description="Current consciousness level")
    neural_activity: Optional[Dict[str, List[float]]] = Field(None, description="Neural activity patterns")
    development_stage: str = Field("F", description="Development stage (F, N0, N1)")
    analysis_type: str = Field("reasoning", description="Type of analysis (reasoning, consciousness, prediction)")
    custom_prompt: Optional[str] = Field(None, description="Custom analysis prompt")

class BrainAnalysisResponse(BaseModel):
    """Response model for brain analysis."""
    request_id: str = Field(..., description="Unique request identifier")
    analysis_text: str = Field(..., description="Generated analysis text")
    reasoning_trace: str = Field(..., description="Step-by-step reasoning")
    consciousness_feedback: Dict[str, float] = Field(..., description="Consciousness impact metrics")
    neural_predictions: Dict[str, Any] = Field(..., description="Predicted neural changes")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Analysis timestamp")

class TrainingRequest(BaseModel):
    """Request model for starting brain training."""
    model_name: Optional[str] = Field(None, description="vLLM model name")
    development_stages: List[str] = Field(["F"], description="Development stages to train")
    episodes_per_stage: int = Field(5, ge=1, le=100, description="Episodes per stage")
    steps_per_episode: int = Field(50, ge=10, le=1000, description="Steps per episode")
    consciousness_analysis_interval: int = Field(20, ge=5, le=100, description="Analysis interval")
    output_directory: Optional[str] = Field(None, description="Training output directory")

class TrainingStatusResponse(BaseModel):
    """Response model for training status."""
    training_id: str = Field(..., description="Training session identifier")
    status: str = Field(..., description="Training status (running, completed, failed)")
    current_stage: Optional[str] = Field(None, description="Current development stage")
    current_episode: int = Field(0, description="Current episode number")
    total_episodes: int = Field(0, description="Total episodes")
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage")
    consciousness_level: float = Field(0.0, ge=0.0, le=1.0, description="Current consciousness level")
    elapsed_time_minutes: float = Field(0.0, description="Elapsed training time")
    estimated_remaining_minutes: float = Field(0.0, description="Estimated remaining time")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")

class ServerStatusResponse(BaseModel):
    """Response model for server status."""
    status: str = Field(..., description="Server status")
    vllm_engine_status: str = Field(..., description="vLLM engine status")
    total_requests: int = Field(..., description="Total requests processed")
    active_training_sessions: int = Field(..., description="Active training sessions")
    performance_metrics: Dict[str, float] = Field(..., description="Server performance metrics")
    hardware_info: Dict[str, Any] = Field(..., description="Hardware information")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")

# Global state management
class ServerState:
    """Global server state management."""
    
    def __init__(self):
        self.vllm_engine: Optional[VLLMBrainEngine] = None
        self.training_sessions: Dict[str, VLLMBrainTrainingPipeline] = {}
        self.websocket_connections: List[WebSocket] = []
        self.request_count = 0
        self.start_time = datetime.now()
        self.server_config = {}
    
    async def initialize_vllm_engine(self, config: VLLMConfig):
        """Initialize vLLM engine."""
        self.vllm_engine = VLLMBrainEngine(config)
        success = await self.vllm_engine.initialize_engine()
        if not success:
            raise RuntimeError("Failed to initialize vLLM engine")
        logger.info("✅ vLLM engine initialized")
    
    async def shutdown(self):
        """Shutdown server components."""
        if self.vllm_engine:
            await self.vllm_engine.shutdown()
        
        for training_session in self.training_sessions.values():
            await training_session.shutdown()
        
        logger.info("✅ Server shutdown complete")

# Global server state
server_state = ServerState()

# Create FastAPI app
app = FastAPI(
    title="vLLM Brain Simulation API",
    description="High-performance brain simulation and consciousness analysis API powered by vLLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication dependency (optional)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from authentication token."""
    if credentials is None:
        return None
    # Implement your authentication logic here
    return {"user_id": "anonymous"}

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    logger.info("🚀 Starting vLLM Brain Simulation API Server")
    
    # Configure vLLM engine
    vllm_config = VLLMConfig(
        model_name=os.getenv("VLLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "8192")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTIL", "0.85"))
    )
    
    await server_state.initialize_vllm_engine(vllm_config)
    logger.info("✅ Server startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    logger.info("🔄 Shutting down server...")
    await server_state.shutdown()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with server information."""
    return {
        "service": "vLLM Brain Simulation API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=ServerStatusResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - server_state.start_time).total_seconds()
    
    # Get performance metrics
    performance_metrics = {}
    if server_state.vllm_engine:
        performance_metrics = await server_state.vllm_engine.get_performance_metrics()
    
    return ServerStatusResponse(
        status="healthy" if server_state.vllm_engine else "unhealthy",
        vllm_engine_status="ready" if server_state.vllm_engine else "not_initialized",
        total_requests=server_state.request_count,
        active_training_sessions=len(server_state.training_sessions),
        performance_metrics=performance_metrics,
        hardware_info={
            "gpu_available": str(torch.cuda.is_available()),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        uptime_seconds=uptime
    )

@app.post("/analyze", response_model=BrainAnalysisResponse)
async def analyze_brain_state(
    request: BrainStateRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Analyze brain state using vLLM inference."""
    if not server_state.vllm_engine:
        raise HTTPException(status_code=503, detail="vLLM engine not initialized")
    
    server_state.request_count += 1
    request_id = str(uuid.uuid4())
    
    try:
        # Convert request to vLLM inference request
        neural_activity = {}
        if request.neural_activity:
            for module, activity in request.neural_activity.items():
                neural_activity[module] = np.array(activity)
        
        brain_request = BrainInferenceRequest(
            request_id=request_id,
            brain_state={
                "active_modules": list(request.brain_modules.keys()),
                "module_levels": request.brain_modules,
                "stage": request.development_stage,
                "synchrony": np.mean(list(request.brain_modules.values())),
                "integration": request.consciousness_level
            },
            neural_activity=neural_activity,
            consciousness_level=request.consciousness_level,
            task_type=request.analysis_type,
            prompt_template=request.custom_prompt or f"Analyze brain state in {request.development_stage} stage"
        )
        
        # Process inference
        response = await server_state.vllm_engine.process_brain_inference(brain_request)
        
        # Broadcast to WebSocket connections
        background_tasks.add_task(broadcast_analysis_update, {
            "type": "analysis_complete",
            "request_id": request_id,
            "consciousness_level": request.consciousness_level,
            "confidence": response.confidence_score
        })
        
        return BrainAnalysisResponse(
            request_id=request_id,
            analysis_text=response.generated_text,
            reasoning_trace=response.reasoning_trace,
            consciousness_feedback=response.consciousness_feedback,
            neural_predictions=response.neural_predictions,
            confidence_score=response.confidence_score,
            processing_time_ms=response.processing_time_ms,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/training/start", response_model=Dict[str, str])
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Start brain simulation training."""
    training_id = str(uuid.uuid4())
    
    try:
        # Create training configuration
        training_config = TrainingConfig(
            vllm_model=request.model_name or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            development_stages=request.development_stages,
            episodes_per_stage=request.episodes_per_stage,
            steps_per_episode=request.steps_per_episode,
            consciousness_analysis_interval=request.consciousness_analysis_interval,
            output_dir=request.output_directory or f"training_outputs_{training_id}"
        )
        
        # Create training pipeline
        training_pipeline = VLLMBrainTrainingPipeline(training_config)
        await training_pipeline.initialize()
        
        # Store training session
        server_state.training_sessions[training_id] = training_pipeline
        
        # Start training in background
        background_tasks.add_task(run_training_session, training_id, training_pipeline)
        
        logger.info(f"🎯 Started training session: {training_id}")
        return {"training_id": training_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"❌ Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=f"Training start failed: {str(e)}")

@app.get("/training/{training_id}/status", response_model=TrainingStatusResponse)
async def get_training_status(training_id: str, user=Depends(get_current_user)):
    """Get training session status."""
    if training_id not in server_state.training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    training_pipeline = server_state.training_sessions[training_id]
    
    # Calculate progress and metrics
    total_episodes = sum(training_pipeline.config.episodes_per_stage for _ in training_pipeline.config.development_stages)
    completed_episodes = len(training_pipeline.training_history)
    progress = (completed_episodes / max(total_episodes, 1)) * 100
    
    # Get current consciousness level
    current_consciousness = 0.0
    if training_pipeline.training_history:
        latest_episode = training_pipeline.training_history[-1]
        if latest_episode.consciousness_evolution:
            current_consciousness = latest_episode.consciousness_evolution[-1]
    
    # Calculate timing
    elapsed_time = 0.0
    estimated_remaining = 0.0
    if training_pipeline.training_start_time:
        elapsed_time = (datetime.now() - training_pipeline.training_start_time).total_seconds() / 60
        if progress > 0:
            estimated_remaining = (elapsed_time / progress) * (100 - progress)
    
    return TrainingStatusResponse(
        training_id=training_id,
        status="running" if completed_episodes < total_episodes else "completed",
        current_stage=training_pipeline.config.development_stages[0] if training_pipeline.config.development_stages else None,
        current_episode=completed_episodes,
        total_episodes=total_episodes,
        progress_percentage=progress,
        consciousness_level=current_consciousness,
        elapsed_time_minutes=elapsed_time,
        estimated_remaining_minutes=estimated_remaining,
        performance_metrics={
            "total_steps": training_pipeline.total_steps_trained,
            "consciousness_analyses": training_pipeline.total_consciousness_analyses
        }
    )

@app.delete("/training/{training_id}")
async def stop_training(training_id: str, user=Depends(get_current_user)):
    """Stop training session."""
    if training_id not in server_state.training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    training_pipeline = server_state.training_sessions[training_id]
    await training_pipeline.shutdown()
    del server_state.training_sessions[training_id]
    
    logger.info(f"🛑 Stopped training session: {training_id}")
    return {"message": "Training stopped", "training_id": training_id}

@app.get("/training", response_model=List[Dict[str, Any]])
async def list_training_sessions(user=Depends(get_current_user)):
    """List all training sessions."""
    sessions = []
    for training_id, pipeline in server_state.training_sessions.items():
        sessions.append({
            "training_id": training_id,
            "status": "running",
            "episodes_completed": len(pipeline.training_history),
            "start_time": pipeline.training_start_time.isoformat() if pipeline.training_start_time else None
        })
    return sessions

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    server_state.websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            
    except WebSocketDisconnect:
        server_state.websocket_connections.remove(websocket)
        logger.info("WebSocket client disconnected")

@app.get("/performance/metrics", response_model=Dict[str, Any])
async def get_performance_metrics(user=Depends(get_current_user)):
    """Get detailed performance metrics."""
    metrics = {}
    
    if server_state.vllm_engine:
        vllm_metrics = await server_state.vllm_engine.get_performance_metrics()
        metrics["vllm"] = vllm_metrics
    
    metrics["server"] = {
        "total_requests": server_state.request_count,
        "active_training_sessions": len(server_state.training_sessions),
        "websocket_connections": len(server_state.websocket_connections),
        "uptime_seconds": (datetime.now() - server_state.start_time).total_seconds()
    }
    
    return metrics

@app.get("/models", response_model=List[Dict[str, str]])
async def list_available_models():
    """List available vLLM models."""
    from ...........................................................deepseek_r1_trainer import DeepSeekConfig
    
    models = []
    for key, config in DeepSeekConfig.MODELS.items():
        models.append({
            "model_id": key,
            "model_name": config["name"],
            "parameters": config["params"],
            "memory_gb": str(config["memory_gb"]),
            "recommended_for": config["recommended_for"]
        })
    
    return models

# Background tasks
async def run_training_session(training_id: str, training_pipeline: VLLMBrainTrainingPipeline):
    """Run training session in background."""
    try:
        logger.info(f"🏃 Running training session: {training_id}")
        await training_pipeline.run_full_training()
        
        # Broadcast completion
        await broadcast_training_update({
            "type": "training_complete",
            "training_id": training_id,
            "status": "completed"
        })
        
    except Exception as e:
        logger.error(f"❌ Training session {training_id} failed: {e}")
        await broadcast_training_update({
            "type": "training_error",
            "training_id": training_id,
            "error": str(e)
        })

async def broadcast_analysis_update(data: Dict[str, Any]):
    """Broadcast analysis update to WebSocket clients."""
    message = json.dumps(data)
    disconnected = []
    
    for websocket in server_state.websocket_connections:
        try:
            await websocket.send_text(message)
        except:
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for ws in disconnected:
        server_state.websocket_connections.remove(ws)

async def broadcast_training_update(data: Dict[str, Any]):
    """Broadcast training update to WebSocket clients."""
    await broadcast_analysis_update(data)

# Development server configuration
def create_app_config():
    """Create development server configuration."""
    return {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", "8000")),
        "reload": os.getenv("ENVIRONMENT") == "development",
        "workers": 1,  # Single worker to maintain state
        "log_level": "info"
    }

if __name__ == "__main__":
    config = create_app_config()
    uvicorn.run("vllm_api_server:app", **config)
