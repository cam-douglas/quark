#!/usr/bin/env python3
"""
Main Orchestrator for Exponential Learning System
Coordinates all components and provides unified interface
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

class ExponentialLearningOrchestrator:
    """
    Main orchestrator that coordinates all exponential learning components
    Provides unified interface for starting, monitoring, and controlling the system
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "exponential_learning_config.yaml"
        self.config = self.load_config()
        
        # Component references
        self.learning_system = None
        self.research_hub = None
        self.knowledge_synthesizer = None
        self.cloud_orchestrator = None
        self.validation_system = None
        self.neuro_agent = None
        
        # System state
        self.is_running = False
        self.start_time = None
        self.active_sessions = []
        self.system_metrics = {
            "total_learning_cycles": 0,
            "total_knowledge_gained": 0,
            "total_research_queries": 0,
            "total_training_jobs": 0,
            "system_uptime": 0
        }
        
        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🎯 Exponential Learning Orchestrator initialized")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        default_config = {
            "max_concurrent_sessions": 5,
            "research_interval_seconds": 60,
            "synthesis_interval_seconds": 300,
            "training_interval_seconds": 1800,
            "validation_interval_seconds": 600,
            "log_level": "INFO",
            "enable_cloud_training": True,
            "enable_research": True,
            "enable_synthesis": True,
            "enable_validation": True
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                    default_config.update(config)
                    logger.info(f"✅ Loaded configuration from {self.config_path}")
            else:
                logger.info(f"📝 Using default configuration (create {self.config_path} to customize)")
                
        except Exception as e:
            logger.warning(f"⚠️ Error loading config: {e}, using defaults")
        
        return default_config
    
    async def initialize_system(self):
        """Initialize all system components"""
        logger.info("🚀 Initializing exponential learning system...")
        
        try:
            # Import components
            from exponential_learning_system import ExponentialLearningSystem
            from research_agents import ResearchAgentHub
            from knowledge_synthesizer import KnowledgeSynthesizer
            from cloud_training_orchestrator import CloudTrainingOrchestrator
            from knowledge_validation_system import KnowledgeValidationSystem
            from neuro_agent_enhancer import NeuroAgentEnhancer
            from response_generator import ResponseGenerator
            from model_training_orchestrator import ModelTrainingOrchestrator
            
            # Initialize components
            self.learning_system = ExponentialLearningSystem()
            self.research_hub = ResearchAgentHub()
            self.knowledge_synthesizer = KnowledgeSynthesizer()
            self.validation_system = KnowledgeValidationSystem()
            self.neuro_agent = NeuroAgentEnhancer()
            self.response_generator = ResponseGenerator()
            self.model_training_orchestrator = ModelTrainingOrchestrator()
            
            # Initialize cloud orchestrator if enabled
            if self.config.get("enable_cloud_training", True):
                self.cloud_orchestrator = CloudTrainingOrchestrator()
                logger.info("☁️ Cloud training orchestrator initialized")
            
            # Initialize research hub
            if self.config.get("enable_research", True):
                await self.research_hub.initialize_all()
                logger.info("🔍 Research agents initialized")
            
            # Initialize neuro agent
            await self.neuro_agent.initialize_components()
            logger.info("🧠 Neuro agent enhanced and initialized")
            
            logger.info("✅ All system components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            raise
    
    async def start_system(self):
        """Start the exponential learning system"""
        if self.is_running:
            logger.warning("⚠️ System is already running")
            return
        
        logger.info("🚀 Starting exponential learning system...")
        
        try:
            self.is_running = True
            self.start_time = datetime.now()
            
            # Start background tasks
            background_tasks = [
                self.monitor_system_health(),
                self.update_metrics(),
                self.log_system_status()
            ]
            
            # Start background tasks
            for task in background_tasks:
                asyncio.create_task(task)
            
            logger.info("✅ Exponential learning system started successfully")
            logger.info("🎯 System is now exponentially learning and growing knowledge!")
            
            # Keep system running
            while self.is_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ System error: {e}")
            await self.stop_system()
    
    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a response to a prompt using the exponential learning system"""
        if not self.is_running:
            raise RuntimeError("System must be running to generate responses")
        
        logger.info(f"🧠 Generating response for: {prompt[:100]}...")
        
        try:
            # Research the topic
            research_results = await self.research_hub.search_all_sources(prompt)
            
            # Synthesize knowledge
            synthesized_knowledge = await self.knowledge_synthesizer.synthesize_research_findings(research_results)
            
            # Generate response
            response = await self.response_generator.generate_response(prompt, research_results, synthesized_knowledge)
            
            # If response quality is low, trigger exponential training
            if response.confidence < 0.7:
                logger.info(f"📚 Response quality low ({response.confidence:.2f}), triggering exponential training...")
                await self.trigger_exponential_training(research_results, response.confidence)
            
            return {
                "response": response.response,
                "confidence": response.confidence,
                "sources": response.sources,
                "reasoning": response.reasoning,
                "improvements": response.learning_improvements
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return {
                "response": "I'm still learning about this topic. Let me research it further.",
                "confidence": 0.1,
                "sources": [],
                "reasoning": "Error occurred during response generation",
                "improvements": [f"Fix error: {str(e)}"]
            }
    
    async def trigger_exponential_training(self, research_data: Dict[str, Any], response_quality: float):
        """Trigger exponential training using existing models"""
        try:
            logger.info("🚀 Triggering exponential training cycle...")
            
            # Start model training using existing models
            job_ids = await self.model_training_orchestrator.start_exponential_training_cycle(
                research_data, response_quality
            )
            
            logger.info(f"✅ Started {len(job_ids)} exponential training jobs: {job_ids}")
            
            # Update system metrics
            self.system_metrics["total_training_jobs"] += len(job_ids)
            
        except Exception as e:
            logger.error(f"❌ Error triggering exponential training: {e}")
    
    async def start_learning_session(self, topic: str, duration_hours: int = 24) -> str:
        """Start a new learning session"""
        if not self.is_running:
            raise RuntimeError("System must be running to start learning sessions")
        
        if len(self.active_sessions) >= self.config.get("max_concurrent_sessions", 5):
            raise RuntimeError("Maximum concurrent sessions reached")
        
        logger.info(f"📚 Starting learning session for topic: {topic}")
        
        # Create session
        session = {
            "id": f"session_{topic}_{int(time.time())}",
            "topic": topic,
            "started_at": datetime.now(),
            "duration_hours": duration_hours,
            "status": "active",
            "learning_cycles": 0,
            "knowledge_gained": 0
        }
        
        self.active_sessions.append(session)
        
        # Start session tasks
        session_tasks = []
        
        if self.config.get("enable_research", True):
            session_tasks.append(self.run_research_session(session))
        
        if self.config.get("enable_synthesis", True):
            session_tasks.append(self.run_synthesis_session(session))
        
        if self.config.get("enable_validation", True):
            session_tasks.append(self.run_validation_session(session))
        
        if self.config.get("enable_cloud_training", True) and self.cloud_orchestrator:
            session_tasks.append(self.run_training_session(session))
        
        # Start all session tasks
        for task in session_tasks:
            asyncio.create_task(task)
        
        logger.info(f"✅ Learning session {session['id']} started with {len(session_tasks)} active processes")
        return session['id']
    
    async def run_research_session(self, session: Dict[str, Any]):
        """Run research process for a session"""
        while session["status"] == "active" and self.is_running:
            try:
                # Generate research queries
                queries = self.generate_research_queries(session["topic"], session["learning_cycles"])
                
                for query in queries:
                    if not self.is_running:
                        break
                    
                    # Execute research
                    results = await self.research_hub.search_all_sources(query)
                    
                    # Update session metrics
                    session["learning_cycles"] += 1
                    session["knowledge_gained"] += len(results.get("concepts", []))
                    self.system_metrics["total_research_queries"] += 1
                    
                    logger.debug(f"🔍 Session {session['id']}: Researched '{query}', gained {len(results.get('concepts', []))} concepts")
                
                # Wait for next research cycle
                await asyncio.sleep(self.config.get("research_interval_seconds", 60))
                
            except Exception as e:
                logger.error(f"❌ Research session error for {session['id']}: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def run_synthesis_session(self, session: Dict[str, Any]):
        """Run knowledge synthesis for a session"""
        while session["status"] == "active" and self.is_running:
            try:
                # Synthesize knowledge from research
                if hasattr(self, 'research_hub') and self.research_hub:
                    # This would integrate with actual research results
                    # For now, simulate synthesis
                    synthesized_concepts = min(session["knowledge_gained"], 10)
                    
                    if synthesized_concepts > 0:
                        # Update learning system
                        if self.learning_system:
                            self.learning_system.learning_cycles += 1
                            self.system_metrics["total_learning_cycles"] += 1
                        
                        logger.debug(f"🔬 Session {session['id']}: Synthesized {synthesized_concepts} concepts")
                
                # Wait for next synthesis cycle
                await asyncio.sleep(self.config.get("synthesis_interval_seconds", 300))
                
            except Exception as e:
                logger.error(f"❌ Synthesis session error for {session['id']}: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def run_validation_session(self, session: Dict[str, Any]):
        """Run knowledge validation for a session"""
        while session["status"] == "active" and self.is_running:
            try:
                # Validate knowledge
                if hasattr(self, 'validation_system') and self.validation_system:
                    # Simulate validation
                    validation_score = min(session["knowledge_gained"] / 100.0, 1.0)
                    
                    if validation_score > 0.5:
                        logger.debug(f"✅ Session {session['id']}: Knowledge validation score: {validation_score:.2f}")
                
                # Wait for next validation cycle
                await asyncio.sleep(self.config.get("validation_interval_seconds", 600))
                
            except Exception as e:
                logger.error(f"❌ Validation session error for {session['id']}: {e}")
                await asyncio.sleep(600)  # Wait before retrying
    
    async def run_training_session(self, session: Dict[str, Any]):
        """Run training process for a session"""
        while session["status"] == "active" and self.is_running:
            try:
                # Check if enough knowledge for training
                if session["knowledge_gained"] > 100:  # Threshold
                    training_config = {
                        "script": f"train_{session['topic']}.py",
                        "hyperparameters": {
                            "learning_rate": 0.001,
                            "batch_size": 32,
                            "epochs": 100
                        },
                        "data_path": f"data/{session['topic']}",
                        "output_path": f"outputs/{session['topic']}"
                    }
                    
                    job_id = await self.cloud_orchestrator.submit_training_job(
                        f"{session['topic']}_{session['learning_cycles']}", 
                        training_config
                    )
                    
                    logger.info(f"🚀 Session {session['id']}: Submitted training job {job_id}")
                    self.system_metrics["total_training_jobs"] += 1
                    
                    # Reset knowledge counter
                    session["knowledge_gained"] = 0
                
                # Wait for next training cycle
                await asyncio.sleep(self.config.get("training_interval_seconds", 1800))
                
            except Exception as e:
                logger.error(f"❌ Training session error for {session['id']}: {e}")
                await asyncio.sleep(1800)  # Wait before retrying
    
    async def monitor_system_health(self):
        """Monitor overall system health"""
        while self.is_running:
            try:
                # Check component health
                health_status = {
                    "learning_system": self.learning_system is not None,
                    "research_hub": self.research_hub is not None,
                    "knowledge_synthesizer": self.knowledge_synthesizer is not None,
                    "validation_system": self.validation_system is not None,
                    "cloud_orchestrator": self.cloud_orchestrator is not None,
                    "neuro_agent": self.neuro_agent is not None
                }
                
                # Log health status
                healthy_components = sum(health_status.values())
                total_components = len(health_status)
                
                if healthy_components < total_components:
                    logger.warning(f"⚠️ System health: {healthy_components}/{total_components} components healthy")
                else:
                    logger.debug(f"✅ System health: All {total_components} components healthy")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Health monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def update_metrics(self):
        """Update system metrics"""
        while self.is_running:
            try:
                # Update uptime
                if self.start_time:
                    uptime = datetime.now() - self.start_time
                    self.system_metrics["system_uptime"] = uptime.total_seconds()
                
                # Update learning system metrics
                if self.learning_system:
                    self.system_metrics["total_learning_cycles"] = self.learning_system.learning_cycles
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"❌ Metrics update error: {e}")
                await asyncio.sleep(60)
    
    async def log_system_status(self):
        """Log periodic system status"""
        while self.is_running:
            try:
                active_sessions = len([s for s in self.active_sessions if s["status"] == "active"])
                
                logger.info(f"📊 System Status: {active_sessions} active sessions, "
                          f"{self.system_metrics['total_learning_cycles']} learning cycles, "
                          f"{self.system_metrics['total_knowledge_gained']} knowledge gained")
                
                await asyncio.sleep(1800)  # Log every 30 minutes
                
            except Exception as e:
                logger.error(f"❌ Status logging error: {e}")
                await asyncio.sleep(1800)
    
    def generate_research_queries(self, topic: str, cycle: int) -> List[str]:
        """Generate research queries for a topic"""
        base_queries = [
            f"{topic} fundamentals",
            f"{topic} advanced concepts",
            f"{topic} recent developments",
            f"{topic} applications",
            f"{topic} challenges and solutions"
        ]
        
        # Add cycle-specific complexity
        complex_queries = [
            f"{topic} quantum computing applications",
            f"{topic} artificial intelligence integration",
            f"{topic} neural network implementations",
            f"{topic} machine learning algorithms",
            f"{topic} deep learning frameworks"
        ]
        
        # Exponential query generation
        all_queries = base_queries + complex_queries
        query_count = min(len(all_queries), 2 ** (cycle % 5))
        
        return all_queries[:query_count]
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "active_sessions": len([s for s in self.active_sessions if s["status"] == "active"]),
            "total_sessions": len(self.active_sessions),
            "system_metrics": self.system_metrics.copy(),
            "configuration": self.config,
            "sessions": self.active_sessions
        }
        
        # Add component status
        if self.learning_system:
            status["learning_system"] = {
                "learning_cycles": self.learning_system.learning_cycles,
                "cycle_efficiency": self.learning_system.cycle_efficiency,
                "knowledge_hunger": self.learning_system.knowledge_hunger,
                "total_knowledge": len(self.learning_system.knowledge_base)
            }
        
        if self.cloud_orchestrator:
            status["cloud_training"] = self.cloud_orchestrator.get_all_jobs()
            status["costs"] = self.cloud_orchestrator.get_cost_summary()
        
        return status
    
    async def stop_learning_session(self, session_id: str):
        """Stop a specific learning session"""
        for session in self.active_sessions:
            if session["id"] == session_id and session["status"] == "active":
                session["status"] = "stopped"
                session["ended_at"] = datetime.now()
                logger.info(f"⏹️ Stopped learning session {session_id}")
                break
    
    async def stop_system(self):
        """Stop the exponential learning system"""
        if not self.is_running:
            return
        
        logger.info("🛑 Stopping exponential learning system...")
        
        try:
            self.is_running = False
            
            # Stop all active sessions
            for session in self.active_sessions:
                if session["status"] == "active":
                    await self.stop_learning_session(session["id"])
            
            # Cleanup components
            if self.research_hub:
                await self.research_hub.cleanup_all()
            
            if self.neuro_agent:
                await self.neuro_agent.cleanup()
            
            logger.info("✅ Exponential learning system stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping system: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop_system())
    
    async def cleanup(self):
        """Clean up all resources"""
        try:
            await self.stop_system()
            logger.info("🧹 System cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

async def main():
    """Test the main orchestrator"""
    orchestrator = ExponentialLearningOrchestrator()
    
    try:
        # Initialize system
        await orchestrator.initialize_system()
        
        # Start system
        system_task = asyncio.create_task(orchestrator.start_system())
        
        # Wait a moment for system to start
        await asyncio.sleep(2)
        
        # Start a learning session
        session_id = await orchestrator.start_learning_session("quantum_computing", duration_hours=1)
        print(f"🚀 Started learning session: {session_id}")
        
        # Let it run for a bit
        await asyncio.sleep(30)
        
        # Get system status
        status = await orchestrator.get_system_status()
        print(f"📊 System status: {status['active_sessions']} active sessions")
        
        # Stop session
        await orchestrator.stop_learning_session(session_id)
        
        # Stop system
        await orchestrator.stop_system()
        
    except Exception as e:
        logger.error(f"❌ Main error: {e}")
        await orchestrator.cleanup()
    
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
