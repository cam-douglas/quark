#!/usr/bin/env python3
"""
Combine All Models and Agents
Unified training system for SmallMind
"""

import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedSmallMind:
    """Combines all models and agents into one system"""
    
    def __init__(self):
        self.models = {}
        self.agents = {}
        self.unified_knowledge = {}
        self.training_cycles = 0
        
    async def discover_models(self):
        """Find all available models"""
        logger.info("🔍 Discovering models...")
        
        model_paths = {
            "deepseek": "models/checkpoints/deepseek-v2",
            "mixtral": "models/checkpoints/mix-tao-moe",
            "qwen": "models/checkpoints/qwen1.5-moe"
        }
        
        for name, path in model_paths.items():
            if Path(path).exists():
                self.models[name] = {
                    "path": path,
                    "type": "llm",
                    "performance": 0.8,
                    "capabilities": ["text", "reasoning", "learning"]
                }
                logger.info(f"✅ Found model: {name}")
        
        logger.info(f"📊 Total models: {len(self.models)}")
    
    async def initialize_agents(self):
        """Initialize all agents"""
        logger.info("🚀 Initializing agents...")
        
        try:
            # Import all agent components
            from exponential_learning_system import ExponentialLearningSystem
            from research_agents import ResearchAgentHub
            from knowledge_synthesizer import KnowledgeSynthesizer
            from response_generator import ResponseGenerator
            from knowledge_validation_system import KnowledgeValidationSystem
            from model_training_orchestrator import ModelTrainingOrchestrator
            
            # Initialize agents
            self.agents["learning"] = ExponentialLearningSystem()
            self.agents["research"] = ResearchAgentHub()
            self.agents["synthesis"] = KnowledgeSynthesizer()
            self.agents["response"] = ResponseGenerator()
            self.agents["validation"] = KnowledgeValidationSystem()
            self.agents["training"] = ModelTrainingOrchestrator()
            
            logger.info(f"✅ Total agents: {len(self.agents)}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing agents: {e}")
    
    async def unified_training_cycle(self, input_data: str):
        """Execute unified training across all components"""
        logger.info(f"🔄 Unified training cycle {self.training_cycles + 1}")
        
        try:
            # Research phase
            research_results = await self.agents["research"].search_all_sources(input_data)
            
            # Synthesis phase
            knowledge = await self.agents["synthesis"].synthesize_research_findings(research_results)
            
            # Response generation
            response = await self.agents["response"].generate_response(input_data, research_results, knowledge)
            
            # Validation
            validation = await self.agents["validation"].validate_knowledge(knowledge)
            
            # Training trigger
            if response.confidence < 0.7:
                await self.trigger_unified_training(research_results, response.confidence)
            
            # Update knowledge
            self.unified_knowledge[input_data] = {
                "response": response.response,
                "confidence": response.confidence,
                "knowledge": knowledge,
                "validation": validation,
                "timestamp": datetime.now().isoformat()
            }
            
            self.training_cycles += 1
            
            return {
                "response": response.response,
                "confidence": response.confidence,
                "cycle": self.training_cycles
            }
            
        except Exception as e:
            logger.error(f"❌ Training cycle failed: {e}")
            return {"error": str(e)}
    
    async def trigger_unified_training(self, research_data: dict, confidence: float):
        """Trigger training across all models"""
        logger.info("🚀 Triggering unified training...")
        
        try:
            # Train each model
            for model_name, model_info in self.models.items():
                logger.info(f"🔬 Training model: {model_name}")
                
                # Create training data
                training_data = {
                    "research": research_data,
                    "target_confidence": 0.9,
                    "model_capabilities": model_info["capabilities"]
                }
                
                # Execute training
                await self.agents["training"].start_exponential_training_cycle(
                    training_data, confidence
                )
                
        except Exception as e:
            logger.error(f"❌ Unified training failed: {e}")
    
    def get_system_status(self):
        """Get unified system status"""
        return {
            "models": len(self.models),
            "agents": len(self.agents),
            "knowledge_entries": len(self.unified_knowledge),
            "training_cycles": self.training_cycles,
            "model_names": list(self.models.keys()),
            "agent_names": list(self.agents.keys())
        }

async def main():
    """Test unified system"""
    print("🚀 Initializing Unified SmallMind...")
    
    system = UnifiedSmallMind()
    
    # Discover models
    await system.discover_models()
    
    # Initialize agents
    await system.initialize_agents()
    
    # Test unified training
    test_query = "What is artificial intelligence?"
    
    print(f"\n🧪 Testing unified training with: {test_query}")
    
    result = await system.unified_training_cycle(test_query)
    
    if "error" not in result:
        print(f"✅ Response: {result['response'][:100]}...")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print(f"🔄 Cycle: {result['cycle']}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Show status
    status = system.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Models: {status['models']}")
    print(f"   Agents: {status['agents']}")
    print(f"   Knowledge: {status['knowledge_entries']}")
    print(f"   Training Cycles: {status['training_cycles']}")

if __name__ == "__main__":
    asyncio.run(main())
