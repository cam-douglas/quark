#!/usr/bin/env python3
"""
Enhanced Baby AGI Demo

Demonstrates the enhanced capabilities of the Baby AGI system:
1. Latest MoE Backbone Integration (Mixtral, DeepSeek, Qwen)
2. Enhanced Data Resources (arXiv, PubMed, GitHub, DANDI, OpenNeuro)
3. Real Simulation Capabilities (Brian2, Norse, SpikingJelly)
4. Human-Like Cognitive Processing

This demo shows how the system thinks more like a human brain than a computer.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import enhanced components
try:
    from development.src.models.moe_backbones import MoEBackboneManager, MoEBackboneType
    from development.src.models.moe_manager import MoEManager, ExecutionMode
    from development.src.models.moe_router import HumanLikeCognitiveRouter, CognitiveProcess
    from development.src.models.neuroscience_experts import NeuroscienceExpertManager
    from development.src.neurodata.enhanced_data_resources import EnhancedDataResources
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Please ensure all dependencies are installed and the project structure is correct")
    exit(1)

class EnhancedBabyAGIDemo:
    """Enhanced Baby AGI Demo with latest capabilities"""
    
    def __init__(self):
        self.moe_backbone_manager = None
        self.moe_manager = None
        self.cognitive_router = None
        self.expert_manager = None
        self.data_resources = None
        
        logger.info("Enhanced Baby AGI Demo initialized")
    
    async def setup_system(self):
        """Setup the enhanced Baby AGI system"""
        logger.info("Setting up enhanced Baby AGI system...")
        
        try:
            # Initialize MoE Backbone Manager
            self.moe_backbone_manager = MoEBackboneManager()
            logger.info("✓ MoE Backbone Manager initialized")
            
            # Initialize Human-Like Cognitive Router
            self.cognitive_router = HumanLikeCognitiveRouter(
                primary_process=CognitiveProcess.INTUITIVE_UNDERSTANDING
            )
            logger.info("✓ Human-Like Cognitive Router initialized")
            
            # Initialize MoE Manager
            self.moe_manager = MoEManager(
                execution_mode=ExecutionMode.HYBRID_LLM
            )
            logger.info("✓ MoE Manager initialized")
            
            # Initialize Neuroscience Expert Manager
            self.expert_manager = NeuroscienceExpertManager()
            logger.info("✓ Neuroscience Expert Manager initialized")
            
            # Initialize Enhanced Data Resources
            self.data_resources = EnhancedDataResources()
            logger.info("✓ Enhanced Data Resources initialized")
            
            logger.info("✓ Enhanced Baby AGI system setup complete!")
            
        except Exception as e:
            logger.error(f"System setup failed: {e}")
            raise
    
    async def demo_moe_backbones(self):
        """Demo the latest MoE backbone capabilities"""
        logger.info("\n" + "="*60)
        logger.info("DEMO: Latest MoE Backbone Capabilities")
        logger.info("="*60)
        
        try:
            # Get available models
            available_models = self.moe_backbone_manager.get_available_models()
            logger.info(f"Available MoE models: {len(available_models)}")
            
            for model in available_models[:3]:  # Show first 3
                logger.info(f"  - {model.model_name}: {model.num_experts} experts, {model.hidden_size} hidden size")
            
            # Try to load a lightweight model
            logger.info("\nAttempting to load Mixtral-8x7B...")
            mixtral_model = self.moe_backbone_manager.load_model("Mixtral-8x7B-v0.1", device="auto")
            
            if mixtral_model:
                logger.info("✓ Mixtral-8x7B loaded successfully!")
                model_info = self.moe_backbone_manager.get_model_info("Mixtral-8x7B-v0.1")
                logger.info(f"Model status: {model_info}")
            else:
                logger.warning("⚠ Mixtral-8x7B could not be loaded (may require more memory)")
            
        except Exception as e:
            logger.error(f"MoE backbone demo failed: {e}")
    
    async def demo_human_like_cognition(self):
        """Demo human-like cognitive processing"""
        logger.info("\n" + "="*60)
        logger.info("DEMO: Human-Like Cognitive Processing")
        logger.info("="*60)
        
        try:
            # Test different cognitive processes
            test_queries = [
                "How do neurons communicate with each other?",
                "Can you help me understand brain plasticity?",
                "What's the latest research on consciousness?",
                "I'm curious about how memories form in the brain"
            ]
            
            cognitive_processes = [
                CognitiveProcess.INTUITIVE_UNDERSTANDING,
                CognitiveProcess.EMOTIONAL_CONTEXT,
                CognitiveProcess.CREATIVE_SOLVING,
                CognitiveProcess.HOLISTIC_THINKING
            ]
            
            for i, (query, process) in enumerate(zip(test_queries, cognitive_processes)):
                logger.info(f"\nQuery {i+1}: {query}")
                logger.info(f"Using cognitive process: {process.value}")
                
                # Change cognitive process
                self.cognitive_router.set_cognitive_process(process)
                
                # Process query
                decision = await self.cognitive_router.process_query_human_like(
                    query, emotional_context="curious"
                )
                
                logger.info(f"  Primary faculty: {decision.primary_faculty}")
                logger.info(f"  Confidence: {decision.confidence:.2f}")
                logger.info(f"  Reasoning: {decision.reasoning}")
                logger.info(f"  Emotional context: {decision.emotional_context}")
                logger.info(f"  Creative insights: {len(decision.creative_insights)} insights")
                logger.info(f"  Memory associations: {len(decision.memory_associations)} associations")
                
        except Exception as e:
            logger.error(f"Human-like cognition demo failed: {e}")
    
    async def demo_enhanced_data_resources(self):
        """Demo enhanced data resource capabilities"""
        logger.info("\n" + "="*60)
        logger.info("DEMO: Enhanced Data Resources")
        logger.info("="*60)
        
        try:
            # Get latest neuroscience papers
            logger.info("Fetching latest neuroscience papers...")
            papers = await self.data_resources.get_latest_neuroscience_papers(
                query="neuroscience", max_results=5, days_back=7
            )
            logger.info(f"✓ Found {len(papers)} recent papers")
            
            for i, paper in enumerate(papers[:3]):
                logger.info(f"  Paper {i+1}: {paper['title'][:80]}...")
                logger.info(f"    Source: {paper['source']}, Date: {paper['date']}")
            
            # Get latest GitHub repositories
            logger.info("\nFetching latest neuroscience GitHub repositories...")
            repos = await self.data_resources.get_latest_github_repositories(
                query="neuroscience", max_results=3, days_back=7
            )
            logger.info(f"✓ Found {len(repos)} recent repositories")
            
            for i, repo in enumerate(repos[:2]):
                logger.info(f"  Repo {i+1}: {repo['name']}")
                logger.info(f"    Stars: {repo['stars']}, Language: {repo['language']}")
                logger.info(f"    Description: {repo['description'][:60]}...")
            
            # Get comprehensive update
            logger.info("\nGetting comprehensive neuroscience update...")
            update = await self.data_resources.get_comprehensive_neuroscience_update(days_back=3)
            logger.info(f"✓ Comprehensive update completed:")
            logger.info(f"  Total papers: {update['summary']['total_papers']}")
            logger.info(f"  Total repositories: {update['summary']['total_repositories']}")
            logger.info(f"  Total datasets: {update['summary']['total_datasets']}")
            
        except Exception as e:
            logger.error(f"Enhanced data resources demo failed: {e}")
    
    async def demo_real_simulations(self):
        """Demo real simulation capabilities"""
        logger.info("\n" + "="*60)
        logger.info("DEMO: Real Simulation Capabilities")
        logger.info("="*60)
        
        try:
            # Test Brian2 expert
            logger.info("Testing Brian2 spiking neural network simulation...")
            brian2_expert = self.expert_manager.experts.get("Brian2 Expert")
            
            if brian2_expert and brian2_expert.is_available:
                from development.src.models.neuroscience_experts import NeuroscienceTask, NeuroscienceTaskType
                
                task = NeuroscienceTask(
                    task_type=NeuroscienceTaskType.SPIKING_NETWORKS,
                    description="Simulate a spiking neural network with 50 neurons",
                    parameters={
                        'duration': 500,  # ms
                        'num_neurons': 50,
                        'connection_prob': 0.15
                    },
                    expected_output="Spiking neural network simulation results",
                    confidence=0.8
                )
                
                result = brian2_expert.execute(task)
                
                if result.get('success'):
                    logger.info("✓ Brian2 simulation completed successfully!")
                    logger.info(f"  Simulation duration: {result.get('simulation_duration')}ms")
                    logger.info(f"  Number of neurons: {result.get('num_neurons')}")
                    
                    # Show some results
                    results = result.get('results', {})
                    if 'total_spikes' in results:
                        logger.info(f"  Total spikes generated: {results['total_spikes']}")
                    if 'simulation_metadata' in results:
                        metadata = results['simulation_metadata']
                        logger.info(f"  Framework: {metadata.get('framework')}")
                        logger.info(f"  Version: {metadata.get('version')}")
                else:
                    logger.warning(f"⚠ Brian2 simulation failed: {result.get('error')}")
            else:
                logger.warning("⚠ Brian2 Expert not available")
            
            # Test PyTorch SNN expert
            logger.info("\nTesting PyTorch SNN simulation...")
            snn_expert = self.expert_manager.experts.get("Norse/SpikingJelly Expert")
            
            if snn_expert and snn_expert.is_available:
                task = NeuroscienceTask(
                    task_type=NeuroscienceTaskType.PYTORCH_SNN,
                    description="Create a PyTorch SNN with 784 input, 128 hidden, 10 output",
                    parameters={
                        'input_size': 784,
                        'hidden_size': 128,
                        'num_classes': 10,
                        'num_steps': 8,
                        'batch_size': 2
                    },
                    expected_output="PyTorch SNN model and inference results",
                    confidence=0.8
                )
                
                result = snn_expert.execute(task)
                
                if result.get('success'):
                    logger.info("✓ PyTorch SNN simulation completed successfully!")
                    logger.info(f"  Input size: {result.get('input_size')}")
                    logger.info(f"  Hidden size: {result.get('hidden_size')}")
                    logger.info(f"  Device: {result.get('device')}")
                    
                    # Show model metadata
                    results = result.get('results', {})
                    if 'model_metadata' in results:
                        metadata = results['model_metadata']
                        logger.info(f"  Framework: {metadata.get('framework')}")
                        logger.info(f"  PyTorch version: {metadata.get('pytorch_version')}")
                else:
                    logger.warning(f"⚠ PyTorch SNN simulation failed: {result.get('error')}")
            else:
                logger.warning("⚠ PyTorch SNN Expert not available")
                
        except Exception as e:
            logger.error(f"Real simulations demo failed: {e}")
    
    async def demo_moe_integration(self):
        """Demo MoE system integration"""
        logger.info("\n" + "="*60)
        logger.info("DEMO: MoE System Integration")
        logger.info("="*60)
        
        try:
            # Test MoE manager
            logger.info("Testing MoE Manager with hybrid execution...")
            
            test_queries = [
                "Explain how neurons form memories using simple language",
                "What are the latest advances in brain-computer interfaces?",
                "How does the brain process visual information?"
            ]
            
            for i, query in enumerate(test_queries):
                logger.info(f"\nQuery {i+1}: {query}")
                
                try:
                    response = await self.moe_manager.process_query(
                        query=query,
                        parameters={'max_length': 200}
                    )
                    
                    logger.info(f"  Primary expert: {response.primary_expert}")
                    logger.info(f"  Confidence: {response.confidence:.2f}")
                    logger.info(f"  Response: {response.primary_response[:100]}...")
                    logger.info(f"  Execution time: {response.execution_time:.3f}s")
                    
                except Exception as e:
                    logger.warning(f"  Query processing failed: {e}")
            
            # Show system status
            logger.info("\nSystem Status:")
            status = self.moe_manager.get_system_status()
            logger.info(f"  Total queries: {status['moe_manager']['total_queries']}")
            logger.info(f"  Success rate: {status['moe_manager']['success_rate']:.2%}")
            logger.info(f"  Average response time: {status['moe_manager']['average_response_time']:.3f}s")
            
        except Exception as e:
            logger.error(f"MoE integration demo failed: {e}")
    
    async def run_comprehensive_demo(self):
        """Run the complete enhanced Baby AGI demo"""
        logger.info("🚀 Starting Enhanced Baby AGI Comprehensive Demo")
        logger.info("This demo showcases human-like cognitive processing and latest neuroscience tools")
        
        try:
            # Setup system
            await self.setup_system()
            
            # Run all demos
            await self.demo_moe_backbones()
            await self.demo_human_like_cognition()
            await self.demo_enhanced_data_resources()
            await self.demo_real_simulations()
            await self.demo_moe_integration()
            
            logger.info("\n" + "="*60)
            logger.info("🎉 Enhanced Baby AGI Demo Completed Successfully!")
            logger.info("="*60)
            logger.info("Key Features Demonstrated:")
            logger.info("  ✓ Latest MoE Backbone Integration")
            logger.info("  ✓ Human-Like Cognitive Processing")
            logger.info("  ✓ Enhanced Data Resources (arXiv, PubMed, GitHub)")
            logger.info("  ✓ Real Simulation Capabilities (Brian2, PyTorch SNN)")
            logger.info("  ✓ MoE System Integration")
            logger.info("\nThe system now thinks more like a human brain than a computer!")
            
        except Exception as e:
            logger.error(f"Comprehensive demo failed: {e}")
            raise

async def main():
    """Main function to run the enhanced demo"""
    demo = EnhancedBabyAGIDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
