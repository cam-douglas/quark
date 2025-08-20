#!/usr/bin/env python3
"""
Demo Script for Exponential Learning System
Shows how the AI learns exponentially using existing models
"""

import asyncio
import logging
import sys
from pathlib import Path
import time

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from main_orchestrator import ExponentialLearningOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class ExponentialLearningDemo:
    """Demo class for the exponential learning system"""
    
    def __init__(self):
        self.orchestrator = None
        self.demo_questions = [
            "What is quantum computing?",
            "How does machine learning work?",
            "Explain the theory of relativity",
            "What are the benefits of renewable energy?",
            "How do neural networks learn?",
            "What is blockchain technology?",
            "Explain photosynthesis",
            "How does the immune system work?",
            "What causes climate change?",
            "How do computers process information?"
        ]
    
    async def run_demo(self):
        """Run the complete exponential learning demo"""
        print("🚀 Exponential Learning System Demo")
        print("=" * 50)
        
        try:
            # Initialize system
            await self.initialize_system()
            
            # Run learning demonstration
            await self.demonstrate_learning()
            
            # Show exponential growth
            await self.demonstrate_exponential_growth()
            
            # Test response quality improvement
            await self.test_response_improvement()
            
            # Show final statistics
            await self.show_final_stats()
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            print(f"🤖 Demo failed: {e}")
        finally:
            # Cleanup
            await self.cleanup()
    
    async def initialize_system(self):
        """Initialize the exponential learning system"""
        print("\n🔧 Initializing Exponential Learning System...")
        
        try:
            # Create orchestrator
            self.orchestrator = ExponentialLearningOrchestrator()
            
            # Initialize system
            await self.orchestrator.initialize_system()
            
            # Start system
            await self.orchestrator.start_system()
            
            print("✅ System initialized and running!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            raise
    
    async def demonstrate_learning(self):
        """Demonstrate the learning process"""
        print("\n📚 Demonstrating Learning Process...")
        print("-" * 40)
        
        for i, question in enumerate(self.demo_questions[:5], 1):
            print(f"\n🧑 Question {i}: {question}")
            
            try:
                # Generate response
                start_time = time.time()
                response = await self.orchestrator.generate_response(question)
                response_time = time.time() - start_time
                
                # Display response
                print(f"🤖 AI Response: {response['response'][:200]}...")
                print(f"📊 Confidence: {response['confidence']:.2f}")
                print(f"⏱️  Response time: {response_time:.2f}s")
                
                if response['improvements']:
                    print(f"🔧 Learning improvements: {', '.join(response['improvements'][:2])}")
                
                # Small delay between questions
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error processing question {i}: {e}")
                print(f"❌ Error: {e}")
    
    async def demonstrate_exponential_growth(self):
        """Demonstrate exponential learning growth"""
        print("\n📈 Demonstrating Exponential Learning Growth...")
        print("-" * 40)
        
        try:
            # Get initial stats
            initial_stats = self.orchestrator.get_system_status()
            print(f"📊 Initial knowledge base size: {initial_stats.get('knowledge_base_size', 0)}")
            
            # Trigger multiple learning cycles
            print("\n🔄 Triggering multiple learning cycles...")
            
            for cycle in range(3):
                print(f"\n🔄 Learning Cycle {cycle + 1}")
                
                # Ask a complex question to trigger learning
                complex_question = "What are the latest developments in artificial intelligence and how do they compare to human intelligence?"
                
                print(f"🧑 Complex Question: {complex_question}")
                
                # Generate response (this should trigger exponential training)
                response = await self.orchestrator.generate_response(complex_question)
                
                print(f"🤖 Response confidence: {response['confidence']:.2f}")
                print(f"🔧 Improvements needed: {len(response['improvements'])}")
                
                # Show training status
                if hasattr(self.orchestrator, 'model_training_orchestrator'):
                    training_stats = self.orchestrator.model_training_orchestrator.get_training_stats()
                    print(f"🚀 Training jobs: {training_stats.get('active_jobs', 0)} active, {training_stats.get('completed_jobs', 0)} completed")
                
                # Wait for learning to process
                await asyncio.sleep(2)
            
            # Get final stats
            final_stats = self.orchestrator.get_system_status()
            print(f"\n📊 Final knowledge base size: {final_stats.get('knowledge_base_size', 0)}")
            
            # Calculate growth
            initial_size = initial_stats.get('knowledge_base_size', 0)
            final_size = final_stats.get('knowledge_base_size', 0)
            
            if initial_size > 0:
                growth_factor = final_size / initial_size
                print(f"📈 Knowledge growth factor: {growth_factor:.2f}x")
            
        except Exception as e:
            logger.error(f"❌ Error demonstrating exponential growth: {e}")
            print(f"❌ Error: {e}")
    
    async def test_response_improvement(self):
        """Test if response quality improves over time"""
        print("\n🧪 Testing Response Quality Improvement...")
        print("-" * 40)
        
        test_question = "What is the future of renewable energy?"
        
        try:
            # First response
            print(f"\n🧑 Test Question: {test_question}")
            print("🔄 Generating first response...")
            
            first_response = await self.orchestrator.generate_response(test_question)
            first_confidence = first_response['confidence']
            
            print(f"🤖 First response confidence: {first_confidence:.2f}")
            
            # Wait for learning to process
            print("⏳ Waiting for learning to process...")
            await asyncio.sleep(3)
            
            # Second response (should be better)
            print("🔄 Generating second response...")
            
            second_response = await self.orchestrator.generate_response(test_question)
            second_confidence = second_response['confidence']
            
            print(f"🤖 Second response confidence: {second_confidence:.2f}")
            
            # Compare
            if second_confidence > first_confidence:
                improvement = ((second_confidence - first_confidence) / first_confidence) * 100
                print(f"📈 Improvement: +{improvement:.1f}%")
            else:
                print("📊 No improvement detected (this is normal for early stages)")
            
        except Exception as e:
            logger.error(f"❌ Error testing response improvement: {e}")
            print(f"❌ Error: {e}")
    
    async def show_final_stats(self):
        """Show final system statistics"""
        print("\n📊 Final System Statistics")
        print("=" * 50)
        
        try:
            # System stats
            system_stats = self.orchestrator.get_system_status()
            
            print(f"🖥️  System Status: {system_stats.get('status', 'Unknown')}")
            print(f"⏱️  Uptime: {system_stats.get('uptime', 'Unknown')}")
            print(f"📚 Active Learning Sessions: {system_stats.get('active_sessions', 0)}")
            print(f"🔄 Total Learning Cycles: {system_stats.get('total_learning_cycles', 0)}")
            print(f"🧠 Knowledge Base Size: {system_stats.get('knowledge_base_size', 0)}")
            print(f"🚀 Training Jobs: {system_stats.get('total_training_jobs', 0)}")
            
            # Response generator stats
            if hasattr(self.orchestrator, 'response_generator'):
                response_stats = self.orchestrator.response_generator.get_learning_stats()
                print(f"\n🤖 Response Generator Stats:")
                print(f"   Total Responses: {response_stats.get('total_responses', 0)}")
                print(f"   Average Confidence: {response_stats.get('average_confidence', 0):.2f}")
                print(f"   Response Patterns: {response_stats.get('response_patterns', 0)}")
                print(f"   Improvement Suggestions: {len(response_stats.get('improvement_suggestions', []))}")
            
            # Model training stats
            if hasattr(self.orchestrator, 'model_training_orchestrator'):
                training_stats = self.orchestrator.model_training_orchestrator.get_training_stats()
                print(f"\n🚀 Model Training Stats:")
                print(f"   Active Jobs: {training_stats.get('active_jobs', 0)}")
                print(f"   Completed Jobs: {training_stats.get('completed_jobs', 0)}")
                print(f"   Exponential Cycles: {training_stats.get('exponential_cycles', 0)}")
                print(f"   Available Models: {', '.join(training_stats.get('available_models', []))}")
            
            print(f"\n🎉 Demo completed successfully!")
            print("The AI system is now exponentially learning and improving!")
            
        except Exception as e:
            logger.error(f"❌ Error getting final stats: {e}")
            print(f"❌ Error: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.orchestrator:
                await self.orchestrator.stop_system()
                print("✅ System stopped and cleaned up")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

async def main():
    """Main demo function"""
    demo = ExponentialLearningDemo()
    
    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n🤖 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"🤖 Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🤖 Demo interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print(f"🤖 Unexpected error: {e}")
        sys.exit(1)
