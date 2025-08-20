#!/usr/bin/env python3
"""
Chat Interface for Exponential Learning AI System
Test the AI's ability to respond to prompts and learn exponentially
"""

import asyncio
import logging
import sys
from pathlib import Path

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

class AIChatInterface:
    """Simple chat interface for the exponential learning AI"""
    
    def __init__(self):
        self.orchestrator = None
        self.is_running = False
    
    async def initialize(self):
        """Initialize the AI system"""
        try:
            logger.info("🚀 Initializing AI chat system...")
            
            # Create orchestrator
            self.orchestrator = ExponentialLearningOrchestrator()
            
            # Initialize system
            await self.orchestrator.initialize_system()
            
            # Start system
            await self.orchestrator.start_system()
            
            self.is_running = True
            logger.info("✅ AI chat system ready!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI chat system: {e}")
            raise
    
    async def chat_loop(self):
        """Main chat loop"""
        print("\n🤖 Welcome to the Exponential Learning AI!")
        print("I'm constantly learning and improving. Ask me anything!")
        print("Type 'quit' to exit, 'stats' for system status\n")
        
        while self.is_running:
            try:
                # Get user input
                user_input = input("🧑 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🤖 Goodbye! I'll keep learning while you're away.")
                    break
                
                elif user_input.lower() == 'stats':
                    await self.show_system_stats()
                    continue
                
                elif not user_input:
                    continue
                
                # Generate AI response
                print("🤖 AI: Thinking...")
                response = await self.orchestrator.generate_response(user_input)
                
                # Display response
                print(f"\n🤖 AI: {response['response']}")
                
                # Show confidence and improvements
                print(f"\n📊 Confidence: {response['confidence']:.2f}")
                if response['sources']:
                    print(f"📚 Sources: {', '.join(response['sources'])}")
                if response['improvements']:
                    print(f"🔧 Learning improvements: {', '.join(response['improvements'])}")
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\n\n🤖 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error in chat loop: {e}")
                print(f"🤖 Sorry, I encountered an error: {e}")
                print("Let me try to recover...\n")
    
    async def show_system_stats(self):
        """Show system statistics"""
        try:
            stats = self.orchestrator.get_system_status()
            
            print("\n📊 System Statistics:")
            print(f"   Status: {'Running' if stats['status'] == 'running' else 'Stopped'}")
            print(f"   Uptime: {stats.get('uptime', 'Unknown')}")
            print(f"   Active sessions: {stats.get('active_sessions', 0)}")
            print(f"   Total learning cycles: {stats.get('total_learning_cycles', 0)}")
            print(f"   Knowledge base size: {stats.get('knowledge_base_size', 0)}")
            print(f"   Training jobs: {stats.get('total_training_jobs', 0)}")
            
            # Show response generator stats if available
            if hasattr(self.orchestrator, 'response_generator'):
                response_stats = self.orchestrator.response_generator.get_learning_stats()
                print(f"   Responses generated: {response_stats.get('total_responses', 0)}")
                print(f"   Average confidence: {response_stats.get('average_confidence', 0):.2f}")
            
            print()  # Empty line
            
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            print(f"🤖 Error getting stats: {e}\n")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.orchestrator and self.is_running:
                await self.orchestrator.stop_system()
                logger.info("✅ AI chat system stopped")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

async def main():
    """Main function"""
    chat_interface = AIChatInterface()
    
    try:
        # Initialize the system
        await chat_interface.initialize()
        
        # Start chat loop
        await chat_interface.chat_loop()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        print(f"🤖 Fatal error: {e}")
        print("Please check the logs for more details.")
        
    finally:
        # Cleanup
        await chat_interface.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🤖 Chat interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print(f"🤖 Unexpected error: {e}")
        sys.exit(1)
