#!/usr/bin/env python3
"""
Main Entry Point for Exponential Learning System
Run this script to start the exponential learning system
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path

# Add the current directory to the Python path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from main_orchestrator import ExponentialLearningOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('exponential_learning.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main function to run the exponential learning system"""
    parser = argparse.ArgumentParser(description='Exponential Learning System')
    parser.add_argument('--config', '-c', type=str, default='exponential_learning_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--topic', '-t', type=str, default='artificial_intelligence',
                       help='Initial learning topic')
    parser.add_argument('--duration', '-d', type=int, default=24,
                       help='Learning session duration in hours')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (shorter duration)')
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting Exponential Learning System...")
        logger.info(f"📚 Initial topic: {args.topic}")
        logger.info(f"⏱️  Session duration: {args.duration} hours")
        
        # Initialize orchestrator
        orchestrator = ExponentialLearningOrchestrator(config_path=args.config)
        
        # Initialize system
        await orchestrator.initialize_system()
        
        if args.test:
            # Test mode - run for shorter duration
            logger.info("🧪 Running in TEST mode")
            args.duration = 1  # 1 hour for testing
        
        # Start system
        system_task = asyncio.create_task(orchestrator.start_system())
        
        # Wait for system to start
        await asyncio.sleep(2)
        
        # Start initial learning session
        session_id = await orchestrator.start_learning_session(args.topic, args.duration)
        logger.info(f"✅ Started learning session: {session_id}")
        
        if args.interactive:
            # Interactive mode
            await run_interactive_mode(orchestrator)
        else:
            # Non-interactive mode - let it run
            logger.info("🎯 System running in background mode")
            logger.info("📊 Use Ctrl+C to stop the system")
            
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(60)  # Check every minute
                    
                    # Log status periodically
                    status = await orchestrator.get_system_status()
                    if status['active_sessions'] > 0:
                        logger.info(f"📊 Status: {status['active_sessions']} active sessions, "
                                  f"{status['system_metrics']['total_learning_cycles']} learning cycles")
                    
            except KeyboardInterrupt:
                logger.info("📡 Received interrupt signal")
        
        # Stop system
        await orchestrator.stop_system()
        logger.info("✅ Exponential Learning System stopped successfully")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        if 'orchestrator' in locals():
            await orchestrator.cleanup()

async def run_interactive_mode(orchestrator):
    """Run the system in interactive mode"""
    logger.info("🎮 Interactive mode enabled")
    logger.info("Commands: start <topic>, stop <session_id>, status, quit")
    
    while True:
        try:
            command = input("\n🎯 Command: ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                break
            elif command.startswith('start '):
                topic = command[6:].strip()
                if topic:
                    session_id = await orchestrator.start_learning_session(topic, 24)
                    print(f"🚀 Started session: {session_id}")
                else:
                    print("❌ Please specify a topic")
            elif command.startswith('stop '):
                session_id = command[5:].strip()
                if session_id:
                    await orchestrator.stop_learning_session(session_id)
                    print(f"⏹️ Stopped session: {session_id}")
                else:
                    print("❌ Please specify a session ID")
            elif command == 'status':
                status = await orchestrator.get_system_status()
                print(f"📊 System Status:")
                print(f"  Running: {status['is_running']}")
                print(f"  Active Sessions: {status['active_sessions']}")
                print(f"  Learning Cycles: {status['system_metrics']['total_learning_cycles']}")
                print(f"  Knowledge Gained: {status['system_metrics']['total_knowledge_gained']}")
                
                if status['sessions']:
                    print(f"\n📚 Active Sessions:")
                    for session in status['sessions']:
                        if session['status'] == 'active':
                            print(f"  - {session['id']}: {session['topic']} "
                                  f"(cycles: {session['learning_cycles']}, "
                                  f"knowledge: {session['knowledge_gained']})")
            elif command == 'help':
                print("🎮 Available Commands:")
                print("  start <topic>     - Start a new learning session")
                print("  stop <session_id> - Stop a specific session")
                print("  status            - Show system status")
                print("  quit/exit         - Exit interactive mode")
            else:
                print("❌ Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n📡 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def print_banner():
    """Print the system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    EXPONENTIAL LEARNING SYSTEM               ║
    ║                                                              ║
    ║  🚀 Perpetual Knowledge Acquisition                          ║
    ║  🔍 Multi-Source Research                                   ║
    ║  🔬 Intelligent Synthesis                                   ║
    ║  ☁️  Cloud Training Orchestration                            ║
    ║  ✅ Knowledge Validation                                     ║
    ║  🧠 Enhanced Neuro Agent                                    ║
    ║                                                              ║
    ║  Never satisfied, always learning, exponentially growing!   ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

if __name__ == "__main__":
    print_banner()
    asyncio.run(main())
