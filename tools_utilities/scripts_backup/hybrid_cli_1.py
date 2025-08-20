#!/usr/bin/env python3
"""
Hybrid CLI for MoE + Neuroscience Expert System
Provides access to both systems with intelligent fallback
"""

import asyncio
import argparse
import json
import sys
from typing import Optional, Dict, Any
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.hybrid_moe_manager import (
    HybridMoEManager, HybridExecutionMode, create_hybrid_manager, quick_hybrid_query
)
from src.models.moe_router import RoutingStrategy
from src.models.neuroscience_experts import NeuroscienceTaskType

class HybridCLI:
    """Interactive CLI for the Hybrid MoE + Neuroscience Expert System"""
    
    def __init__(self):
        self.manager: Optional[HybridMoEManager] = None
        self.interactive_mode = False
        
    async def initialize(self, 
                        execution_mode: str = "moe_routing", 
                        enable_moe: bool = True):
        """Initialize the hybrid manager"""
        try:
            # Parse execution mode
            mode = HybridExecutionMode(execution_mode)
            
            print(f"🔧 Initializing Hybrid MoE + Neuroscience System...")
            print(f"   Execution Mode: {mode.value}")
            print(f"   MoE Enabled: {enable_moe}")
            
            self.manager = create_hybrid_manager(mode, enable_moe)
            
            # Check system health
            health = await self.manager.health_check()
            if health["status"] == "healthy":
                print(f"✅ System initialized successfully!")
                print(f"   Available experts: {len(self.manager.get_available_experts())}")
                print(f"   MoE status: {'enabled' if self.manager.enable_moe else 'disabled'}")
            else:
                print(f"⚠️  System initialized with warnings: {health['status']}")
                
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            sys.exit(1)
    
    async def process_single_query(self, 
                                 query: str, 
                                 task_type: Optional[str] = None,
                                 force_neuroscience: bool = False) -> None:
        """Process a single query"""
        if not self.manager:
            print("❌ System not initialized")
            return
        
        try:
            print(f"\n🔍 Processing query: {query}")
            if task_type:
                print(f"   Task type: {task_type}")
            if force_neuroscience:
                print(f"   Force neuroscience mode: enabled")
            
            # Parse task type if provided
            parsed_task_type = None
            if task_type:
                try:
                    parsed_task_type = NeuroscienceTaskType(task_type)
                except ValueError:
                    print(f"⚠️  Unknown task type '{task_type}', using auto-detection")
            
            # Process query
            response = await self.manager.process_query(
                query, parsed_task_type, force_neuroscience=force_neuroscience
            )
            
            # Display results
            self._display_response(response)
            
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
    
    def _display_response(self, response) -> None:
        """Display the response in a formatted way"""
        print(f"\n{'='*60}")
        print(f"🎯 RESPONSE FROM: {response.execution_method.upper()}")
        print(f"📊 Confidence: {response.confidence:.2f}")
        print(f"⏱️  Execution time: {response.execution_time:.2f}s")
        
        if response.expert_used:
            print(f"🧠 Expert Used: {response.expert_used}")
        if response.moe_model_used:
            print(f"🤖 MoE Model: {response.moe_model_used}")
        
        print(f"{'='*60}")
        
        print(f"\n💡 Response:")
        print(f"{response.primary_response}")
        
        print(f"\n📋 Metadata:")
        for key, value in response.metadata.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"     {k}: {v}")
            else:
                print(f"   {key}: {value}")
    
    def interactive_mode(self):
        """Run in interactive mode"""
        if not self.manager:
            print("❌ System not initialized")
            return
        
        print(f"\n🎮 Interactive Mode - Type 'help' for commands, 'quit' to exit")
        print(f"Available experts: {', '.join(self.manager.get_available_experts())}")
        print(f"MoE enabled: {self.manager.enable_moe}")
        print(f"Execution mode: {self.manager.execution_mode.value}")
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n🧠 Hybrid> ").strip()
                
                if not user_input:
                    continue
                
                # Parse commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'status':
                    self._show_status()
                elif user_input.lower() == 'experts':
                    self._show_experts()
                elif user_input.lower() == 'stats':
                    self._show_stats()
                elif user_input.lower() == 'reset':
                    # Note: reset not available in hybrid manager
                    print("🔄 Reset not available in hybrid mode")
                elif user_input.lower().startswith('mode '):
                    self._change_mode(user_input[5:])
                elif user_input.lower().startswith('moe '):
                    self._toggle_moe(user_input[4:])
                elif user_input.lower().startswith('task '):
                    # Extract task type and query
                    parts = user_input[5:].split(' ', 1)
                    if len(parts) == 2:
                        task_type, query = parts
                        print(f"💡 Task execution not available in interactive mode")
                        print(f"   Use non-interactive mode: python -m src.cli.hybrid_cli \"{query}\" --task-type {task_type}")
                    else:
                        print("❌ Usage: task <task_type> <query>")
                elif user_input.lower().startswith('force '):
                    # Force neuroscience mode
                    query = user_input[6:].strip()
                    print(f"💡 Force neuroscience mode not available in interactive mode")
                    print(f"   Use non-interactive mode: python -m src.cli.hybrid_cli \"{query}\" --force-neuroscience")
                else:
                    # Treat as regular query
                    print(f"💡 Query execution not available in interactive mode")
                    print(f"   Use non-interactive mode: python -m src.cli.hybrid_cli \"{user_input}\"")
                    
            except KeyboardInterrupt:
                print(f"\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
Available Commands:
  help                    - Show this help
  status                 - Show system status
  experts                - List available experts
  stats                  - Show system statistics
  mode <mode>           - Change execution mode
  moe <on/off>          - Enable/disable MoE routing
  quit/exit/q           - Exit the system
  
Execution Modes:
  neuroscience_only      - Use only neuroscience experts
  moe_routing           - Use MoE for routing + experts for execution
  moe_fallback          - Try MoE first, fallback to experts
  expert_ensemble       - Combine multiple expert responses
  
MoE Control:
  moe on                - Enable MoE routing
  moe off               - Disable MoE routing (neuroscience only)
  
Task Types:
  biomedical_literature - Research and literature tasks
  biophysical_simulation - Neural simulation tasks
  neural_analysis       - Data analysis and code generation tasks
  cognitive_modeling    - Cognitive and behavioral modeling
  whole_brain_dynamics  - Whole brain and connectome analysis
  pytorch_snn           - PyTorch spiking neural networks
        """
        print(help_text)
    
    def _show_status(self):
        """Show system status"""
        if not self.manager:
            print("❌ System not initialized")
            return
        
        status = self.manager.get_system_status()
        
        print(f"\n📊 HYBRID SYSTEM STATUS")
        print(f"{'='*40}")
        
        # Hybrid manager status
        hybrid_status = status["hybrid_manager"]
        print(f"Total Queries: {hybrid_status['total_queries']}")
        print(f"Neuroscience Queries: {hybrid_status['neuroscience_queries']}")
        print(f"MoE Queries: {hybrid_status['moe_queries']}")
        print(f"Avg Response Time: {hybrid_status['average_response_time']:.2f}s")
        print(f"Execution Mode: {hybrid_status['execution_mode']}")
        print(f"MoE Enabled: {hybrid_status['moe_enabled']}")
        
        # Neuroscience status
        neuroscience_status = status["neuroscience_status"]
        print(f"\n🧠 NEUROSCIENCE STATUS")
        print(f"Total Experts: {neuroscience_status['total_experts']}")
        print(f"System Health: {neuroscience_status['system_health']}")
        
        # MoE status
        moe_status = status["moe_status"]
        print(f"\n🤖 MOE STATUS")
        print(f"Enabled: {moe_status['enabled']}")
        print(f"Router Available: {moe_status['router_available']}")
    
    def _show_experts(self):
        """Show available experts and capabilities"""
        if not self.manager:
            print("❌ System not initialized")
            return
        
        experts = self.manager.get_expert_capabilities()
        
        print(f"\n🧠 AVAILABLE NEUROSCIENCE EXPERTS")
        print(f"{'='*50}")
        
        for expert_name, capabilities in experts.items():
            print(f"\n📋 {expert_name}")
            print(f"   Task Types: {', '.join(capabilities.get('task_types', []))}")
            print(f"   Available: {capabilities.get('is_available', False)}")
            if 'description' in capabilities:
                print(f"   Description: {capabilities['description']}")
    
    def _show_stats(self):
        """Show system statistics"""
        if not self.manager:
            print("❌ System not initialized")
            return
        
        status = self.manager.get_system_status()
        
        print(f"\n📈 SYSTEM STATISTICS")
        print(f"{'='*40}")
        
        hybrid_status = status["hybrid_manager"]
        total = hybrid_status['total_queries']
        if total > 0:
            neuro_pct = (hybrid_status['neuroscience_queries'] / total) * 100
            moe_pct = (hybrid_status['moe_queries'] / total) * 100
            
            print(f"Query Distribution:")
            print(f"  Neuroscience: {hybrid_status['neuroscience_queries']} ({neuro_pct:.1f}%)")
            print(f"  MoE Routing: {hybrid_status['moe_queries']} ({moe_pct:.1f}%)")
            print(f"  Total: {total}")
            print(f"  Average Response Time: {hybrid_status['average_response_time']:.2f}s")
        else:
            print("No queries processed yet")
    
    def _change_mode(self, mode_name: str):
        """Change execution mode"""
        if not self.manager:
            print("❌ System not initialized")
            return
        
        try:
            new_mode = HybridExecutionMode(mode_name)
            self.manager.set_execution_mode(new_mode)
            print(f"✅ Execution mode changed to: {mode_name}")
        except ValueError:
            print(f"❌ Unknown execution mode: {mode_name}")
            print(f"Available modes: {[m.value for m in HybridExecutionMode]}")
    
    def _toggle_moe(self, state: str):
        """Enable or disable MoE routing"""
        if not self.manager:
            print("❌ System not initialized")
            return
        
        if state.lower() in ['on', 'enable', 'true', '1']:
            self.manager.toggle_moe(True)
            print(f"✅ MoE routing enabled")
        elif state.lower() in ['off', 'disable', 'false', '0']:
            self.manager.toggle_moe(False)
            print(f"✅ MoE routing disabled")
        else:
            print(f"❌ Unknown state: {state}")
            print(f"Use: moe on/off")

async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Hybrid MoE + Neuroscience Expert System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query with MoE routing
  python -m src.cli.hybrid_cli "What are the latest findings on hippocampal place cells?"
  
  # Query with task type
  python -m src.cli.hybrid_cli --task-type biophysical_simulation "Simulate a microcircuit with 100 neurons"
  
  # Force neuroscience mode only
  python -m src.cli.hybrid_cli --force-neuroscience "What is synaptic plasticity?"
  
  # Interactive mode
  python -m src.cli.hybrid_cli --interactive
  
  # Custom configuration
  python -m src.cli.hybrid_cli --mode neuroscience_only --disable-moe "Your query here"
        """
    )
    
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--task-type", "-t", help="Specify task type")
    parser.add_argument("--mode", "-m", default="moe_routing",
                       choices=[m.value for m in HybridExecutionMode],
                       help="Execution mode to use")
    parser.add_argument("--disable-moe", action="store_true",
                       help="Disable MoE routing (neuroscience only)")
    parser.add_argument("--force-neuroscience", action="store_true",
                       help="Force use of neuroscience experts only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Initialize CLI
    cli = HybridCLI()
    await cli.initialize(args.mode, not args.disable_moe)
    
    if args.interactive:
        cli.interactive_mode()
    elif args.query:
        await cli.process_single_query(args.query, args.task_type, args.force_neuroscience)
    else:
        # No query provided, show help
        parser.print_help()
        print(f"\n💡 Tip: Use --interactive for interactive mode or provide a query")

if __name__ == "__main__":
    asyncio.run(main())
