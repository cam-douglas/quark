#!/usr/bin/env python3
"""
Enhanced CLI for Hybrid MoE Neuroscience Expert System

This CLI provides access to both the neuroscience expert system and basic model management.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# from src.models.moe_manager import MoEManager, ExecutionMode  # Temporarily disabled
from .....................................................models.model_manager import MoEModelManager
from .....................................................models.moe_router import MoERouter, RoutingStrategy
from .....................................................models.moe_manager import MoEManager, ExecutionMode  # Add this import

def download_models(manager: MoEModelManager, model_names: list, force: bool = False):
    """Download specified models with speed optimizations."""
    print(f"🚀 Downloading models: {', '.join(model_names)}")
    print(f"   Speed optimizations: Parallel downloads, resume support, essential files only")
    
    for model_name in model_names:
        try:
            print(f"\n📥 Starting download of {model_name}...")
            local_path = manager.download_model(model_name, force_redownload=force)
            print(f"✅ {model_name} download completed successfully!")
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")

def list_models(manager: MoEModelManager):
    """List all available and downloaded models."""
    print("\n📋 Model Information:")
    print("=" * 50)
    
    # Available models
    available = manager.get_available_models()
    print(f"Available models ({len(available)}):")
    for model in available:
        config = manager.model_configs[model]
        print(f"  • {model}: {config.model_name}")
        print(f"    - HuggingFace ID: {config.model_id}")
        print(f"    - Experts: {config.num_experts}")
        print(f"    - Top-k: {config.top_k}")
        print()
    
    # Downloaded models
    downloaded = manager.get_downloaded_models()
    if downloaded:
        print(f"Downloaded models ({len(downloaded)}):")
        for model in downloaded:
            info = manager.get_model_info(model)
            if info:
                status = "🟢 Loaded" if info.is_loaded else "🔵 Downloaded"
                print(f"  • {model}: {status}")
                print(f"    - Size: {info.model_size_gb:.2f} GB")
                print(f"    - Path: {info.local_path}")
                print()
    else:
        print("No models downloaded yet.")

def test_routing(router: MoERouter, text: str):
    """Test the routing system with given text."""
    print(f"\n🧠 Testing MoE Routing:")
    print(f"Input: {text}")
    print("-" * 50)
    
    try:
        # Get routing decision - handle async properly
        import asyncio
        decision = asyncio.run(router.route_query(text))
        
        print(f"Task Type: {decision.strategy_used.value}")
        print(f"Primary Expert: {decision.expert_name}")
        print(f"Fallback Experts: {decision.fallback_experts}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reasoning: {decision.reasoning}")
        
    except Exception as e:
        print(f"❌ Routing failed: {e}")

def execute_with_routing(manager: MoEManager, text: str, max_tokens: int = 256):
    """Execute a request using the routing system."""
    print(f"\n🚀 Executing with MoE Routing:")
    print(f"Input: {text}")
    print("-" * 50)
    
    try:
        # Use the hybrid LLM mode
        manager.set_execution_mode(ExecutionMode.HYBRID_LLM)
        
        # Process the query
        import asyncio
        response = asyncio.run(manager.process_query(text))
        
        print(f"🏆 Primary Expert ({response.primary_expert}):")
        print(f"   Response: {response.primary_response}")
        print(f"   Execution Time: {response.execution_time:.2f}s")
        print(f"   Confidence: {response.confidence:.2f}")
        
        if response.llm_model_used:
            print(f"   LLM Model: {response.llm_model_used}")
        
        if response.metadata:
            print(f"\n📋 Metadata:")
            for key, value in response.metadata.items():
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")

def show_routing_stats(router: MoERouter):
    """Show routing statistics."""
    stats = router.get_routing_stats()
    
    print(f"\n📊 Routing Statistics:")
    print("=" * 50)
    
    if stats["total_routes"] == 0:
        print("No requests processed yet.")
        return
    
    print(f"Total Routes: {stats['total_routes']}")
    
    # Task distribution
    if stats.get("task_distribution"):
        print(f"\nTask Distribution:")
        for task, count in stats["task_distribution"].items():
            percentage = (count / stats["total_routes"]) * 100
            print(f"  • {task}: {count} ({percentage:.1f}%)")
    
    # Expert usage
    if stats.get("expert_usage"):
        print(f"\nExpert Usage:")
        for expert, count in stats["expert_usage"].items():
            percentage = (count / stats["total_routes"]) * 100
            print(f"  • {expert}: {count} ({percentage:.1f}%)")

def show_system_status(manager: MoEManager):
    """Show comprehensive system status."""
    print(f"\n🔍 System Status:")
    print("=" * 50)
    
    try:
        status = manager.get_system_status()
        
        # MoE Manager status
        moe_status = status["moe_manager"]
        print(f"📊 MoE Manager:")
        print(f"   Total Queries: {moe_status['total_queries']}")
        print(f"   Success Rate: {moe_status['success_rate']:.2%}")
        print(f"   Avg Response Time: {moe_status['average_response_time']:.2f}s")
        print(f"   Execution Mode: {moe_status['execution_mode']}")
        print(f"   Routing Strategy: {moe_status['routing_strategy']}")
        
        # Expert status
        expert_status = status.get("expert_status", {})
        if expert_status:
            print(f"\n🧠 Expert System:")
            print(f"   Total Experts: {expert_status.get('total_experts', 0)}")
            print(f"   System Health: {expert_status.get('system_health', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Failed to get system status: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced MoE Neuroscience Expert System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a model with speed optimizations
  python moe_cli.py download deepseek-v2
  
  # Download multiple models
  python moe_cli.py download qwen1.5-moe mix-tao-moe
  
  # Force re-download
  python moe_cli.py download deepseek-v2 --force
  
  # List all models
  python moe_cli.py list
  
  # Test routing
  python moe_cli.py route "Simulate a neural circuit"
  
  # Execute with routing
  python moe_cli.py execute "Explain synaptic plasticity" --max-tokens 512
  
  # Show system status
  python moe_cli.py status
        """
    )
    
    parser.add_argument(
        "--base-dir", 
        default="./models",
        help="Base directory for models (default: ./models)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download models with speed optimizations")
    download_parser.add_argument("models", nargs="+", help="Model names to download")
    download_parser.add_argument("--force", action="store_true", help="Force re-download")
    
    # List command
    subparsers.add_parser("list", help="List all models")
    
    # Route command
    route_parser = subparsers.add_parser("route", help="Test routing with text")
    route_parser.add_argument("text", help="Text to route")
    
    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute with routing")
    execute_parser.add_argument("text", help="Text to execute")
    execute_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    
    # Stats command
    subparsers.add_parser("stats", help="Show routing statistics")
    
    # Status command
    subparsers.add_parser("status", help="Show comprehensive system status")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Start interactive mode")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize manager and router
        model_manager = MoEModelManager(base_dir=args.base_dir)
        router = MoERouter()
        
        if args.command == "download":
            download_models(model_manager, args.models, args.force)
        
        elif args.command == "list":
            list_models(model_manager)
        
        elif args.command == "route":
            test_routing(router, args.text)
        
        elif args.command == "execute":
            # Create MoE manager for execution
            moe_manager = MoEManager()
            execute_with_routing(moe_manager, args.text, args.max_tokens)
        
        elif args.command == "stats":
            show_routing_stats(router)
        
        elif args.command == "status":
            # Create MoE manager for status
            moe_manager = MoEManager()
            show_system_status(moe_manager)
        
        elif args.command == "interactive":
            interactive_mode(model_manager, router)
        
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def interactive_mode(model_manager: MoEModelManager, router: MoERouter):
    """Start interactive mode for testing."""
    print("\n🎮 Interactive MoE Testing Mode")
    print("Type 'help' for commands, 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n🤖 > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.lower() == 'help':
                print("""
Available commands:
  help                    - Show this help
  list                    - List all models
  download <model>        - Download a model with speed optimizations
  route <text>           - Test routing
  execute <text>         - Execute with routing
  stats                  - Show routing statistics
  status                 - Show system status
  quit/exit/q            - Exit interactive mode
                """)
                continue
            
            if user_input.lower().startswith('download '):
                model_name = user_input[9:].strip()
                download_models(model_manager, [model_name])
                continue
            
            if user_input.lower().startswith('route '):
                text = user_input[6:].strip()
                test_routing(router, text)
                continue
            
            if user_input.lower().startswith('execute '):
                text = user_input[8:].strip()
                # Create MoE manager for execution
                moe_manager = MoEManager()
                execute_with_routing(moe_manager, text)
                continue
            
            if user_input.lower() == 'list':
                list_models(model_manager)
                continue
            
            if user_input.lower() == 'stats':
                show_routing_stats(router)
                continue
            
            if user_input.lower() == 'status':
                # Create MoE manager for status
                moe_manager = MoEManager()
                show_system_status(moe_manager)
                continue
            
            # Default: treat as text to route
            print(f"🤔 Routing: {user_input}")
            test_routing(router, user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
