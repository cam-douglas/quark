#!/usr/bin/env python3
"""
Quick Start Demo for MoE Neuroscience Expert System
Demonstrates the system's capabilities with example queries
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from development.src.models.moe_manager import create_moe_manager, ExecutionMode
from development.src.models.moe_router import RoutingStrategy
from development.src.models.neuroscience_experts import NeuroscienceTaskType

async def demo_moe_system():
    """Demonstrate the MoE system with example queries"""
    
    print("🧠 MoE Neuroscience Expert System - Quick Start Demo")
    print("=" * 60)
    
    # Initialize the system
    print("\n🔧 Initializing system...")
    manager = create_moe_manager(
        routing_strategy=RoutingStrategy.CONFIDENCE_BASED,
        execution_mode=ExecutionMode.SINGLE_EXPERT
    )
    
    # Check system health
    health = await manager.health_check()
    print(f"✅ System status: {health['status']}")
    print(f"📊 Available experts: {len(manager.get_available_experts())}")
    
    # Example queries to demonstrate different expert types
    demo_queries = [
        {
            "query": "What are the latest findings on hippocampal place cells?",
            "description": "Literature/Research Query",
            "expected_expert": "BioGPT"
        },
        {
            "query": "Simulate a microcircuit with 100 spiking neurons",
            "description": "Simulation Query", 
            "expected_expert": "Brian2 or NEURON"
        },
        {
            "query": "Fetch the latest Allen Brain Atlas dataset for visual cortex",
            "description": "Data Query",
            "expected_expert": "Data Expert"
        },
        {
            "query": "Write a Python script to analyze spike train data",
            "description": "Code Generation Query",
            "expected_expert": "Coding Expert"
        },
        {
            "query": "Compare the performance of different neural network architectures",
            "description": "Analysis Query",
            "expected_expert": "General"
        }
    ]
    
    print(f"\n🚀 Running {len(demo_queries)} demo queries...")
    print("-" * 60)
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n📝 Query {i}: {demo['description']}")
        print(f"   Query: {demo['query']}")
        print(f"   Expected Expert: {demo['expected_expert']}")
        
        try:
            # Process query
            response = await manager.process_query(demo['query'])
            
            # Display results
            print(f"   ✅ Routed to: {response.primary_expert}")
            print(f"   📊 Confidence: {response.confidence:.2f}")
            print(f"   ⏱️  Time: {response.execution_time:.2f}s")
            print(f"   💡 Response: {response.primary_response[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        print("-" * 40)
    
    # Show system statistics
    print(f"\n📈 System Performance Summary")
    print("=" * 40)
    
    status = manager.get_system_status()
    moe_status = status["moe_manager"]
    
    print(f"Total Queries: {moe_status['total_queries']}")
    print(f"Success Rate: {moe_status['success_rate']:.2%}")
    print(f"Average Response Time: {moe_status['average_response_time']:.2f}s")
    print(f"Execution Mode: {moe_status['execution_mode']}")
    print(f"Routing Strategy: {moe_status['routing_strategy']}")
    
    # Show routing statistics
    routing_stats = manager.router.get_routing_stats()
    print(f"\n🔄 Routing Statistics:")
    print(f"Total Routes: {routing_stats['total_routes']}")
    print(f"Strategy Usage: {routing_stats['strategy_usage']}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"💡 Try running with --interactive for interactive mode")

async def demo_advanced_features():
    """Demonstrate advanced MoE features"""
    
    print(f"\n🔬 Advanced Features Demo")
    print("=" * 40)
    
    # Create manager with different configurations
    manager = create_moe_manager(
        routing_strategy=RoutingStrategy.LOAD_BALANCED,
        execution_mode=ExecutionMode.EXPERT_ENSEMBLE
    )
    
    print(f"📊 Testing ensemble mode with load balancing...")
    
    # Test ensemble execution
    query = "Analyze the relationship between neural oscillations and attention"
    
    try:
        response = await manager.process_query(query)
        
        print(f"✅ Ensemble execution successful!")
        print(f"   Primary Expert: {response.primary_expert}")
        print(f"   Fallback Experts: {len(response.fallback_responses)}")
        print(f"   Total Experts Consulted: {response.metadata.get('total_experts_consulted', 1)}")
        
        if response.fallback_responses:
            print(f"   Fallback Responses:")
            for expert_name, fallback_response in response.fallback_responses.items():
                print(f"     {expert_name}: {fallback_response[:80]}...")
        
    except Exception as e:
        print(f"❌ Ensemble execution failed: {e}")
    
    # Test different routing strategies
    strategies = [RoutingStrategy.CONFIDENCE_BASED, RoutingStrategy.LOAD_BALANCED]
    
    print(f"\n🔄 Testing different routing strategies...")
    
    for strategy in strategies:
        try:
            manager.set_routing_strategy(strategy)
            response = await manager.process_query("What is synaptic plasticity?")
            print(f"   {strategy.value}: Routed to {response.primary_expert} "
                  f"(confidence: {response.confidence:.2f})")
        except Exception as e:
            print(f"   {strategy.value}: Failed - {e}")

async def main():
    """Main demo function"""
    try:
        # Basic demo
        await demo_moe_system()
        
        # Advanced features demo
        await demo_advanced_features()
        
        print(f"\n🎯 Demo completed! The MoE system is ready for use.")
        print(f"💡 Next steps:")
        print(f"   1. Run 'python -m src.cli.moe_cli --interactive' for interactive mode")
        print(f"   2. Use 'python -m src.cli.moe_cli \"your query\"' for single queries")
        print(f"   3. Explore different routing strategies and execution modes")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
