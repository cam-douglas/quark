#!/usr/bin/env python3
"""
Hybrid MoE + Neuroscience Expert System Demo
Demonstrates the hybrid approach combining both systems
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.hybrid_moe_manager import (
    create_hybrid_manager, HybridExecutionMode, quick_hybrid_query
)
from src.models.neuroscience_experts import NeuroscienceTaskType

async def demo_hybrid_system():
    """Demonstrate the hybrid system capabilities"""
    
    print("🧠 Hybrid MoE + Neuroscience Expert System - Demo")
    print("=" * 70)
    print("🎯 This system combines Mixtral-8x7B intelligent routing with")
    print("   specialized neuroscience experts for the best of both worlds!")
    print("=" * 70)
    
    # Test 1: Neuroscience Only Mode
    print(f"\n🔬 Test 1: Neuroscience Only Mode")
    print("-" * 50)
    
    try:
        manager = create_hybrid_manager(
            execution_mode=HybridExecutionMode.NEUROSCIENCE_ONLY,
            enable_moe=False
        )
        
        print(f"✅ Manager initialized in neuroscience-only mode")
        print(f"   Available experts: {len(manager.get_available_experts())}")
        
        # Test a simple query
        query = "What is synaptic plasticity?"
        print(f"\n📝 Query: {query}")
        
        response = await manager.process_query(query)
        
        print(f"✅ Response received!")
        print(f"   Execution method: {response.execution_method}")
        print(f"   Expert used: {response.expert_used}")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Response length: {len(response.primary_response)} characters")
        
    except Exception as e:
        print(f"❌ Neuroscience-only test failed: {e}")
    
    # Test 2: MoE Routing Mode (if available)
    print(f"\n🤖 Test 2: MoE Routing Mode")
    print("-" * 50)
    
    try:
        manager = create_hybrid_manager(
            execution_mode=HybridExecutionMode.MOE_ROUTING,
            enable_moe=True
        )
        
        print(f"✅ Manager initialized in MoE routing mode")
        print(f"   MoE enabled: {manager.enable_moe}")
        print(f"   MoE router available: {manager.moe_router is not None}")
        
        # Test a complex query that benefits from MoE routing
        query = "Analyze the relationship between neural oscillations and attention in the context of recent research findings"
        print(f"\n📝 Query: {query}")
        
        response = await manager.process_query(query)
        
        print(f"✅ Response received!")
        print(f"   Execution method: {response.execution_method}")
        print(f"   Expert used: {response.expert_used}")
        print(f"   MoE model used: {response.moe_model_used}")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Response length: {len(response.primary_response)} characters")
        
        # Show MoE routing metadata if available
        if response.routing_decision:
            print(f"   MoE routing confidence: {response.routing_decision.confidence:.2f}")
            print(f"   MoE reasoning: {response.routing_decision.reasoning}")
        
    except Exception as e:
        print(f"❌ MoE routing test failed: {e}")
        print(f"   This is expected if Mixtral-8x7B is not available")
    
    # Test 3: Expert Ensemble Mode
    print(f"\n👥 Test 3: Expert Ensemble Mode")
    print("-" * 50)
    
    try:
        manager = create_hybrid_manager(
            execution_mode=HybridExecutionMode.EXPERT_ENSEMBLE,
            enable_moe=False
        )
        
        print(f"✅ Manager initialized in expert ensemble mode")
        
        # Test a query that benefits from multiple expert opinions
        query = "Compare different approaches to modeling neural networks"
        print(f"\n📝 Query: {query}")
        
        response = await manager.process_query(query)
        
        print(f"✅ Response received!")
        print(f"   Execution method: {response.execution_method}")
        print(f"   Expert used: {response.expert_used}")
        print(f"   Total experts consulted: {response.metadata.get('total_experts_consulted', 1)}")
        print(f"   Response length: {len(response.primary_response)} characters")
        
        # Show additional experts if any
        if 'additional_experts' in response.metadata:
            print(f"   Additional experts: {', '.join(response.metadata['additional_experts'])}")
        
    except Exception as e:
        print(f"❌ Expert ensemble test failed: {e}")
    
    # Test 4: MoE Fallback Mode
    print(f"\n🔄 Test 4: MoE Fallback Mode")
    print("-" * 50)
    
    try:
        manager = create_hybrid_manager(
            execution_mode=HybridExecutionMode.MOE_FALLBACK,
            enable_moe=True
        )
        
        print(f"✅ Manager initialized in MoE fallback mode")
        
        # Test a query that will fallback to neuroscience experts
        query = "What are the key principles of neural development?"
        print(f"\n📝 Query: {query}")
        
        response = await manager.process_query(query)
        
        print(f"✅ Response received!")
        print(f"   Execution method: {response.execution_method}")
        print(f"   Expert used: {response.expert_used}")
        print(f"   MoE model used: {response.moe_model_used}")
        print(f"   Confidence: {response.confidence:.2f}")
        
    except Exception as e:
        print(f"❌ MoE fallback test failed: {e}")
    
    # Show system status
    print(f"\n📊 System Status Summary")
    print("=" * 50)
    
    try:
        status = manager.get_system_status()
        
        hybrid_status = status["hybrid_manager"]
        print(f"Total Queries: {hybrid_status['total_queries']}")
        print(f"Neuroscience Queries: {hybrid_status['neuroscience_queries']}")
        print(f"MoE Queries: {hybrid_status['moe_queries']}")
        print(f"Average Response Time: {hybrid_status['average_response_time']:.2f}s")
        print(f"Final Execution Mode: {hybrid_status['execution_mode']}")
        print(f"MoE Enabled: {hybrid_status['moe_enabled']}")
        
        # Show expert status
        neuroscience_status = status["neuroscience_status"]
        print(f"\n🧠 Neuroscience Status:")
        print(f"   Total Experts: {neuroscience_status['total_experts']}")
        print(f"   System Health: {neuroscience_status['system_health']}")
        
        # Show MoE status
        moe_status = status["moe_status"]
        print(f"\n🤖 MoE Status:")
        print(f"   Enabled: {moe_status['enabled']}")
        print(f"   Router Available: {moe_status['router_available']}")
        
    except Exception as e:
        print(f"❌ Failed to get system status: {e}")
    
    print(f"\n🎉 Hybrid system demo completed!")
    print(f"💡 Next steps:")
    print(f"   1. Try the hybrid CLI: python -m src.cli.hybrid_cli --interactive")
    print(f"   2. Test specific modes: python -m src.cli.hybrid_cli --mode neuroscience_only \"your query\"")
    print(f"   3. Force neuroscience: python -m src.cli.hybrid_cli --force-neuroscience \"your query\"")
    print(f"   4. Enable MoE routing: python -m src.cli.hybrid_cli --mode moe_routing \"your query\"")

async def demo_quick_queries():
    """Demonstrate quick query functionality"""
    
    print(f"\n⚡ Quick Query Demo")
    print("=" * 40)
    
    queries = [
        "What is a neuron?",
        "Explain synaptic plasticity",
        "How do neural networks learn?",
        "What are the latest findings in neuroscience?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Quick Query {i}: {query}")
        
        try:
            response = await quick_hybrid_query(query)
            
            print(f"   ✅ Method: {response.execution_method}")
            print(f"   🧠 Expert: {response.expert_used}")
            print(f"   🤖 MoE: {response.moe_model_used or 'N/A'}")
            print(f"   📊 Confidence: {response.confidence:.2f}")
            print(f"   💬 Response: {response.primary_response[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        print("-" * 30)

async def main():
    """Main demo function"""
    try:
        # Main hybrid system demo
        await demo_hybrid_system()
        
        # Quick query demo
        await demo_quick_queries()
        
        print(f"\n🚀 The hybrid system is ready for use!")
        print(f"🎯 You now have:")
        print(f"   • Neuroscience experts for specialized tasks")
        print(f"   • MoE routing for intelligent query understanding")
        print(f"   • Automatic fallback between systems")
        print(f"   • Multiple execution modes for different needs")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
