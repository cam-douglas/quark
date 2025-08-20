#!/usr/bin/env python3
"""
Test Script for MoE Neuroscience Expert System
Verifies that all components are working correctly
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from development.src.models.moe_router import MoERouter, RoutingStrategy, RoutingDecision
        print("✅ MoE Router imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MoE Router: {e}")
        return False
    
    try:
        from development.src.models.moe_manager import MoEManager, ExecutionMode, create_moe_manager
        print("✅ MoE Manager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MoE Manager: {e}")
        return False
    
    try:
        from development.src.models.neuroscience_experts import NeuroscienceExpertManager, NeuroscienceTask, NeuroscienceTaskType
        print("✅ Neuroscience Experts imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Neuroscience Experts: {e}")
        return False
    
    return True

def test_expert_manager():
    """Test the neuroscience expert manager"""
    print("\n🧠 Testing Neuroscience Expert Manager...")
    
    try:
        from development.src.models.neuroscience_experts import NeuroscienceExpertManager
        
        manager = NeuroscienceExpertManager()
        experts = manager.get_available_experts()
        
        print(f"✅ Expert Manager initialized successfully")
        print(f"   Available experts: {len(experts)}")
        
        if experts:
            print(f"   Expert names: {', '.join(experts)}")
            
            # Test capabilities
            capabilities = manager.get_expert_capabilities()
            print(f"   Expert capabilities retrieved: {len(capabilities)}")
            
            # Test system status
            status = manager.get_system_status()
            print(f"   System health: {status['system_health']}")
            
        return True
        
    except Exception as e:
        print(f"❌ Expert Manager test failed: {e}")
        return False

def test_moe_router():
    """Test the MoE router"""
    print("\n🔄 Testing MoE Router...")
    
    try:
        from development.src.models.moe_router import MoERouter, RoutingStrategy
        
        router = MoERouter(RoutingStrategy.CONFIDENCE_BASED)
        print(f"✅ MoE Router initialized successfully")
        print(f"   Routing strategy: {router.strategy.value}")
        
        # Test routing stats
        stats = router.get_routing_stats()
        print(f"   Initial routing stats: {stats['total_routes']} routes")
        
        return True
        
    except Exception as e:
        print(f"❌ MoE Router test failed: {e}")
        return False

async def test_moe_manager():
    """Test the MoE manager"""
    print("\n🎯 Testing MoE Manager...")
    
    try:
        from development.src.models.moe_manager import create_moe_manager, ExecutionMode
        from development.src.models.moe_router import RoutingStrategy
        
        manager = create_moe_manager(
            routing_strategy=RoutingStrategy.CONFIDENCE_BASED,
            execution_mode=ExecutionMode.SINGLE_EXPERT
        )
        
        print(f"✅ MoE Manager initialized successfully")
        print(f"   Execution mode: {manager.execution_mode.value}")
        print(f"   Routing strategy: {manager.router.strategy.value}")
        
        # Test health check
        health = await manager.health_check()
        print(f"   System health: {health['status']}")
        
        # Test system status
        status = manager.get_system_status()
        print(f"   Total experts: {status['expert_status']['total_experts']}")
        
        return True
        
    except Exception as e:
        print(f"❌ MoE Manager test failed: {e}")
        return False

async def test_basic_query():
    """Test a basic query through the system"""
    print("\n💬 Testing Basic Query...")
    
    try:
        from development.src.models.moe_manager import create_moe_manager
        
        manager = create_moe_manager()
        
        # Simple test query
        query = "What is a neuron?"
        
        print(f"   Testing query: {query}")
        
        response = await manager.process_query(query)
        
        print(f"✅ Query processed successfully!")
        print(f"   Routed to: {response.primary_expert}")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Response length: {len(response.primary_response)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic query test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 MoE Neuroscience Expert System - System Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Expert Manager
    if test_expert_manager():
        tests_passed += 1
    
    # Test 3: MoE Router
    if test_moe_router():
        tests_passed += 1
    
    # Test 4: MoE Manager
    if await test_moe_manager():
        tests_passed += 1
    
    # Test 5: Basic Query
    if await test_basic_query():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results")
    print("=" * 40)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print(f"\n🎉 All tests passed! The system is ready for use.")
        print(f"💡 Next steps:")
        print(f"   1. Run the demo: python quick_start.py")
        print(f"   2. Try interactive mode: python -m src.cli.moe_cli --interactive")
        print(f"   3. Process a query: python -m src.cli.moe_cli \"your question\"")
    else:
        print(f"\n⚠️  Some tests failed. Check the errors above.")
        print(f"💡 Common issues:")
        print(f"   - Missing dependencies: pip install -r requirements.txt")
        print(f"   - Model files not downloaded")
        print(f"   - Configuration issues")
        
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
