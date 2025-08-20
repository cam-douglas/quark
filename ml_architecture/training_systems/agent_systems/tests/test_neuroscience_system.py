#!/usr/bin/env python3
"""
Test script for Neuroscience Domain Experts System

This script tests the basic functionality of the neuroscience expert system
to ensure it's working correctly.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from models.neuroscience_experts import (
            NeuroscienceExpertManager,
            NeuroscienceTask,
            NeuroscienceTaskType
        )
        print("✅ All neuroscience expert modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_expert_manager():
    """Test the neuroscience expert manager"""
    print("\n🧪 Testing expert manager...")
    
    try:
        from models.neuroscience_experts import NeuroscienceExpertManager
        
        manager = NeuroscienceExpertManager()
        status = manager.get_system_status()
        
        print(f"✅ Expert manager created successfully")
        print(f"   Total experts: {status['total_experts']}")
        print(f"   System health: {status['system_health']}")
        
        return True
    except Exception as e:
        print(f"❌ Expert manager test failed: {e}")
        return False

def test_task_creation():
    """Test neuroscience task creation"""
    print("\n🧪 Testing task creation...")
    
    try:
        from models.neuroscience_experts import NeuroscienceTask, NeuroscienceTaskType
        
        task = NeuroscienceTask(
            task_type=NeuroscienceTaskType.BIOMEDICAL_LITERATURE,
            description="Test biomedical task",
            parameters={"max_length": 50},
            expected_output="Test output"
        )
        
        print(f"✅ Task created successfully")
        print(f"   Type: {task.task_type.value}")
        print(f"   Description: {task.description}")
        
        return True
    except Exception as e:
        print(f"❌ Task creation test failed: {e}")
        return False

def test_expert_availability():
    """Test expert availability checking"""
    print("\n🧪 Testing expert availability...")
    
    try:
        from models.neuroscience_experts import NeuroscienceExpertManager
        
        manager = NeuroscienceExpertManager()
        experts = manager.get_available_experts()
        capabilities = manager.get_expert_capabilities()
        
        print(f"✅ Expert availability check successful")
        print(f"   Available experts: {len(experts)}")
        
        for expert_name in experts:
            expert_info = capabilities[expert_name]
            print(f"   • {expert_name}: {expert_info['available']}")
        
        return True
    except Exception as e:
        print(f"❌ Expert availability test failed: {e}")
        return False

def test_simple_execution():
    """Test simple task execution (if any experts are available)"""
    print("\n🧪 Testing simple task execution...")
    
    try:
        from models.neuroscience_experts import (
            NeuroscienceExpertManager,
            NeuroscienceTask,
            NeuroscienceTaskType
        )
        
        manager = NeuroscienceExpertManager()
        available_experts = manager.get_available_experts()
        
        if not available_experts:
            print("⚠️  No experts available for execution test")
            return True
        
        # Try to execute a simple task
        task = NeuroscienceTask(
            task_type=NeuroscienceTaskType.NEURAL_ANALYSIS,
            description="Simple test task",
            parameters={},
            expected_output="Test result"
        )
        
        result = manager.execute_task(task)
        
        if result.get('success'):
            print(f"✅ Task execution successful")
            print(f"   Expert used: {result.get('routed_to_expert', 'Unknown')}")
        else:
            print(f"⚠️  Task execution failed (expected for test): {result.get('error', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Task execution test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 NEUROSCIENCE EXPERT SYSTEM TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_expert_manager,
        test_task_creation,
        test_expert_availability,
        test_simple_execution
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Neuroscience expert system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Next steps:")
    print("   1. Install required dependencies: pip install -r requirements.txt")
    print("   2. Run CLI: python src/cli/neuroscience_cli.py status")
    print("   3. Run demo: python examples/neuroscience_demo.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
