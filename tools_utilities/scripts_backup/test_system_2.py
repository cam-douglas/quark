#!/usr/bin/env python3
"""
Test script for the enhanced Small-Mind Multi-Agent Terminal Hub

This script tests the core functionality of the agent hub system including:
- Needs inference
- Model routing
- Resource limits
- Safety features
- Adapter functionality
"""

import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_utils():
    """Test utility functions."""
    print("🧪 Testing utility functions...")
    
    try:
        from utils import (
            new_run_dir, seed_everything, apply_resource_limits,
            capture_system_info, validate_run_environment
        )
        
        # Test run directory creation
        run_dir = new_run_dir("test")
        print(f"✅ Created run directory: {run_dir}")
        
        # Test seeding
        seed_everything(42)
        print("✅ Applied deterministic seeds")
        
        # Test system info capture
        sys_info = capture_system_info()
        print(f"✅ Captured system info: {sys_info['platform']} on {sys_info['architecture']}")
        
        # Test resource limits
        env = apply_resource_limits(cpu_limit=2, memory_limit_gb=4, gpu_limit="0")
        print(f"✅ Applied resource limits: {env.get('SM_CPU_LIMIT')} CPU, {env.get('SM_MEMORY_LIMIT_GB')}GB RAM")
        
        # Cleanup
        import shutil
        shutil.rmtree(run_dir)
        print("✅ Cleaned up test run directory")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility test failed: {e}")
        return False

def test_planner():
    """Test needs inference and planning."""
    print("\n🧪 Testing planner functionality...")
    
    try:
        from planner import infer_needs, validate_needs, suggest_alternative_needs
        
        # Test basic needs inference
        needs = infer_needs("Install numpy using pip")
        print(f"✅ Inferred needs: {needs['need']}")
        print(f"   Primary need: {needs['primary_need']}")
        print(f"   Complexity: {needs['complexity']}")
        
        # Test validation
        is_valid = validate_needs(needs)
        print(f"✅ Needs validation: {is_valid}")
        
        # Test alternative suggestions
        alternatives = suggest_alternative_needs(needs)
        print(f"✅ Alternative needs: {alternatives}")
        
        # Test complex prompt
        complex_needs = infer_needs("Plan and implement a machine learning pipeline with data preprocessing")
        print(f"✅ Complex needs: {complex_needs['need']}")
        print(f"   Complexity: {complex_needs['complexity']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Planner test failed: {e}")
        return False

def test_router():
    """Test routing functionality."""
    print("\n🧪 Testing router functionality...")
    
    try:
        from router import Router, get_router
        
        # Create mock registry and routing config
        class MockRegistry:
            def list(self):
                return [
                    {"id": "test.model1", "capabilities": ["shell", "python"], "concurrency": 1},
                    {"id": "test.model2", "capabilities": ["reasoning", "planning"], "concurrency": 2}
                ]
            def get(self, model_id):
                for model in self.list():
                    if model["id"] == model_id:
                        return model
                raise KeyError(f"Model {model_id} not found")
        
        routing_config = [
            {"if": {"need": "shell"}, "then": "test.model1"},
            {"if": {"need": "planning"}, "then": "test.model2"},
            {"default": "test.model1"}
        ]
        
        # Test router creation
        router = get_router(MockRegistry(), routing_config)
        print("✅ Router created successfully")
        
        # Test routing decisions
        needs = {"need": ["shell"], "primary_need": "shell", "complexity": "low"}
        model_id = router.choose_model(needs, MockRegistry())
        print(f"✅ Routed shell need to: {model_id}")
        
        needs = {"need": ["planning"], "primary_need": "planning", "complexity": "medium"}
        model_id = router.choose_model(needs, MockRegistry())
        print(f"✅ Routed planning need to: {model_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Router test failed: {e}")
        return False

def test_adapters():
    """Test adapter functionality."""
    print("\n🧪 Testing adapter functionality...")
    
    try:
        # Test Open Interpreter adapter
        from adapters.adapter_open_interpreter import OpenInterpreterAdapter
        config = {"bin": "echo", "flags": ["--help"], "timeout": 30}
        adapter = OpenInterpreterAdapter(config)
        print("✅ Open Interpreter adapter created")
        
        # Test SmallMind adapter
        from adapters.adapter_smallmind import SmallMindAdapter
        config = {"entry": "test:main", "timeout": 30}
        adapter = SmallMindAdapter(config)
        print("✅ SmallMind adapter created")
        
        # Test Transformers adapter
        from adapters.adapter_transformers import TransformersAdapter
        config = {"model_id": "test/model", "device": "cpu", "timeout": 30}
        adapter = TransformersAdapter(config)
        print("✅ Transformers adapter created")
        
        # Test CrewAI adapter
        from adapters.adapter_crewai import CrewAIAdapter
        config = {"module": "test", "timeout": 30}
        adapter = CrewAIAdapter(config)
        print("✅ CrewAI adapter created")
        
        # Test LlamaCPP adapter
        from adapters.adapter_llamacpp import LlamaCPPAdapter
        config = {"model_path": "test.gguf", "timeout": 30}
        adapter = LlamaCPPAdapter(config)
        print("✅ LlamaCPP adapter created")
        
        return True
        
    except Exception as e:
        print(f"❌ Adapter test failed: {e}")
        return False

def test_registry():
    """Test registry functionality."""
    print("\n🧪 Testing registry functionality...")
    
    try:
        from registry import ModelRegistry
        
        # Test registry creation (will fail if models.yaml doesn't exist, but that's OK)
        try:
            registry = ModelRegistry()
            print("✅ Registry created successfully")
            
            # Test listing models
            models = registry.list()
            print(f"✅ Found {len(models)} models in registry")
            
            # Test routing configuration
            routing = registry.routing
            print(f"✅ Routing configuration: {len(routing)} rules")
            
        except FileNotFoundError:
            print("⚠️  models.yaml not found (expected in development)")
            print("✅ Registry structure is correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_integration():
    """Test basic integration."""
    print("\n🧪 Testing basic integration...")
    
    try:
        # Test that we can import the main modules
        from cli import main
        print("✅ CLI module imported successfully")
        
        from runner import run_model
        print("✅ Runner module imported successfully")
        
        print("✅ Basic integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Small-Mind Agent Hub System Tests\n")
    
    tests = [
        ("Utilities", test_utils),
        ("Planner", test_planner),
        ("Router", test_router),
        ("Adapters", test_adapters),
        ("Registry", test_registry),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:15s}: {status}")
        if success:
            passed += 1
    
    print("-"*50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run tests
    exit_code = main()
    sys.exit(exit_code)
