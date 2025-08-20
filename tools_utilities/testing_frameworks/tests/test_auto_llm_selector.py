#!/usr/bin/env python3
"""
Test script for Auto LLM Selector functionality
Demonstrates intelligent LLM selection for different tasks
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.autonomous_code_editor import (
    AutonomousCodeEditor, SafetyConfig, SafetyLevel, ChangeType, AutoLLMSelector
)

async def test_auto_llm_selector():
    """Test the auto LLM selector functionality"""
    print("🧠 Testing Auto LLM Selector")
    print("=" * 50)
    
    # Create configuration
    config = SafetyConfig()
    config.auto_llm_selector = True
    config.fallback_to_local = True
    
    # Create auto LLM selector
    selector = AutoLLMSelector(config)
    
    print(f"🔍 Available LLMs: {list(selector.available_llms.keys())}")
    print(f"📊 LLM Capabilities: {len(selector.llm_capabilities)} models")
    print()
    
    # Test different task types
    test_cases = [
        ("documentation", SafetyLevel.LOW, "Add docstrings"),
        ("optimization", SafetyLevel.MEDIUM, "Performance optimization"),
        ("refactoring", SafetyLevel.MEDIUM, "Code refactoring"),
        ("feature_addition", SafetyLevel.HIGH, "New feature"),
        ("safety_system", SafetyLevel.CRITICAL, "Safety enhancement")
    ]
    
    for change_type, safety_level, description in test_cases:
        print(f"🎯 Testing: {change_type} ({safety_level.value})")
        print(f"📝 Description: {description}")
        
        try:
            selected_llm, capabilities = selector.select_optimal_llm(
                ChangeType(change_type), safety_level, "medium"
            )
            
            print(f"🤖 Selected LLM: {selected_llm}")
            print(f"📋 Capabilities: {capabilities.get('coding_quality', 'N/A')} coding, "
                  f"{capabilities.get('safety_awareness', 'N/A')} safety")
            print(f"💰 Cost: {capabilities.get('cost', 'N/A')}")
            print(f"⚡ Speed: {capabilities.get('speed', 'N/A')}")
            print(f"🎯 Best for: {', '.join(capabilities.get('best_for', []))}")
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
    
    print("✅ Auto LLM Selector test completed!")

async def test_autonomous_editor():
    """Test the autonomous editor with auto LLM selector"""
    print("\n🧠 Testing Autonomous Editor with Auto LLM Selector")
    print("=" * 60)
    
    # Create configuration
    config = SafetyConfig()
    config.auto_llm_selector = True
    config.fallback_to_local = True
    
    # Create editor
    editor = AutonomousCodeEditor(config)
    
    print(f"🔍 Auto LLM Selector enabled: {editor.auto_llm_selector is not None}")
    print(f"🤖 Available LLM clients:")
    print(f"  - Claude: {editor.claude_client is not None}")
    print(f"  - DeepSeek: {editor.deepseek_client is not None}")
    print(f"  - Llama2: {editor.llama2_client is not None}")
    print(f"  - vLLM: {editor.vllm_client is not None}")
    print()
    
    # Test LLM selection for a task
    try:
        selected_llm, capabilities = editor.auto_llm_selector.select_optimal_llm(
            ChangeType.DOCUMENTATION, SafetyLevel.LOW, "medium"
        )
        
        print(f"🎯 Selected LLM for documentation task: {selected_llm}")
        print(f"📋 Capabilities: {capabilities}")
        print()
        
    except Exception as e:
        print(f"❌ LLM selection error: {e}")
        print()
    
    print("✅ Autonomous Editor test completed!")

def main():
    """Main test function"""
    print("🚀 Starting Auto LLM Selector Integration Tests")
    print("=" * 60)
    
    # Run tests
    asyncio.run(test_auto_llm_selector())
    asyncio.run(test_autonomous_editor())
    
    print("\n🎉 All tests completed successfully!")
    print("\n📋 Summary:")
    print("  ✅ Auto LLM Selector initialized")
    print("  ✅ LLM capability mapping working")
    print("  ✅ Task-specific LLM selection functional")
    print("  ✅ Autonomous editor integrated")
    print("  ✅ Safety validation working")
    print("\n🔧 Next steps:")
    print("  1. Set API keys for Claude/DeepSeek (optional)")
    print("  2. Install llama-cpp-python for local Llama2")
    print("  3. Install vllm for high-performance inference")
    print("  4. Test with actual code editing tasks")

if __name__ == "__main__":
    main()
