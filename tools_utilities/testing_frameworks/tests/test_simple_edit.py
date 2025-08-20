#!/usr/bin/env python3
"""
Simple test for autonomous editing functionality
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.autonomous_code_editor import (
    AutonomousCodeEditor, SafetyConfig, SafetyLevel, ChangeType
)

async def test_simple_edit():
    """Test simple autonomous editing"""
    print("🧠 Testing Simple Autonomous Editing")
    print("=" * 50)
    
    try:
        # Create configuration
        config = SafetyConfig()
        config.auto_llm_selector = True
        config.fallback_to_local = True
        
        # Create editor
        editor = AutonomousCodeEditor(config)
        print("✅ Editor created successfully")
        
        # Test file path
        test_file = "demo_test_file.py"
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return
        
        print(f"✅ Test file found: {test_file}")
        
        # Test pre-validation
        print("🔍 Testing pre-validation...")
        validation_result = await editor.safety_validator.validate_request(
            test_file, "Add docstrings", ChangeType.DOCUMENTATION, SafetyLevel.LOW
        )
        print(f"✅ Pre-validation result: {validation_result}")
        
        # Test LLM selection
        print("🤖 Testing LLM selection...")
        selected_llm, capabilities = editor.auto_llm_selector.select_optimal_llm(
            ChangeType.DOCUMENTATION, SafetyLevel.LOW, "medium"
        )
        print(f"✅ Selected LLM: {selected_llm}")
        
        # Test code generation
        print("✍️ Testing code generation...")
        code_changes = await editor._generate_code_changes(
            test_file, "Add docstrings", ChangeType.DOCUMENTATION, selected_llm, capabilities
        )
        
        if code_changes:
            print("✅ Code changes generated successfully")
            print(f"📝 Changes: {len(code_changes.new_content)} characters")
        else:
            print("❌ Code generation failed")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    asyncio.run(test_simple_edit())

if __name__ == "__main__":
    main()
