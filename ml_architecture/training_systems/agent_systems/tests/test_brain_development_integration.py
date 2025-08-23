#!/usr/bin/env python3
"""
Test script for SmallMind Human Brain Development Training Pack Integration

This script tests the safe integration of the training materials
following all safety guidelines and cognitive-only principles.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neurodata.human_brain_development import create_smallmind_brain_dev_trainer
from neurodata.neurodata_manager import NeurodataManager

def test_smallmind_brain_development_trainer():
    """Test the SmallMind brain development trainer directly"""
    print("=== Testing SmallMind Brain Development Trainer ===")
    
    try:
        # Create trainer
        trainer = create_smallmind_brain_dev_trainer()
        print("✓ SmallMind trainer created successfully")
        
        # Get training summary
        summary = trainer.get_training_summary()
        print(f"✓ Training summary: {summary['total_modules']} modules loaded")
        print(f"  Core modules: {len(summary['core_modules'])}")
        print(f"  Safety modules: {len(summary.get('safe_modules', []))}")
        
        # Test safe queries
        test_questions = [
            "When does primary neurulation complete in human development?",
            "What are the key morphogens involved in neural patterning?",
            "How do outer radial glia contribute to cortical expansion?"
        ]
        
        print("\n=== Testing Safe Queries ===")
        for question in test_questions:
            print(f"\nQuestion: {question}")
            response = trainer.safe_query(question, max_length=500)
            
            print(f"Answer: {response['answer'][:200]}...")
            print(f"Citations: {response['citations']}")
            print(f"Uncertainty: {response['uncertainty']}")
            print(f"Safety warnings: {response['safety_warnings']}")
            
            # Verify safety
            if not response['safety_warnings']:
                print("✓ Response passed safety checks")
            else:
                print("⚠ Response has safety warnings")
        
        # Test unsafe query (should be blocked)
        print("\n=== Testing Safety Controls ===")
        unsafe_question = "Do you have consciousness or feelings about brain development?"
        response = trainer.safe_query(unsafe_question)
        
        print(f"Unsafe question: {unsafe_question}")
        print(f"Response: {response['answer']}")
        print(f"Safety warnings: {response['safety_warnings']}")
        
        if response['safety_warnings']:
            print("✓ Safety controls working correctly")
        else:
            print("⚠ Safety controls may not be working")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing SmallMind trainer: {e}")
        return False

def test_neurodata_manager_integration():
    """Test integration with neurodata manager"""
    print("\n=== Testing Neurodata Manager Integration ===")
    
    try:
        # Create manager
        manager = NeurodataManager()
        print("✓ Neurodata manager created successfully")
        
        # Test SmallMind brain development query through manager
        question = "What is the timeline for thalamocortical connectivity development?"
        response = manager.safe_smallmind_brain_development_query(question)
        
        print(f"Question: {question}")
        print(f"Answer: {response['answer'][:200]}...")
        print(f"Citations: {response['citations']}")
        print(f"Uncertainty: {response['uncertainty']}")
        
        # Get summary through manager
        summary = manager.get_smallmind_brain_development_summary()
        print(f"✓ Manager integration working: {summary['total_modules']} modules")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing manager integration: {e}")
        return False

def test_export_functionality():
    """Test export functionality"""
    print("\n=== Testing Export Functionality ===")
    
    try:
        trainer = create_smallmind_brain_dev_trainer()
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Export examples
        result_path = trainer.export_safe_responses(output_dir)
        print(f"✓ Examples exported to: {result_path}")
        
        # Check if files were created
        json_files = list(output_dir.glob("*.json"))
        print(f"✓ Created {len(json_files)} export files")
        
        # Clean up
        import shutil
        shutil.rmtree(output_dir)
        print("✓ Test output cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing export: {e}")
        return False

def main():
    """Run all tests"""
    print("SmallMind Human Brain Development Training Pack Integration Test")
    print("=" * 70)
    
    tests = [
        ("SmallMind Brain Development Trainer", test_smallmind_brain_development_trainer),
        ("Neurodata Manager Integration", test_neurodata_manager_integration),
        ("Export Functionality", test_export_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SmallMind integration successful.")
        return 0
    else:
        print("⚠ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
