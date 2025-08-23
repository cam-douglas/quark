#!/usr/bin/env python3
"""
Sequential Test Runner - Forces tests to run one at a time with live streaming.
"""

import os
import sys
import time
import importlib.util
import pytest

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from testing.visualizations.visual_utils import live_series, start_live_server


def run_test_sequentially(test_file, test_name=None):
    """Run a single test with live streaming, ensuring only one test is active."""
    print(f"🧪 Running test: {test_name or 'all tests'}")
    
    # Start live server
    start_live_server()
    time.sleep(1)  # Allow server to start
    
    # Import the test module
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    # Find all test functions
    test_functions = []
    for attr_name in dir(test_module):
        if attr_name.startswith('test_'):
            test_functions.append(attr_name)
    
    if test_name:
        # Run specific test
        test_functions = [test_name] if test_name in test_functions else []
    
    results = []
    
    for i, test_func_name in enumerate(test_functions):
        print(f"\n📋 Test {i+1}/{len(test_functions)}: {test_func_name}")
        
        # Clear previous test data
        live_series("sequential_clear_previous", "clear", 0)
        
        # Start current test
        live_series("sequential_test_start", {
            "name": test_func_name,
            "number": i + 1,
            "total": len(test_functions),
            "status": "running"
        }, 0)
        
        start_time = time.time()
        
        try:
            # Run the test
            test_func = getattr(test_module, test_func_name)
            test_func()
            
            # Test passed
            duration = time.time() - start_time
            live_series("sequential_test_result", {
                "name": test_func_name,
                "outcome": "PASSED",
                "duration": duration,
                "status": "completed"
            }, 0)
            
            results.append({"name": test_func_name, "status": "PASSED", "duration": duration})
            print(f"✅ {test_func_name} PASSED ({duration:.4f}s)")
            
        except Exception as e:
            # Test failed
            duration = time.time() - start_time
            live_series("sequential_test_result", {
                "name": test_func_name,
                "outcome": "FAILED",
                "duration": duration,
                "error": str(e)[:200],
                "status": "completed"
            }, 0)
            
            results.append({"name": test_func_name, "status": "FAILED", "duration": duration, "error": str(e)})
            print(f"❌ {test_func_name} FAILED ({duration:.4f}s): {e}")
        
        # Wait a moment to show the result
        time.sleep(1)
    
    # Final summary
    live_series("sequential_summary", {
        "total_tests": len(test_functions),
        "passed": len([r for r in results if r["status"] == "PASSED"]),
        "failed": len([r for r in results if r["status"] == "FAILED"]),
        "status": "completed"
    }, 0)
    
    print(f"\n📊 Sequential Test Summary:")
    print(f"   Total: {len(test_functions)}")
    print(f"   Passed: {len([r for r in results if r['status'] == 'PASSED'])}")
    print(f"   Failed: {len([r for r in results if r['status'] == 'FAILED'])}")
    
    return results


if __name__ == "__main__":
    # Test with the phase1 prototypes
    test_file = "testing/testing_frameworks/tests/test_phase1_prototypes.py"
    
    if os.path.exists(test_file):
        print("🚀 Sequential Test Runner with Live Streaming")
        print("=" * 50)
        results = run_test_sequentially(test_file)
        print(f"\n✅ Sequential testing completed: {len(results)} tests")
    else:
        print(f"❌ Test file not found: {test_file}")
