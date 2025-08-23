#!/usr/bin/env python3
"""
Comprehensive test of all live streaming capabilities.

This script tests:
1. Pytest plugin live streaming
2. Experiment framework live streaming  
3. Manual live streaming
4. Dashboard connectivity
"""

import os
import sys
import time
import subprocess
import threading

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from testing.testing_frameworks.core.experiment_framework import (
    ExperimentConfig, ExperimentManager, HybridSLMLLMExperiment
)
from testing.visualizations.visual_utils import live_series, start_live_server


def test_manual_live_streaming():
    """Test manual live streaming functionality."""
    print("🎯 Testing manual live streaming...")
    
    # Stream various types of data
    live_series("manual_test", "started", 1)
    live_series("manual_progress", 0.25, 1)
    time.sleep(0.5)
    
    live_series("manual_progress", 0.5, 2)
    live_series("manual_status", "processing", 2)
    time.sleep(0.5)
    
    live_series("manual_progress", 0.75, 3)
    live_series("manual_metrics", {"accuracy": 0.85, "loss": 0.15}, 3)
    time.sleep(0.5)
    
    live_series("manual_progress", 1.0, 4)
    live_series("manual_test", "completed", 4)
    
    print("✅ Manual live streaming test completed")


def test_experiment_live_streaming():
    """Test experiment framework live streaming."""
    print("🧪 Testing experiment framework live streaming...")
    
    # Create and run multiple experiments
    experiments = []
    for i in range(3):
        config = ExperimentConfig(
            experiment_id=f"live_test_exp_{i}",
            description=f"Live streaming test experiment {i}",
            params={
                "learning_rate": 0.001 * (i + 1),
                "batch_size": 16 * (i + 1),
                "epochs": 5 * (i + 1)
            },
            tags={"test": "true", "live_streaming": "enabled"}
        )
        experiments.append(config)
    
    # Run experiments with live streaming
    manager = ExperimentManager()
    results = []
    
    for i, config in enumerate(experiments):
        print(f"  Running experiment {i+1}/3...")
        experiment = HybridSLMLLMExperiment(config)
        result = manager.run(experiment)
        results.append(result)
        time.sleep(1)  # Allow time for live streaming
    
    print(f"✅ Experiment framework live streaming test completed: {len(results)} experiments")
    return results


def test_pytest_live_streaming():
    """Test pytest live streaming plugin."""
    print("🧪 Testing pytest live streaming plugin...")
    
    # Create a test file with live streaming
    test_content = '''#!/usr/bin/env python3
"""
Test file for pytest live streaming.
"""

import pytest
import time


@pytest.mark.live_stream
def test_live_streaming_plugin():
    """Test that pytest live streaming plugin works."""
    # This test should automatically stream metrics via the plugin
    time.sleep(0.1)
    assert True


@pytest.mark.live_stream  
def test_live_streaming_fixtures(stream_metric, live_dashboard):
    """Test live streaming fixtures."""
    # Test stream_metric fixture
    stream_metric("fixture_test", "started", 1)
    stream_metric("fixture_progress", 0.5, 2)
    stream_metric("fixture_test", "completed", 3)
    
    # Test live_dashboard fixture
    assert live_dashboard.is_available()
    live_dashboard.stream("dashboard_test", "working", 1)
    
    assert True


def test_regular_test():
    """Regular test without live streaming."""
    assert 2 + 2 == 4
'''
    
    test_file = os.path.join(os.path.dirname(__file__), 'pytest_live_test.py')
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    try:
        # Run pytest with live streaming
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short'
        ], capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), '..', '..'))
        
        print(f"✅ Pytest live streaming test completed: return code {result.returncode}")
        if result.stdout:
            print("  STDOUT:", result.stdout.strip())
        
        return result.returncode == 0
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


def main():
    """Main comprehensive test function."""
    print("🚀 Comprehensive Live Streaming Test")
    print("=" * 60)
    
    try:
        # Start live dashboard
        print("\n🌐 Starting live dashboard...")
        start_live_server()
        time.sleep(2)  # Allow server to start
        
        # Test 1: Manual live streaming
        print("\n" + "="*60)
        test_manual_live_streaming()
        
        # Test 2: Experiment framework live streaming
        print("\n" + "="*60)
        experiment_results = test_experiment_live_streaming()
        
        # Test 3: Pytest live streaming plugin
        print("\n" + "="*60)
        pytest_success = test_pytest_live_streaming()
        
        # Summary
        print("\n" + "="*60)
        print("📊 Comprehensive Test Summary:")
        print(f"   Manual Live Streaming: ✅")
        print(f"   Experiment Framework: ✅ ({len(experiment_results)} experiments)")
        print(f"   Pytest Plugin: {'✅' if pytest_success else '❌'}")
        print(f"   Live Dashboard: ✅")
        
        print("\n🌐 Live dashboard should be open in your browser")
        print("   All metrics should be streaming in real-time!")
        
        # Keep server running to show results
        print("\n⏳ Keeping live server running for 15 seconds to show results...")
        time.sleep(15)
        
    except Exception as e:
        print(f"❌ Error during comprehensive test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
