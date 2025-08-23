#!/usr/bin/env python3
"""
Demo script to test live streaming capabilities of both pytest and experiment framework.

This script demonstrates:
1. Live streaming from experiments via ExperimentManager
2. Live streaming from tests via pytest plugin
3. Real-time dashboard updates
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


def run_experiment_with_live_streaming():
    """Run an experiment that streams live metrics."""
    print("🧪 Running experiment with live streaming...")
    
    # Create experiment config
    config = ExperimentConfig(
        experiment_id="live_streaming_demo",
        description="Demonstration of live streaming capabilities",
        params={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        },
        tags={"demo": "true", "live_streaming": "enabled"}
    )
    
    # Create and run experiment
    experiment = HybridSLMLLMExperiment(config)
    manager = ExperimentManager()
    
    # This will automatically start live streaming
    result = manager.run(experiment)
    
    print(f"✅ Experiment completed: {result.success}")
    print(f"📊 Metrics: {result.metrics}")
    
    return result


def run_pytest_with_live_streaming():
    """Run pytest tests that stream live metrics."""
    print("🧪 Running pytest with live streaming...")
    
    # Run pytest with our live streaming plugin
    test_file = os.path.join(os.path.dirname(__file__), 'test_live_streaming.py')
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short'
        ], capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), '..', '..'))
        
        print("✅ Pytest completed")
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return False


def create_test_file():
    """Create a test file that uses live streaming fixtures."""
    test_content = '''#!/usr/bin/env python3
"""
Test file that demonstrates live streaming capabilities.
"""

import pytest
import time
import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


@pytest.mark.live_stream
def test_basic_live_streaming(stream_metric):
    """Test basic live streaming functionality."""
    stream_metric("test_progress", 0.25, 1)
    time.sleep(0.1)
    stream_metric("test_progress", 0.5, 2)
    time.sleep(0.1)
    stream_metric("test_progress", 0.75, 3)
    time.sleep(0.1)
    stream_metric("test_progress", 1.0, 4)
    
    assert True


@pytest.mark.live_stream
def test_live_dashboard_fixture(live_dashboard):
    """Test live dashboard fixture."""
    assert live_dashboard.is_available()
    
    live_dashboard.stream("dashboard_test", "started", 1)
    time.sleep(0.1)
    live_dashboard.stream("dashboard_test", "processing", 2)
    time.sleep(0.1)
    live_dashboard.stream("dashboard_test", "completed", 3)
    
    assert True


@pytest.mark.live_stream
def test_metric_streaming(stream_metric):
    """Test streaming various types of metrics."""
    # Numeric metrics
    stream_metric("accuracy", 0.85, 1)
    stream_metric("loss", 0.15, 1)
    
    # String metrics
    stream_metric("status", "training", 1)
    stream_metric("phase", "validation", 2)
    
    # Boolean metrics
    stream_metric("converged", True, 3)
    
    assert True


def test_without_live_streaming():
    """Test that doesn't use live streaming (should still work)."""
    # This test should run normally without live streaming
    assert 2 + 2 == 4
'''
    
    test_file_path = os.path.join(os.path.dirname(__file__), 'test_live_streaming.py')
    with open(test_file_path, 'w') as f:
        f.write(test_content)
    
    print(f"📝 Created test file: {test_file_path}")
    return test_file_path


def main():
    """Main demo function."""
    print("🚀 Live Streaming Demo")
    print("=" * 50)
    
    # Create test file
    test_file = create_test_file()
    
    try:
        # Start live dashboard
        print("\n🌐 Starting live dashboard...")
        from testing.visualizations.visual_utils import start_live_server
        start_live_server()
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Run experiment with live streaming
        print("\n" + "="*50)
        experiment_result = run_experiment_with_live_streaming()
        
        # Wait for experiment to complete
        time.sleep(3)
        
        # Run pytest with live streaming
        print("\n" + "="*50)
        pytest_success = run_pytest_with_live_streaming()
        
        # Wait for tests to complete
        time.sleep(3)
        
        # Summary
        print("\n" + "="*50)
        print("📊 Demo Summary:")
        print(f"   Experiment: {'✅' if experiment_result.success else '❌'}")
        print(f"   Pytest: {'✅' if pytest_success else '❌'}")
        print("\n🌐 Live dashboard should be open in your browser")
        print("   Check the dashboard for real-time metrics from both experiments and tests!")
        
        # Keep server running for a bit to show results
        print("\n⏳ Keeping live server running for 10 seconds to show results...")
        time.sleep(10)
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"🧹 Cleaned up test file: {test_file}")


if __name__ == "__main__":
    main()
