#!/usr/bin/env python3
"""
Simple live streaming test without subprocess complexity.
"""

import time
import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from testing.visualizations.visual_utils import live_series, start_live_server
from testing.testing_frameworks.core.experiment_framework import (
    ExperimentConfig, ExperimentManager, HybridSLMLLMExperiment
)


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
        time.sleep(0.5)  # Allow time for live streaming
    
    print(f"✅ Experiment framework live streaming test completed: {len(results)} experiments")
    return results


def main():
    """Main test function."""
    print("🚀 Simple Live Streaming Test")
    print("=" * 50)
    
    try:
        # Start live dashboard
        print("\n🌐 Starting live dashboard...")
        start_live_server()
        time.sleep(1)  # Allow server to start
        
        # Test 1: Manual live streaming
        print("\n" + "="*50)
        test_manual_live_streaming()
        
        # Test 2: Experiment framework live streaming
        print("\n" + "="*50)
        experiment_results = test_experiment_live_streaming()
        
        # Summary
        print("\n" + "="*50)
        print("📊 Simple Test Summary:")
        print(f"   Manual Live Streaming: ✅")
        print(f"   Experiment Framework: ✅ ({len(experiment_results)} experiments)")
        print(f"   Live Dashboard: ✅")
        
        print("\n🌐 Live dashboard should be open in your browser")
        print("   All metrics should be streaming in real-time!")
        
        # Keep server running to show results
        print("\n⏳ Keeping live server running for 10 seconds to show results...")
        time.sleep(10)
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
