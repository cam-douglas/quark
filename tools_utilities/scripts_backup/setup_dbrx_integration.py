#!/usr/bin/env python3
"""
DBRX Integration Setup Script
============================

Purpose: Setup and configure DBRX Instruct model integration with brain simulation
Inputs: HuggingFace token, configuration parameters
Outputs: Configured DBRX integration, test results, deployment ready system
Seeds: Configuration parameters, test scenarios
Dependencies: dbrx_brain_integration, brain_launcher_v4, transformers

Automated setup script for integrating DBRX Instruct with brain simulation.
Handles model download, configuration, testing, and integration validation.
"""

import os, sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.dbrx_brain_integration import DBRXBrainIntegration, DBRXConfig
from core.brain_launcher_v4 import NeuralEnhancedBrain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_dbrx_integration(
    hf_token: str,
    config: Optional[DBRXConfig] = None,
    test_mode: bool = True
) -> Dict[str, Any]:
    """Setup DBRX integration with brain simulation"""
    
    logger.info("🚀 Setting up DBRX Brain Integration...")
    
    # Create configuration
    if config is None:
        config = DBRXConfig()
    
    # Create integration
    dbrx_integration = DBRXBrainIntegration(config)
    
    # Initialize model
    logger.info("📥 Initializing DBRX Instruct model...")
    success = dbrx_integration.initialize_model(hf_token=hf_token)
    
    if not success:
        return {
            "success": False,
            "error": "Failed to initialize DBRX model",
            "integration": None
        }
    
    # Test mode - create mock brain simulation
    if test_mode:
        logger.info("🧪 Running in test mode with mock brain simulation...")
        
        # Create mock brain simulation
        mock_brain = create_mock_brain_simulation()
        
        # Connect to brain simulation
        dbrx_integration.connect_brain_simulation(mock_brain)
        
        # Run integration test
        test_results = run_integration_test(dbrx_integration)
        
        return {
            "success": True,
            "integration": dbrx_integration,
            "test_results": test_results,
            "config": config
        }
    
    return {
        "success": True,
        "integration": dbrx_integration,
        "config": config
    }

def create_mock_brain_simulation() -> Any:
    """Create a mock brain simulation for testing"""
    
    class MockBrainSimulation:
        def __init__(self):
            self.step_count = 0
        
        def get_neural_summary(self) -> Dict[str, Any]:
            """Return mock neural summary"""
            self.step_count += 1
            
            # Simulate realistic brain metrics
            import random
            import math
            
            # Oscillating consciousness level
            consciousness_level = 0.3 + 0.4 * math.sin(self.step_count * 0.1)
            
            return {
                'firing_rates': {
                    'pfc': 15.0 + random.uniform(-2, 2),
                    'bg': 8.0 + random.uniform(-1, 1),
                    'thalamus': 12.0 + random.uniform(-1.5, 1.5),
                    'hippocampus': 6.0 + random.uniform(-0.5, 0.5)
                },
                'loop_stability': 0.7 + random.uniform(-0.1, 0.1),
                'synchrony': 0.6 + random.uniform(-0.15, 0.15),
                'consciousness_level': max(0.0, min(1.0, consciousness_level)),
                'module_states': {
                    'pfc': {'status': 'active', 'energy': 0.8},
                    'bg': {'status': 'active', 'energy': 0.6},
                    'thalamus': {'status': 'active', 'energy': 0.7},
                    'hippocampus': {'status': 'active', 'energy': 0.5}
                },
                'energy_consumption': {
                    'total': 45.0 + random.uniform(-5, 5),
                    'pfc': 20.0,
                    'bg': 12.0,
                    'thalamus': 8.0,
                    'hippocampus': 5.0
                }
            }
    
    return MockBrainSimulation()

def run_integration_test(dbrx_integration: DBRXBrainIntegration) -> Dict[str, Any]:
    """Run integration test with DBRX"""
    
    logger.info("🧪 Running DBRX integration test...")
    
    test_results = {
        "consciousness_analyses": [],
        "neural_interpretations": [],
        "performance_metrics": {},
        "test_duration": 0
    }
    
    import time
    start_time = time.time()
    
    try:
        # Start integration
        dbrx_integration.start_integration()
        
        # Run for 30 seconds to collect data
        import time
        for i in range(30):  # 30 seconds
            time.sleep(1)
            
            # Get current report
            report = dbrx_integration.get_integration_report()
            
            # Store consciousness analyses
            if dbrx_integration.last_analysis:
                test_results["consciousness_analyses"].append(
                    dbrx_integration.last_analysis
                )
            
            # Log progress
            if i % 5 == 0:
                logger.info(f"⏱️  Test progress: {i+1}/30 seconds")
        
        # Stop integration
        dbrx_integration.stop_integration()
        
        # Get final metrics
        final_report = dbrx_integration.get_integration_report()
        test_results["performance_metrics"] = final_report["performance_metrics"]
        test_results["test_duration"] = time.time() - start_time
        
        logger.info("✅ Integration test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        test_results["error"] = str(e)
    
    return test_results

def save_integration_config(
    dbrx_integration: DBRXBrainIntegration,
    config: DBRXConfig,
    output_dir: str = "config"
) -> str:
    """Save integration configuration"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    config_data = {
        "model_name": config.model_name,
        "device_map": config.device_map,
        "torch_dtype": config.torch_dtype,
        "max_length": config.max_length,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "consciousness_analysis_interval": config.consciousness_analysis_interval,
        "neural_reasoning_threshold": config.neural_reasoning_threshold,
        "enable_consciousness_feedback": config.enable_consciousness_feedback,
        "enable_neural_interpretation": config.enable_neural_interpretation,
        "use_flash_attention": config.use_flash_attention,
        "enable_gradient_checkpointing": config.enable_gradient_checkpointing,
        "memory_efficient_mode": config.memory_efficient_mode,
        "setup_timestamp": str(datetime.now()),
        "version": "1.0"
    }
    
    config_file = os.path.join(output_dir, "dbrx_integration_config.json")
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    logger.info(f"💾 Configuration saved to: {config_file}")
    return config_file

def create_usage_example(dbrx_integration: DBRXBrainIntegration) -> str:
    """Create usage example script"""
    
    example_script = '''#!/usr/bin/env python3
"""
DBRX Brain Integration Usage Example
==================================

Example of how to use DBRX integration with brain simulation.
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.dbrx_brain_integration import DBRXBrainIntegration, DBRXConfig
from core.brain_launcher_v4 import NeuralEnhancedBrain

def main():
    # Create DBRX configuration
    config = DBRXConfig(
        temperature=0.7,
        consciousness_analysis_interval=10,
        enable_consciousness_feedback=True
    )
    
    # Create integration
    dbrx_integration = DBRXBrainIntegration(config)
    
    # Initialize model (replace with your token)
    success = dbrx_integration.initialize_model(hf_token="your_token_here")
    if not success:
        print("❌ Failed to initialize DBRX model")
        return
    
    # Load brain simulation
    brain = NeuralEnhancedBrain("src/config/connectome_v3.yaml", stage="F")
    
    # Connect to brain simulation
    dbrx_integration.connect_brain_simulation(brain)
    
    # Start integration
    dbrx_integration.start_integration()
    
    # Run brain simulation
    for step in range(100):
        brain.step()
        
        # Get integration report every 10 steps
        if step % 10 == 0:
            report = dbrx_integration.get_integration_report()
            print(f"Step {step}: {report['performance_metrics']['total_analyses']} analyses")
    
    # Stop integration
    dbrx_integration.stop_integration()
    
    # Final report
    final_report = dbrx_integration.get_integration_report()
    print(f"Final Report: {final_report}")

if __name__ == "__main__":
    main()
'''
    
    example_file = "examples/dbrx_usage_example.py"
    os.makedirs("examples", exist_ok=True)
    
    with open(example_file, 'w') as f:
        f.write(example_script)
    
    logger.info(f"📝 Usage example created: {example_file}")
    return example_file

def main():
    """Main setup function"""
    
    parser = argparse.ArgumentParser(description="Setup DBRX Brain Integration")
    parser.add_argument("--hf-token", required=True, help="HuggingFace access token")
    parser.add_argument("--test-mode", action="store_true", default=True, help="Run in test mode")
    parser.add_argument("--output-dir", default="config", help="Output directory for config")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--analysis-interval", type=int, default=10, help="Consciousness analysis interval")
    
    args = parser.parse_args()
    
    # Create configuration
    config = DBRXConfig(
        temperature=args.temperature,
        consciousness_analysis_interval=args.analysis_interval
    )
    
    # Setup integration
    result = setup_dbrx_integration(
        hf_token=args.hf_token,
        config=config,
        test_mode=args.test_mode
    )
    
    if result["success"]:
        logger.info("✅ DBRX integration setup completed successfully!")
        
        # Save configuration
        config_file = save_integration_config(
            result["integration"],
            result["config"],
            args.output_dir
        )
        
        # Create usage example
        example_file = create_usage_example(result["integration"])
        
        # Print test results if available
        if "test_results" in result:
            test_results = result["test_results"]
            logger.info(f"📊 Test Results:")
            logger.info(f"   - Test Duration: {test_results['test_duration']:.2f} seconds")
            logger.info(f"   - Total Analyses: {test_results['performance_metrics'].get('total_analyses', 0)}")
            logger.info(f"   - Avg Generation Time: {test_results['performance_metrics'].get('average_generation_time', 0):.3f}s")
            logger.info(f"   - Consciousness Insights: {test_results['performance_metrics'].get('consciousness_insights', 0)}")
        
        logger.info(f"📁 Configuration saved to: {config_file}")
        logger.info(f"📝 Usage example created: {example_file}")
        
        return True
    else:
        logger.error(f"❌ DBRX integration setup failed: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    from datetime import datetime
    success = main()
    sys.exit(0 if success else 1)

