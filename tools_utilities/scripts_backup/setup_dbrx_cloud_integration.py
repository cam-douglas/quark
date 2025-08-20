#!/usr/bin/env python3
"""
DBRX Cloud Integration Setup Script
==================================

Purpose: Setup cloud-optimized DBRX integration with sparse usage for massive computational efficiency
Inputs: HuggingFace token, cloud configuration, resource limits
Outputs: Configured cloud-optimized DBRX integration, resource monitoring, cost controls
Seeds: Cloud configuration parameters, resource limits
Dependencies: dbrx_cloud_integration, cloud resource monitoring, cost optimization

Cloud-optimized setup script for DBRX Instruct integration with intelligent resource management,
cost controls, and sparse usage to minimize computational costs while maximizing research value.
"""

import os, sys
import json
import argparse
import logging
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.dbrx_cloud_integration import DBRXCloudIntegration, DBRXCloudConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_system_resources() -> Dict[str, Any]:
    """Check if system has sufficient resources for DBRX"""
    
    logger.info("🔍 Checking system resources for DBRX...")
    
    # Check CPU
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Check memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    
    # Check GPU
    gpu_info = []
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info.append({
                'name': gpu.name,
                'memory_total': gpu.memoryTotal,
                'memory_free': gpu.memoryFree,
                'memory_used': gpu.memoryUsed,
                'load': gpu.load * 100 if gpu.load else 0
            })
    except Exception as e:
        logger.warning(f"Could not get GPU info: {e}")
    
    # Check disk space
    disk = psutil.disk_usage('/')
    disk_gb = disk.free / (1024**3)
    
    # Resource assessment
    resource_assessment = {
        'cpu': {
            'count': cpu_count,
            'usage_percent': cpu_percent,
            'sufficient': cpu_count >= 16
        },
        'memory': {
            'total_gb': round(memory_gb, 1),
            'available_gb': round(memory.available / (1024**3), 1),
            'sufficient': memory_gb >= 300
        },
        'gpu': {
            'count': len(gpu_info),
            'info': gpu_info,
            'sufficient': any(gpu['memory_total'] >= 264 for gpu in gpu_info)
        },
        'disk': {
            'free_gb': round(disk_gb, 1),
            'sufficient': disk_gb >= 500
        }
    }
    
    # Overall assessment
    resource_assessment['overall_sufficient'] = all([
        resource_assessment['cpu']['sufficient'],
        resource_assessment['memory']['sufficient'],
        resource_assessment['gpu']['sufficient'],
        resource_assessment['disk']['sufficient']
    ])
    
    return resource_assessment

def setup_dbrx_cloud_integration(
    hf_token: str,
    config: Optional[DBRXCloudConfig] = None,
    test_mode: bool = True,
    resource_check: bool = True
) -> Dict[str, Any]:
    """Setup cloud-optimized DBRX integration"""
    
    logger.info("☁️ Setting up DBRX Cloud Integration...")
    
    # Check system resources
    if resource_check:
        resource_assessment = check_system_resources()
        
        if not resource_assessment['overall_sufficient']:
            logger.warning("⚠️ System resources may be insufficient for DBRX:")
            if not resource_assessment['cpu']['sufficient']:
                logger.warning(f"   CPU: {resource_assessment['cpu']['count']} cores (need 16+)")
            if not resource_assessment['memory']['sufficient']:
                logger.warning(f"   Memory: {resource_assessment['memory']['total_gb']}GB (need 300GB+)")
            if not resource_assessment['gpu']['sufficient']:
                logger.warning(f"   GPU: No GPU with 264GB+ memory found")
            if not resource_assessment['disk']['sufficient']:
                logger.warning(f"   Disk: {resource_assessment['disk']['free_gb']}GB free (need 500GB+)")
            
            logger.warning("💡 Consider using cloud instances or reducing model precision")
        
        logger.info("📊 Resource Assessment:")
        logger.info(f"   CPU: {resource_assessment['cpu']['count']} cores ({resource_assessment['cpu']['usage_percent']:.1f}% usage)")
        logger.info(f"   Memory: {resource_assessment['memory']['total_gb']:.1f}GB total, {resource_assessment['memory']['available_gb']:.1f}GB available")
        logger.info(f"   GPU: {resource_assessment['gpu']['count']} GPUs found")
        logger.info(f"   Disk: {resource_assessment['disk']['free_gb']:.1f}GB free")
    
    # Create configuration
    if config is None:
        config = DBRXCloudConfig()
    
    # Adjust configuration based on resources
    if resource_check and not resource_assessment['overall_sufficient']:
        logger.info("🔧 Adjusting configuration for limited resources...")
        config.max_requests_per_hour = 5  # Very conservative
        config.analysis_cooldown_minutes = 60  # Longer cooldown
        config.cache_enabled = True  # Enable caching
        config.memory_efficient_mode = True
        config.enable_gradient_checkpointing = True
    
    # Create integration
    dbrx_integration = DBRXCloudIntegration(config)
    
    # Initialize model
    logger.info("📥 Initializing DBRX Instruct model with cloud optimizations...")
    success = dbrx_integration.initialize_model(hf_token=hf_token)
    
    if not success:
        return {
            "success": False,
            "error": "Failed to initialize DBRX model",
            "integration": None,
            "resource_assessment": resource_assessment if resource_check else None
        }
    
    # Test mode - create mock brain simulation
    if test_mode:
        logger.info("🧪 Running in test mode with mock brain simulation...")
        
        # Create mock brain simulation
        mock_brain = create_mock_brain_simulation()
        
        # Connect to brain simulation
        dbrx_integration.connect_brain_simulation(mock_brain)
        
        # Run sparse integration test
        test_results = run_sparse_integration_test(dbrx_integration)
        
        return {
            "success": True,
            "integration": dbrx_integration,
            "test_results": test_results,
            "config": config,
            "resource_assessment": resource_assessment if resource_check else None
        }
    
    return {
        "success": True,
        "integration": dbrx_integration,
        "config": config,
        "resource_assessment": resource_assessment if resource_check else None
    }

def create_mock_brain_simulation() -> Any:
    """Create a mock brain simulation for testing"""
    
    class MockBrainSimulation:
        def __init__(self):
            self.step_count = 0
        
        def get_neural_summary(self) -> Dict[str, Any]:
            """Return mock neural summary"""
            self.step_count += 1
            
            # Simulate realistic brain metrics with consciousness emergence
            import random
            import math
            
            # Oscillating consciousness level that builds over time
            base_consciousness = 0.2 + 0.3 * math.sin(self.step_count * 0.05)
            consciousness_level = min(1.0, base_consciousness + (self.step_count * 0.001))
            
            return {
                'firing_rates': {
                    'pfc': 15.0 + random.uniform(-2, 2) + (consciousness_level * 5),
                    'bg': 8.0 + random.uniform(-1, 1) + (consciousness_level * 3),
                    'thalamus': 12.0 + random.uniform(-1.5, 1.5) + (consciousness_level * 4),
                    'hippocampus': 6.0 + random.uniform(-0.5, 0.5) + (consciousness_level * 2)
                },
                'loop_stability': 0.6 + random.uniform(-0.1, 0.1) + (consciousness_level * 0.3),
                'synchrony': 0.5 + random.uniform(-0.15, 0.15) + (consciousness_level * 0.4),
                'consciousness_level': consciousness_level,
                'module_states': {
                    'pfc': {'status': 'active', 'energy': 0.7 + consciousness_level * 0.2},
                    'bg': {'status': 'active', 'energy': 0.5 + consciousness_level * 0.3},
                    'thalamus': {'status': 'active', 'energy': 0.6 + consciousness_level * 0.2},
                    'hippocampus': {'status': 'active', 'energy': 0.4 + consciousness_level * 0.2}
                },
                'energy_consumption': {
                    'total': 40.0 + random.uniform(-5, 5) + (consciousness_level * 20),
                    'pfc': 18.0 + consciousness_level * 8,
                    'bg': 10.0 + consciousness_level * 5,
                    'thalamus': 7.0 + consciousness_level * 4,
                    'hippocampus': 5.0 + consciousness_level * 3
                }
            }
    
    return MockBrainSimulation()

def run_sparse_integration_test(dbrx_integration: DBRXCloudIntegration) -> Dict[str, Any]:
    """Run sparse integration test with DBRX"""
    
    logger.info("🧪 Running sparse DBRX integration test...")
    
    test_results = {
        "consciousness_analyses": [],
        "cache_performance": [],
        "usage_tracking": [],
        "performance_metrics": {},
        "test_duration": 0
    }
    
    import time
    start_time = time.time()
    
    try:
        # Run for 60 seconds to test sparse usage
        for i in range(60):  # 60 seconds
            time.sleep(1)
            
            # Get brain state
            brain_state = dbrx_integration.brain_simulation.get_neural_summary()
            
            # Analyze consciousness (will be sparse due to limits)
            analysis = dbrx_integration.analyze_consciousness_sparse(brain_state)
            
            # Store results
            if not analysis.get('skipped', False):
                test_results["consciousness_analyses"].append(analysis)
            
            # Get current report every 10 seconds
            if i % 10 == 0:
                report = dbrx_integration.get_integration_report()
                test_results["cache_performance"].append(report['cache_stats'])
                test_results["usage_tracking"].append(report['usage_stats'])
                
                logger.info(f"⏱️  Test progress: {i+1}/60 seconds")
                logger.info(f"   Analyses: {report['performance_metrics']['total_analyses']}")
                logger.info(f"   Cache hits: {report['cache_stats']['hits']}")
                logger.info(f"   Requests remaining: {report['usage_stats']['requests_remaining']}")
        
        # Get final metrics
        final_report = dbrx_integration.get_integration_report()
        test_results["performance_metrics"] = final_report["performance_metrics"]
        test_results["test_duration"] = time.time() - start_time
        
        logger.info("✅ Sparse integration test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        test_results["error"] = str(e)
    
    return test_results

def save_cloud_integration_config(
    dbrx_integration: DBRXCloudIntegration,
    config: DBRXCloudConfig,
    resource_assessment: Optional[Dict[str, Any]] = None,
    output_dir: str = "config"
) -> str:
    """Save cloud integration configuration"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    config_data = {
        "model_name": config.model_name,
        "device_map": config.device_map,
        "torch_dtype": config.torch_dtype,
        "max_length": config.max_length,
        "temperature": config.temperature,
        "top_p": config.top_p,
        
        # Cloud optimization settings
        "cache_enabled": config.cache_enabled,
        "cache_duration_hours": config.cache_duration_hours,
        "max_requests_per_hour": config.max_requests_per_hour,
        "min_consciousness_threshold": config.min_consciousness_threshold,
        "analysis_cooldown_minutes": config.analysis_cooldown_minutes,
        
        # Performance optimization
        "use_flash_attention": config.use_flash_attention,
        "enable_gradient_checkpointing": config.enable_gradient_checkpointing,
        "memory_efficient_mode": config.memory_efficient_mode,
        "enable_model_offloading": config.enable_model_offloading,
        
        # Cloud-specific
        "cloud_provider": config.cloud_provider,
        "instance_type": config.instance_type,
        "enable_spot_instances": config.enable_spot_instances,
        "max_cost_per_hour": config.max_cost_per_hour,
        
        # Setup information
        "setup_timestamp": str(datetime.now()),
        "version": "1.0",
        "resource_assessment": resource_assessment
    }
    
    config_file = os.path.join(output_dir, "dbrx_cloud_integration_config.json")
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    logger.info(f"💾 Cloud configuration saved to: {config_file}")
    return config_file

def create_cloud_usage_example(dbrx_integration: DBRXCloudIntegration) -> str:
    """Create cloud-optimized usage example script"""
    
    example_script = '''#!/usr/bin/env python3
"""
DBRX Cloud Integration Usage Example
===================================

Example of how to use DBRX cloud integration with sparse usage for cost efficiency.
"""

import os, sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.dbrx_cloud_integration import DBRXCloudIntegration, DBRXCloudConfig
from core.brain_launcher_v4 import NeuralEnhancedBrain

def main():
    # Create cloud-optimized DBRX configuration
    config = DBRXCloudConfig(
        max_requests_per_hour=5,  # Very conservative
        min_consciousness_threshold=0.4,  # Only analyze significant consciousness
        analysis_cooldown_minutes=60,  # Long cooldown between analyses
        cache_enabled=True,  # Enable caching for efficiency
        memory_efficient_mode=True
    )
    
    # Create integration
    dbrx_integration = DBRXCloudIntegration(config)
    
    # Initialize model (replace with your token)
    success = dbrx_integration.initialize_model(hf_token="your_token_here")
    if not success:
        print("❌ Failed to initialize DBRX model")
        return
    
    # Load brain simulation
    brain = NeuralEnhancedBrain("src/config/connectome_v3.yaml", stage="F")
    
    # Connect to brain simulation
    dbrx_integration.connect_brain_simulation(brain)
    
    # Run brain simulation with sparse analysis
    print("🧠 Running brain simulation with sparse DBRX analysis...")
    
    for step in range(1000):  # Longer simulation
        brain.step()
        
        # Get brain state
        brain_state = brain.get_neural_summary()
        
        # Analyze consciousness (sparse due to limits)
        analysis = dbrx_integration.analyze_consciousness_sparse(brain_state)
        
        # Log results every 100 steps
        if step % 100 == 0:
            report = dbrx_integration.get_integration_report()
            print(f"Step {step}:")
            print(f"  Consciousness Level: {brain_state.get('consciousness_level', 0):.3f}")
            print(f"  Total Analyses: {report['performance_metrics']['total_analyses']}")
            print(f"  Cache Hits: {report['cache_stats']['hits']}")
            print(f"  Requests Remaining: {report['usage_stats']['requests_remaining']}")
            
            if not analysis.get('skipped', False):
                print(f"  Analysis: {analysis['parsed_analysis']['stability_assessment']}")
    
    # Final report
    final_report = dbrx_integration.get_integration_report()
    print(f"\\n📊 Final Report:")
    print(f"  Total Analyses: {final_report['performance_metrics']['total_analyses']}")
    print(f"  Cached Analyses: {final_report['performance_metrics']['cached_analyses']}")
    print(f"  Cloud Analyses: {final_report['performance_metrics']['cloud_analyses']}")
    print(f"  Cache Hit Rate: {final_report['cache_stats']['hit_rate']:.2%}")
    print(f"  Average Generation Time: {final_report['performance_metrics']['average_generation_time']:.3f}s")

if __name__ == "__main__":
    main()
'''
    
    example_file = "examples/dbrx_cloud_usage_example.py"
    os.makedirs("examples", exist_ok=True)
    
    with open(example_file, 'w') as f:
        f.write(example_script)
    
    logger.info(f"📝 Cloud usage example created: {example_file}")
    return example_file

def create_cost_monitoring_script() -> str:
    """Create cost monitoring script for cloud usage"""
    
    monitoring_script = '''#!/usr/bin/env python3
"""
DBRX Cloud Cost Monitoring
=========================

Monitor and control costs for DBRX cloud integration.
"""

import time
import json
from datetime import datetime, timedelta
from pathlib import Path

class DBRXCostMonitor:
    def __init__(self, max_daily_cost: float = 100.0):
        self.max_daily_cost = max_daily_cost
        self.cost_log_file = Path("logs/dbrx_costs.json")
        self.cost_log_file.parent.mkdir(exist_ok=True)
        
        # Load existing cost data
        self.cost_data = self._load_cost_data()
    
    def _load_cost_data(self) -> dict:
        """Load existing cost data"""
        if self.cost_log_file.exists():
            try:
                with open(self.cost_log_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"daily_costs": {}, "total_cost": 0.0}
    
    def _save_cost_data(self):
        """Save cost data"""
        with open(self.cost_log_file, 'w') as f:
            json.dump(self.cost_data, f, indent=2)
    
    def estimate_request_cost(self, generation_time: float, tokens_generated: int) -> float:
        """Estimate cost for a single request"""
        # Rough cost estimation (adjust based on your cloud provider)
        # Assuming $50/hour for g5.48xlarge instance
        hourly_rate = 50.0
        cost_per_second = hourly_rate / 3600
        
        estimated_cost = generation_time * cost_per_second
        return estimated_cost
    
    def record_request_cost(self, cost: float):
        """Record the cost of a request"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in self.cost_data["daily_costs"]:
            self.cost_data["daily_costs"][today] = 0.0
        
        self.cost_data["daily_costs"][today] += cost
        self.cost_data["total_cost"] += cost
        
        self._save_cost_data()
    
    def can_make_request(self) -> bool:
        """Check if a request can be made within cost limits"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_cost = self.cost_data["daily_costs"].get(today, 0.0)
        
        return today_cost < self.max_daily_cost
    
    def get_cost_summary(self) -> dict:
        """Get cost summary"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_cost = self.cost_data["daily_costs"].get(today, 0.0)
        
        return {
            "today_cost": today_cost,
            "max_daily_cost": self.max_daily_cost,
            "remaining_budget": max(0, self.max_daily_cost - today_cost),
            "total_cost": self.cost_data["total_cost"],
            "daily_costs": self.cost_data["daily_costs"]
        }

# Example usage
if __name__ == "__main__":
    monitor = DBRXCostMonitor(max_daily_cost=50.0)
    
    print("💰 DBRX Cost Monitor")
    print(f"Max daily cost: ${monitor.max_daily_cost}")
    
    summary = monitor.get_cost_summary()
    print(f"Today's cost: ${summary['today_cost']:.2f}")
    print(f"Remaining budget: ${summary['remaining_budget']:.2f}")
    print(f"Can make request: {monitor.can_make_request()}")
'''
    
    monitoring_file = "scripts/dbrx_cost_monitor.py"
    os.makedirs("scripts", exist_ok=True)
    
    with open(monitoring_file, 'w') as f:
        f.write(monitoring_script)
    
    logger.info(f"💰 Cost monitoring script created: {monitoring_file}")
    return monitoring_file

def main():
    """Main setup function"""
    
    parser = argparse.ArgumentParser(description="Setup DBRX Cloud Integration")
    parser.add_argument("--hf-token", required=True, help="HuggingFace access token")
    parser.add_argument("--test-mode", action="store_true", default=True, help="Run in test mode")
    parser.add_argument("--output-dir", default="config", help="Output directory for config")
    parser.add_argument("--max-requests-per-hour", type=int, default=5, help="Max requests per hour")
    parser.add_argument("--min-consciousness-threshold", type=float, default=0.4, help="Min consciousness threshold")
    parser.add_argument("--analysis-cooldown", type=int, default=60, help="Analysis cooldown in minutes")
    parser.add_argument("--skip-resource-check", action="store_true", help="Skip resource check")
    
    args = parser.parse_args()
    
    # Create cloud-optimized configuration
    config = DBRXCloudConfig(
        max_requests_per_hour=args.max_requests_per_hour,
        min_consciousness_threshold=args.min_consciousness_threshold,
        analysis_cooldown_minutes=args.analysis_cooldown,
        cache_enabled=True,
        memory_efficient_mode=True
    )
    
    # Setup integration
    result = setup_dbrx_cloud_integration(
        hf_token=args.hf_token,
        config=config,
        test_mode=args.test_mode,
        resource_check=not args.skip_resource_check
    )
    
    if result["success"]:
        logger.info("✅ DBRX Cloud integration setup completed successfully!")
        
        # Save configuration
        config_file = save_cloud_integration_config(
            result["integration"],
            result["config"],
            result.get("resource_assessment"),
            args.output_dir
        )
        
        # Create usage example
        example_file = create_cloud_usage_example(result["integration"])
        
        # Create cost monitoring script
        monitoring_file = create_cost_monitoring_script()
        
        # Print test results if available
        if "test_results" in result:
            test_results = result["test_results"]
            logger.info(f"📊 Test Results:")
            logger.info(f"   - Test Duration: {test_results['test_duration']:.2f} seconds")
            logger.info(f"   - Total Analyses: {test_results['performance_metrics'].get('total_analyses', 0)}")
            logger.info(f"   - Cached Analyses: {test_results['performance_metrics'].get('cached_analyses', 0)}")
            logger.info(f"   - Cloud Analyses: {test_results['performance_metrics'].get('cloud_analyses', 0)}")
            logger.info(f"   - Avg Generation Time: {test_results['performance_metrics'].get('average_generation_time', 0):.3f}s")
        
        # Print resource assessment if available
        if "resource_assessment" in result and result["resource_assessment"]:
            ra = result["resource_assessment"]
            logger.info(f"💻 Resource Assessment:")
            logger.info(f"   - CPU: {ra['cpu']['count']} cores ({ra['cpu']['usage_percent']:.1f}% usage)")
            logger.info(f"   - Memory: {ra['memory']['total_gb']:.1f}GB total")
            logger.info(f"   - GPU: {ra['gpu']['count']} GPUs found")
            logger.info(f"   - Disk: {ra['disk']['free_gb']:.1f}GB free")
            logger.info(f"   - Sufficient: {'✅' if ra['overall_sufficient'] else '⚠️'}")
        
        logger.info(f"📁 Configuration saved to: {config_file}")
        logger.info(f"📝 Usage example created: {example_file}")
        logger.info(f"💰 Cost monitoring created: {monitoring_file}")
        
        return True
    else:
        logger.error(f"❌ DBRX Cloud integration setup failed: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

