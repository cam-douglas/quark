# 🛡️ Ultimate Resource Authority for Mac M2 Max

The Ultimate Resource Authority is a comprehensive resource management system designed specifically for Mac Silicon M2 Max processors. It provides supreme authority over system resources with intelligent cloud offloading capabilities to maintain optimal performance during intensive brain simulation and machine learning workloads.

## ✨ Features

### 🔍 Real-Time Resource Monitoring
- **Memory Monitoring**: Tracks RAM usage with 64GB Mac M2 Max optimization
- **CPU Monitoring**: Monitors 12-core CPU (8 performance + 4 efficiency cores)
- **Temperature Monitoring**: Mac-specific thermal monitoring and throttling protection
- **GPU Monitoring**: Unified memory architecture awareness for M2 Max GPU
- **Disk I/O Monitoring**: SSD performance monitoring and optimization

### ☁️ Intelligent Cloud Offloading
- **Google Colab Integration**: Automatic offloading to free GPU instances
- **Kaggle Integration**: Leverage Kaggle's free compute resources
- **GitHub Codespaces**: Utilize development environment compute
- **Multi-Provider Support**: Intelligent routing across multiple free cloud platforms
- **Cost Optimization**: Maximizes free tiers and resource limits

### 🧠 Predictive Resource Management
- **Trend Analysis**: Predicts future resource usage based on historical data
- **Proactive Optimization**: Prevents resource constraints before they occur
- **Adaptive Thresholds**: Learns optimal resource thresholds from usage patterns
- **Smart Routing**: Intelligently decides between local and cloud execution

### 🚨 Emergency Safety Controls
- **Three-Tier Alerts**: Normal, Warning, Critical resource states
- **Process Termination**: Automatically terminates runaway processes
- **Parameter Reduction**: Reduces computational parameters when needed
- **Emergency Shutdown**: Ultimate protection against system overload
- **State Preservation**: Saves system state during emergency conditions

### 🧪 Testing Framework Integration
- **Automatic Optimization**: Tests automatically adjust to resource constraints
- **Cloud Test Offloading**: Resource-intensive tests run on cloud platforms
- **Parameter Auto-Reduction**: Test parameters scale based on available resources
- **Performance Tracking**: Detailed resource usage analysis for tests

## 🚀 Quick Start

### Basic Usage

```python
from brain_modules.resource_monitor import create_resource_monitor

# Create resource manager with cloud offloading
manager = create_resource_monitor(enable_cloud_offload=True)

# Use context manager for automatic monitoring
with manager.integrated_management_context():
    # Your resource-intensive code here
    result = run_brain_simulation()
    
# Get comprehensive status
status = manager.get_comprehensive_status()
print(f"Memory: {status['current_resources']['memory_percent']:.1f}%")
print(f"CPU: {status['current_resources']['cpu_percent']:.1f}%")
```

### Testing Integration

```python
from brain_modules.resource_monitor import monitor_test_execution

# Automatically optimize test execution
def my_intensive_test():
    # Your test code here
    return test_results

result = monitor_test_execution(
    my_intensive_test,
    enable_cloud_offload=True,
    auto_optimize=True
)

print(f"Test completed: {result['test_result']}")
print(f"Optimizations applied: {result['optimizations_applied']}")
print(f"Cloud jobs executed: {result['cloud_jobs_executed']}")
```

### Decorator-Based Testing

```python
from tools_utilities.testing_frameworks.resource_optimized_testing import (
    resource_optimized_test,
    cloud_offload_test
)

@resource_optimized_test()
def test_neural_training(population_size=1000, num_epochs=50):
    # Test will automatically optimize if resources are constrained
    return train_neural_network(population_size, num_epochs)

@cloud_offload_test()
def test_large_parameter_sweep(param_combinations=100):
    # Test will offload to cloud if local resources insufficient
    return run_parameter_sweep(param_combinations)
```

## 📋 System Requirements

### Hardware
- **Mac Silicon M2 Max** (primary target)
- **64GB RAM** (optimized for this configuration)
- **12-core CPU** (8 performance + 4 efficiency cores)
- **SSD Storage** (recommended for optimal I/O performance)

### Software
- **macOS 12.0+** (Monterey or later)
- **Python 3.8+**
- **Required packages**: `psutil`, `numpy`, `requests`
- **Optional**: Google Colab API, Kaggle API for cloud integration

### Cloud Accounts (Optional)
- **Google Account** for Colab integration
- **Kaggle Account** for Kaggle compute
- **GitHub Account** for Codespaces integration

## ⚙️ Configuration

### Resource Limits (Default Mac M2 Max Settings)

```python
from brain_modules.resource_monitor import IntegratedResourceConfig

config = IntegratedResourceConfig(
    # Memory limits (out of 64GB total)
    max_memory_gb=48.0,          # Reserve 16GB for system
    warning_memory_gb=40.0,      # Warning at 40GB usage
    critical_memory_gb=44.0,     # Critical at 44GB usage
    
    # CPU limits (12-core M2 Max)
    max_cpu_percent=85.0,        # Maximum 85% CPU usage
    warning_cpu_percent=70.0,    # Warning at 70%
    critical_cpu_percent=80.0,   # Critical at 80%
    
    # Cloud offloading thresholds
    cloud_offload_memory_threshold=70.0,  # Offload at 70% memory
    cloud_offload_cpu_threshold=75.0,     # Offload at 75% CPU
    
    # Advanced features
    enable_predictive_offloading=True,    # Predict resource needs
    enable_adaptive_thresholds=True,      # Learn optimal thresholds
    enable_emergency_shutdown=True,       # Emergency protection
    
    # Monitoring intervals
    monitoring_interval=1.0,              # Monitor every second
    cloud_status_check_interval=30.0,     # Check cloud every 30s
    max_concurrent_cloud_jobs=5           # Max 5 concurrent cloud jobs
)

manager = IntegratedResourceManager(config=config)
```

### Cloud Provider Configuration

```python
# The system automatically detects and configures available cloud providers:
# - Google Colab (25GB RAM, T4 GPU, 12 hour sessions)
# - Kaggle (30GB RAM, P100 GPU, 12 hour sessions)  
# - GitHub Codespaces (8GB RAM, 4 CPU cores, 60 hours/month)
# - Replit (4GB RAM, 2 CPU cores, 24 hour sessions)
# - Local Fallback (reduced parameters for local execution)
```

## 🏗️ Architecture

### Core Components

```
IntegratedResourceManager
├── UltimateResourceAuthority (Local Resource Management)
│   ├── Real-time monitoring (1-second intervals)
│   ├── Resource optimization (memory, CPU, temperature)
│   ├── Emergency controls (process termination, shutdown)
│   └── Safety guardrails (thresholds, limits, alerts)
├── CloudOffloadAuthority (Cloud Resource Management)
│   ├── Provider selection (Google Colab, Kaggle, etc.)
│   ├── Job scheduling (priority queue, load balancing)
│   ├── Performance tracking (success rates, execution times)
│   └── Cost optimization (free tier maximization)
└── PredictiveManager (Intelligence Layer)
    ├── Resource prediction (trend analysis, forecasting)
    ├── Decision making (local vs cloud execution)
    ├── Adaptive learning (threshold optimization)
    └── Integration coordination (callbacks, state sync)
```

### Resource Monitoring Flow

```
Resource Metrics Collection
├── System metrics (memory, CPU, temperature, I/O)
├── Process metrics (per-process resource usage)
├── Cloud metrics (provider status, job queue)
└── Prediction metrics (trend analysis, forecasting)
          ↓
Resource Status Assessment
├── Normal (< warning thresholds)
├── Warning (between warning and critical)
├── Critical (> critical thresholds)
└── Emergency (repeated critical breaches)
          ↓
Optimization Decision Making
├── Local optimization (parameter reduction, process cleanup)
├── Cloud offloading (job submission, provider selection)
├── Emergency actions (process termination, shutdown)
└── Predictive actions (preemptive optimization)
          ↓
Action Execution & Monitoring
├── Apply optimizations
├── Monitor results
├── Learn from outcomes
└── Adjust future decisions
```

## 📊 Monitoring & Reporting

### Real-Time Status

```python
# Get comprehensive system status
status = manager.get_comprehensive_status()

print("System Status:")
print(f"  Memory: {status['current_resources']['memory_percent']:.1f}%")
print(f"  CPU: {status['current_resources']['cpu_percent']:.1f}%")
print(f"  Temperature: {status['current_resources']['temperature_celsius']:.1f}°C")

print("Cloud Status:")
print(f"  Active Jobs: {status['active_cloud_jobs']}")
print(f"  Available Providers: {len(status['cloud_providers'])}")

print("Recent Decisions:")
for decision in status['recent_decisions'][-3:]:
    print(f"  {decision['execution_location']}: {', '.join(decision['reasoning'])}")
```

### Comprehensive Reporting

```python
# Export detailed report
report_file = manager.export_comprehensive_report()
print(f"Report saved to: {report_file}")

# Report includes:
# - Resource usage history
# - Decision history with reasoning
# - Cloud job performance metrics
# - Optimization effectiveness analysis
# - Predictive accuracy statistics
# - Provider performance comparisons
```

### Performance Analytics

```python
# Get optimization recommendations
recommendations = manager.get_optimization_recommendations()

print("Current Recommendations:")
for rec in recommendations['recommendations']:
    print(f"  {rec['type']}: {rec['description']} (severity: {rec['severity']})")
```

## 🧪 Testing Integration Examples

### Pytest Integration

```python
# conftest.py
import pytest
from brain_modules.resource_monitor import create_resource_monitor

@pytest.fixture(scope="session")
def resource_manager():
    manager = create_resource_monitor()
    manager.start_integrated_management()
    yield manager
    manager.stop_integrated_management()

# test_brain_simulation.py
def test_large_neural_population(resource_manager):
    with resource_manager.integrated_management_context():
        result = train_neural_population(size=5000, epochs=100)
        assert result['accuracy'] > 0.85
```

### Unittest Integration

```python
import unittest
from tools_utilities.testing_frameworks.resource_optimized_testing import ResourceOptimizedTester

class TestBrainSimulation(unittest.TestCase):
    def setUp(self):
        self.tester = ResourceOptimizedTester()
    
    def test_memory_intensive_simulation(self):
        result = self.tester.execute_optimized_test(
            test_function=run_memory_simulation,
            test_name="memory_simulation",
            test_params={'memory_gb': 8, 'duration': 300}
        )
        self.assertTrue(result['success'])
```

### Custom Test Framework

```python
from brain_modules.resource_monitor import monitor_test_execution

class BrainTestSuite:
    def __init__(self):
        self.results = []
    
    def run_test(self, test_func, **params):
        result = monitor_test_execution(
            lambda: test_func(**params),
            enable_cloud_offload=True,
            auto_optimize=True
        )
        self.results.append(result)
        return result
    
    def run_suite(self):
        # Define test cases
        tests = [
            (test_neural_training, {'population_size': 1000}),
            (test_parameter_optimization, {'combinations': 100}),
            (test_biological_validation, {'validation_type': 'comprehensive'})
        ]
        
        for test_func, params in tests:
            self.run_test(test_func, **params)
```

## 🔧 Advanced Configuration

### Custom Resource Limits

```python
# Define custom limits for specific use cases
custom_config = IntegratedResourceConfig(
    # More conservative limits for long-running simulations
    max_memory_gb=32.0,
    warning_memory_gb=24.0,
    critical_memory_gb=28.0,
    
    # More aggressive cloud offloading
    cloud_offload_memory_threshold=50.0,
    cloud_offload_cpu_threshold=60.0,
    
    # Faster monitoring for real-time applications
    monitoring_interval=0.5,
    
    # More concurrent cloud jobs for parallel workloads
    max_concurrent_cloud_jobs=10
)
```

### Custom Cloud Providers

```python
# Add custom cloud provider
from brain_modules.resource_monitor.cloud_offload_authority import CloudProvider, CloudResource

custom_provider = CloudResource(
    provider=CloudProvider.CUSTOM,
    session_id="my_custom_cloud",
    status="available",
    capabilities=["gpu", "high_memory"],
    cost_tier="free",
    resource_limits={
        "max_memory_gb": 16.0,
        "max_gpu_memory_gb": 8.0,
        "max_runtime_hours": 6.0
    },
    last_used=datetime.now(),
    performance_score=0.8
)

manager.cloud_authority.available_providers[CloudProvider.CUSTOM] = custom_provider
```

### Custom Optimization Callbacks

```python
# Register custom optimization callback
def my_optimization_callback(metrics):
    """Custom optimization logic."""
    if metrics.memory_percent > 80:
        # Custom memory optimization
        return optimize_my_application_memory()
    return {}

manager.resource_authority.register_optimization_callback(
    'my_optimization',
    my_optimization_callback
)

# Register emergency callback
def my_emergency_callback(emergency_type):
    """Handle emergency situations."""
    if emergency_type == "EMERGENCY_SHUTDOWN":
        save_critical_data()
        cleanup_resources()

manager.resource_authority.register_emergency_callback(
    'my_emergency_handler',
    my_emergency_callback
)
```

## 🔍 Troubleshooting

### Common Issues

#### Memory Monitoring Not Working
```bash
# Check if psutil is installed and working
python -c "import psutil; print(psutil.virtual_memory())"

# On macOS, may need to grant accessibility permissions
# System Preferences → Security & Privacy → Privacy → Accessibility
```

#### Cloud Offloading Fails
```python
# Check cloud provider status
status = manager.cloud_authority.get_system_status()
for provider, info in status['providers'].items():
    print(f"{provider}: {info['status']} (score: {info['performance_score']})")

# Enable debug logging
import logging
logging.getLogger('CloudOffloadAuthority').setLevel(logging.DEBUG)
```

#### Temperature Monitoring Issues
```bash
# Temperature monitoring requires sudo access for powermetrics
# Grant permission when prompted, or run with reduced monitoring
sudo python demo_resource_authority.py
```

#### High Resource Usage
```python
# Check what's using resources
current_metrics = manager.resource_authority.get_current_metrics()
print(f"Memory: {current_metrics.memory_used_gb:.1f}GB")
print(f"CPU: {current_metrics.cpu_percent:.1f}%")

# Get optimization recommendations
recommendations = manager.get_optimization_recommendations()
for rec in recommendations['recommendations']:
    print(f"Recommendation: {rec['description']}")
```

### Debug Mode

```python
# Enable comprehensive debugging
manager = create_integrated_resource_manager()
manager.logger.setLevel(logging.DEBUG)
manager.resource_authority.logger.setLevel(logging.DEBUG)
manager.cloud_authority.logger.setLevel(logging.DEBUG)

# Export debug report
debug_report = manager.export_comprehensive_report()
print(f"Debug report: {debug_report}")
```

### Performance Tuning

```python
# Optimize for specific workloads
if workload_type == "neural_training":
    config.cloud_offload_memory_threshold = 60.0  # Earlier offload
    config.max_concurrent_cloud_jobs = 3          # Fewer concurrent jobs
elif workload_type == "parameter_sweep":
    config.cloud_offload_memory_threshold = 80.0  # Later offload
    config.max_concurrent_cloud_jobs = 10         # More concurrent jobs
```

## 📚 API Reference

### Core Classes

#### `IntegratedResourceManager`
Main class combining local and cloud resource management.

**Methods:**
- `start_integrated_management()`: Start monitoring and management
- `stop_integrated_management()`: Stop all monitoring
- `execute_task_with_management(task_type, parameters, priority)`: Execute task with optimization
- `get_comprehensive_status()`: Get current system status
- `export_comprehensive_report()`: Generate detailed report

#### `UltimateResourceAuthority`
Local resource monitoring and optimization.

**Methods:**
- `get_current_metrics()`: Get current resource metrics
- `assess_resource_status(metrics)`: Assess resource status
- `optimize_system_resources(metrics, severity)`: Apply optimizations
- `register_optimization_callback(name, callback)`: Register optimization callback

#### `CloudOffloadAuthority`
Cloud resource management and job offloading.

**Methods:**
- `submit_job(job_type, parameters, priority)`: Submit job for cloud execution
- `get_job_status(job_id)`: Get status of submitted job
- `wait_for_job(job_id, timeout)`: Wait for job completion
- `get_system_status()`: Get cloud system status

### Data Classes

#### `ResourceMetrics`
Current system resource metrics.

**Fields:**
- `memory_used_gb`: Memory usage in GB
- `memory_percent`: Memory usage percentage
- `cpu_percent`: CPU usage percentage
- `temperature_celsius`: System temperature
- `disk_io_read_mbps`: Disk read speed
- `disk_io_write_mbps`: Disk write speed

#### `OptimizationAction`
Record of optimization action taken.

**Fields:**
- `action_type`: Type of optimization
- `severity`: Severity level that triggered action
- `description`: Human-readable description
- `parameters_changed`: Parameters that were modified
- `resource_impact`: Expected resource impact

### Utility Functions

```python
# Create resource monitor with defaults
create_resource_monitor(enable_cloud_offload=True, log_level="INFO")

# Monitor test execution
monitor_test_execution(test_function, enable_cloud_offload=True, auto_optimize=True)

# Create specific components
create_ultimate_authority()
create_cloud_offload_authority()
create_integrated_resource_manager()
```

## 🤝 Contributing

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/quark.git
cd quark

# Install development dependencies
pip install -e ."[dev]"

# Run tests
python -m pytest tests/

# Run demo
python demo_resource_authority.py
```

### Adding Cloud Providers

1. Add provider to `CloudProvider` enum
2. Create `CloudResource` configuration
3. Implement execution method in `CloudOffloadAuthority`
4. Add provider initialization in `_initialize_cloud_providers()`
5. Add tests for new provider

### Adding Optimization Strategies

1. Create optimization method in `UltimateResourceAuthority`
2. Add trigger conditions in `assess_resource_status()`
3. Include in optimization selection logic
4. Add tests and documentation

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Built for Mac Silicon M2 Max optimization
- Integrates with existing Quark brain simulation framework
- Designed for scientific computing and machine learning workloads
- Optimized for free cloud computing platforms

---

**Need help?** Check out the [demonstration script](../../demo_resource_authority.py) for working examples, or run the interactive demo:

```bash
python demo_resource_authority.py --interactive
```
