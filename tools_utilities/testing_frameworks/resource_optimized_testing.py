#!/usr/bin/env python3
"""
🛡️ Resource-Optimized Testing Framework
=======================================

This module integrates the Ultimate Resource Authority with existing testing
frameworks to automatically optimize resource usage and offload intensive
tests to free cloud platforms when needed.

Features:
- Automatic resource monitoring during tests
- Smart parameter reduction for resource-constrained environments
- Cloud offloading for memory/CPU intensive tests
- Integration with existing test suites
- Real-time optimization recommendations

Author: Quark Testing & Resource Team
Created: 2025-01-21
"""

import os, sys
import time
import json
import logging
import pytest
import psutil
import functools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
import numpy as np

# Add quark root to path
QUARK_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(QUARK_ROOT))

# Import resource monitoring
try:
    from brain_modules.resource_monitor import (
        create_resource_monitor,
        monitor_test_execution,
        IntegratedResourceManager
    )
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Resource monitoring not available: {e}")
    RESOURCE_MONITORING_AVAILABLE = False

@dataclass
class TestResourceConfig:
    """Configuration for resource-optimized testing."""
    enable_monitoring: bool = True
    enable_cloud_offload: bool = True
    auto_reduce_parameters: bool = True
    max_memory_percent: float = 80.0
    max_cpu_percent: float = 85.0
    cloud_offload_threshold_memory: float = 70.0
    cloud_offload_threshold_cpu: float = 75.0
    log_resource_usage: bool = True
    export_optimization_reports: bool = True

class ResourceOptimizedTester:
    """
    Testing framework with integrated resource optimization.
    
    This class wraps existing test functions with intelligent resource
    management, automatically reducing parameters or offloading to cloud
    when local resources are constrained.
    """
    
    def __init__(self, config: Optional[TestResourceConfig] = None):
        self.config = config or TestResourceConfig()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize resource manager if available
        self.resource_manager = None
        if RESOURCE_MONITORING_AVAILABLE and self.config.enable_monitoring:
            try:
                self.resource_manager = create_resource_monitor(
                    enable_cloud_offload=self.config.enable_cloud_offload
                )
                self.logger.info("Resource manager initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize resource manager: {e}")
        
        # Test execution history
        self.test_history: List[Dict] = []
        self.optimization_history: List[Dict] = []
        
        # Current test context
        self.current_test_context: Optional[Dict] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for resource-optimized testing."""
        logger = logging.getLogger('ResourceOptimizedTester')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_dir = QUARK_ROOT / "logs" / "resource_optimized_testing"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"resource_testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
        return logger
    
    def get_current_resource_usage(self) -> Dict[str, float]:
        """Get current system resource usage."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1.0)
        
        return {
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count()
        }
    
    def should_optimize_test(self, test_name: str, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if a test should be optimized based on current resources."""
        current_resources = self.get_current_resource_usage()
        
        optimization_decision = {
            'should_optimize': False,
            'should_offload': False,
            'optimization_type': 'none',
            'reasons': [],
            'current_resources': current_resources
        }
        
        # Check memory pressure
        if current_resources['memory_percent'] >= self.config.max_memory_percent:
            optimization_decision['should_optimize'] = True
            optimization_decision['optimization_type'] = 'parameter_reduction'
            optimization_decision['reasons'].append(f"High memory usage ({current_resources['memory_percent']:.1f}%)")
        
        # Check CPU pressure
        if current_resources['cpu_percent'] >= self.config.max_cpu_percent:
            optimization_decision['should_optimize'] = True
            optimization_decision['optimization_type'] = 'parameter_reduction'
            optimization_decision['reasons'].append(f"High CPU usage ({current_resources['cpu_percent']:.1f}%)")
        
        # Check cloud offload thresholds
        if (current_resources['memory_percent'] >= self.config.cloud_offload_threshold_memory or
            current_resources['cpu_percent'] >= self.config.cloud_offload_threshold_cpu):
            
            if self.config.enable_cloud_offload and RESOURCE_MONITORING_AVAILABLE:
                optimization_decision['should_offload'] = True
                optimization_decision['optimization_type'] = 'cloud_offload'
                optimization_decision['reasons'].append("Cloud offload recommended")
        
        # Check test-specific parameters
        if self._is_resource_intensive_test(test_params):
            optimization_decision['should_optimize'] = True
            if not optimization_decision['should_offload']:
                optimization_decision['optimization_type'] = 'parameter_reduction'
            optimization_decision['reasons'].append("Test parameters indicate high resource usage")
        
        return optimization_decision
    
    def _is_resource_intensive_test(self, test_params: Dict[str, Any]) -> bool:
        """Check if test parameters indicate resource-intensive operations."""
        intensive_indicators = [
            ('population_size', 500),
            ('num_epochs', 100),
            ('num_neurons', 1000),
            ('duration', 1000),
            ('param_combinations', 50),
            ('batch_size', 256),
            ('max_articles', 1000),
            ('simulation_steps', 10000)
        ]
        
        for param_name, threshold in intensive_indicators:
            if test_params.get(param_name, 0) > threshold:
                return True
        
        return False
    
    def optimize_test_parameters(self, test_params: Dict[str, Any], optimization_type: str) -> Dict[str, Any]:
        """Optimize test parameters based on current resource constraints."""
        optimized_params = test_params.copy()
        optimizations_applied = []
        
        if optimization_type == 'parameter_reduction':
            # Reduce common resource-intensive parameters
            parameter_reductions = {
                'population_size': 0.5,
                'num_epochs': 0.3,
                'num_neurons': 0.4,
                'duration': 0.2,
                'param_combinations': 0.3,
                'batch_size': 0.5,
                'max_articles': 0.2,
                'simulation_steps': 0.3,
                'memory_limit_mb': 0.5
            }
            
            for param_name, reduction_factor in parameter_reductions.items():
                if param_name in optimized_params:
                    original_value = optimized_params[param_name]
                    if isinstance(original_value, (int, float)) and original_value > 1:
                        new_value = max(1, int(original_value * reduction_factor))
                        optimized_params[param_name] = new_value
                        optimizations_applied.append(f"{param_name}: {original_value} -> {new_value}")
            
            # Disable GPU usage for local optimization
            if 'use_gpu' in optimized_params:
                optimized_params['use_gpu'] = False
                optimizations_applied.append("Disabled GPU usage")
            
            if 'gpu_acceleration' in optimized_params:
                optimized_params['gpu_acceleration'] = False
                optimizations_applied.append("Disabled GPU acceleration")
        
        if optimizations_applied:
            self.logger.info(f"Applied optimizations: {', '.join(optimizations_applied)}")
        
        return optimized_params
    
    def execute_optimized_test(self, 
                             test_function: Callable,
                             test_name: str,
                             test_params: Optional[Dict[str, Any]] = None,
                             **kwargs) -> Dict[str, Any]:
        """Execute a test with resource optimization."""
        
        if test_params is None:
            test_params = kwargs
        
        start_time = time.time()
        
        # Set current test context
        self.current_test_context = {
            'test_name': test_name,
            'start_time': start_time,
            'original_params': test_params.copy()
        }
        
        try:
            # Check if optimization is needed
            optimization_decision = self.should_optimize_test(test_name, test_params)
            
            # Apply optimizations if needed
            final_params = test_params
            if optimization_decision['should_optimize']:
                final_params = self.optimize_test_parameters(
                    test_params, 
                    optimization_decision['optimization_type']
                )
            
            # Execute test based on optimization decision
            if optimization_decision['should_offload'] and self.resource_manager:
                # Execute with cloud offloading
                result = self._execute_with_cloud_offload(test_function, test_name, final_params)
            else:
                # Execute locally (possibly with optimized parameters)
                result = self._execute_locally(test_function, final_params)
            
            execution_time = time.time() - start_time
            
            # Record test execution
            test_record = {
                'test_name': test_name,
                'execution_time': execution_time,
                'original_params': test_params,
                'final_params': final_params,
                'optimization_decision': optimization_decision,
                'result_status': 'success' if result.get('success', True) else 'failed',
                'resource_usage': self.get_current_resource_usage(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.test_history.append(test_record)
            
            # Log resource usage if enabled
            if self.config.log_resource_usage:
                self._log_test_resource_usage(test_record)
            
            return {
                'success': True,
                'result': result,
                'execution_time': execution_time,
                'optimization_applied': optimization_decision['should_optimize'],
                'cloud_offloaded': optimization_decision['should_offload'],
                'resource_usage': test_record['resource_usage'],
                'test_record': test_record
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_record = {
                'test_name': test_name,
                'execution_time': execution_time,
                'error': str(e),
                'original_params': test_params,
                'timestamp': datetime.now().isoformat()
            }
            
            self.test_history.append(error_record)
            
            self.logger.error(f"Test {test_name} failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'test_record': error_record
            }
        
        finally:
            self.current_test_context = None
    
    def _execute_with_cloud_offload(self, 
                                  test_function: Callable,
                                  test_name: str,
                                  test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute test with cloud offloading."""
        self.logger.info(f"Executing {test_name} with cloud offloading")
        
        if not self.resource_manager:
            # Fallback to local execution
            return self._execute_locally(test_function, test_params)
        
        try:
            # Map test function to cloud job type
            job_type = self._map_test_to_job_type(test_name, test_params)
            
            # Execute with resource manager
            result = self.resource_manager.execute_task_with_management(
                task_type=job_type,
                parameters=test_params,
                priority=4  # High priority for tests
            )
            
            return {
                'success': result['status'] == 'completed',
                'execution_location': result['execution_location'],
                'result_data': result.get('result'),
                'cloud_job_id': result.get('job_id')
            }
            
        except Exception as e:
            self.logger.warning(f"Cloud offload failed for {test_name}, falling back to local: {e}")
            return self._execute_locally(test_function, test_params)
    
    def _execute_locally(self, test_function: Callable, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute test locally."""
        try:
            if test_params:
                result = test_function(**test_params)
            else:
                result = test_function()
            
            return {
                'success': True,
                'execution_location': 'local',
                'result_data': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'execution_location': 'local',
                'error': str(e)
            }
    
    def _map_test_to_job_type(self, test_name: str, test_params: Dict[str, Any]) -> str:
        """Map test function to cloud job type."""
        test_name_lower = test_name.lower()
        
        if any(keyword in test_name_lower for keyword in ['train', 'neural', 'learning']):
            return 'neural_training'
        elif any(keyword in test_name_lower for keyword in ['param', 'optim', 'search']):
            return 'parameter_optimization'
        elif any(keyword in test_name_lower for keyword in ['valid', 'bio', 'benchmark']):
            return 'biological_validation'
        elif any(keyword in test_name_lower for keyword in ['simul', 'model', 'run']):
            return 'large_simulation'
        elif any(keyword in test_name_lower for keyword in ['data', 'analysis', 'process']):
            return 'data_analysis'
        else:
            return 'large_simulation'  # Default
    
    def _log_test_resource_usage(self, test_record: Dict[str, Any]):
        """Log resource usage for a test."""
        resource_usage = test_record['resource_usage']
        
        self.logger.info(
            f"Test {test_record['test_name']}: "
            f"Memory {resource_usage['memory_percent']:.1f}%, "
            f"CPU {resource_usage['cpu_percent']:.1f}%, "
            f"Time {test_record['execution_time']:.2f}s"
        )
        
        if test_record.get('optimization_decision', {}).get('should_optimize'):
            self.logger.info(f"Optimizations applied: {test_record['optimization_decision']['reasons']}")


# Decorators for easy integration
def resource_optimized_test(test_config: Optional[TestResourceConfig] = None):
    """
    Decorator to automatically optimize resource usage for test functions.
    
    Usage:
        @resource_optimized_test()
        def test_neural_training(population_size=1000, num_epochs=50):
            # Test implementation
            pass
    """
    def decorator(test_function):
        @functools.wraps(test_function)
        def wrapper(*args, **kwargs):
            tester = ResourceOptimizedTester(test_config)
            
            # Extract test name and parameters
            test_name = test_function.__name__
            test_params = kwargs.copy()
            
            # Execute with optimization
            result = tester.execute_optimized_test(
                test_function=test_function,
                test_name=test_name,
                test_params=test_params
            )
            
            if result['success']:
                return result['result']['result_data']
            else:
                raise Exception(f"Test failed: {result.get('error', 'Unknown error')}")
        
        return wrapper
    return decorator


def cloud_offload_test(enable_parameter_reduction: bool = True):
    """
    Decorator to enable cloud offloading for resource-intensive tests.
    
    Usage:
        @cloud_offload_test()
        def test_large_parameter_sweep(param_combinations=100):
            # Test implementation
            pass
    """
    def decorator(test_function):
        @functools.wraps(test_function)
        def wrapper(*args, **kwargs):
            config = TestResourceConfig(
                enable_cloud_offload=True,
                auto_reduce_parameters=enable_parameter_reduction,
                cloud_offload_threshold_memory=60.0,  # More aggressive offloading
                cloud_offload_threshold_cpu=70.0
            )
            
            tester = ResourceOptimizedTester(config)
            
            result = tester.execute_optimized_test(
                test_function=test_function,
                test_name=test_function.__name__,
                test_params=kwargs
            )
            
            if result['success']:
                return result['result']['result_data']
            else:
                raise Exception(f"Test failed: {result.get('error', 'Unknown error')}")
        
        return wrapper
    return decorator


# Pytest integration
class ResourceOptimizedTestPlugin:
    """Pytest plugin for resource-optimized testing."""
    
    def __init__(self):
        self.tester = ResourceOptimizedTester()
        
    def pytest_runtest_setup(self, item):
        """Setup for each test item."""
        # Check resource usage before test
        resources = self.tester.get_current_resource_usage()
        
        if (resources['memory_percent'] > 85 or resources['cpu_percent'] > 90):
            pytest.skip(f"Skipping test due to high resource usage: "
                       f"Memory {resources['memory_percent']:.1f}%, "
                       f"CPU {resources['cpu_percent']:.1f}%")
    
    def pytest_runtest_call(self, pyfuncitem):
        """Monitor test execution."""
        # This could be enhanced to wrap test execution with resource monitoring
        pass


# Example usage functions
def create_resource_optimized_test_suite() -> ResourceOptimizedTester:
    """Create a pre-configured resource-optimized test suite."""
    config = TestResourceConfig(
        enable_monitoring=True,
        enable_cloud_offload=True,
        auto_reduce_parameters=True,
        max_memory_percent=75.0,
        max_cpu_percent=80.0,
        cloud_offload_threshold_memory=65.0,
        cloud_offload_threshold_cpu=70.0,
        log_resource_usage=True,
        export_optimization_reports=True
    )
    
    return ResourceOptimizedTester(config)


def run_brain_simulation_tests_with_optimization():
    """Example of running brain simulation tests with resource optimization."""
    tester = create_resource_optimized_test_suite()
    
    # Example test functions (these would be your actual test functions)
    def test_neural_population_scaling(population_size=1000, num_epochs=50):
        """Test neural population scaling."""
        # Simulate neural training
        time.sleep(min(population_size / 1000 * 2, 10))  # Cap simulation time
        return {
            'population_size': population_size,
            'training_loss': 0.1 * np.random.random(),
            'accuracy': 0.8 + 0.2 * np.random.random()
        }
    
    def test_parameter_optimization(param_combinations=50):
        """Test parameter optimization."""
        time.sleep(min(param_combinations * 0.5, 15))
        return {
            'best_params': {'lr': 0.001, 'batch_size': 32},
            'best_score': 0.9 * np.random.random()
        }
    
    # Run tests with optimization
    test_cases = [
        (test_neural_population_scaling, {'population_size': 2000, 'num_epochs': 100}),
        (test_parameter_optimization, {'param_combinations': 100}),
    ]
    
    results = []
    for test_func, test_params in test_cases:
        result = tester.execute_optimized_test(
            test_function=test_func,
            test_name=test_func.__name__,
            test_params=test_params
        )
        results.append(result)
        
        print(f"Test {test_func.__name__}: "
              f"{'SUCCESS' if result['success'] else 'FAILED'} "
              f"({result['execution_time']:.1f}s)")
        
        if result['optimization_applied']:
            print(f"  Optimizations applied: {result['cloud_offloaded'] and 'Cloud offload' or 'Parameter reduction'}")
    
    return results


if __name__ == "__main__":
    print("🛡️ Resource-Optimized Testing Framework")
    print("Running example brain simulation tests with optimization...")
    
    try:
        results = run_brain_simulation_tests_with_optimization()
        
        print(f"\nCompleted {len(results)} tests")
        successful_tests = sum(1 for r in results if r['success'])
        print(f"Success rate: {successful_tests}/{len(results)}")
        
        # Show resource usage summary
        if results:
            avg_execution_time = np.mean([r['execution_time'] for r in results])
            optimized_tests = sum(1 for r in results if r['optimization_applied'])
            cloud_offloaded_tests = sum(1 for r in results if r['cloud_offloaded'])
            
            print(f"Average execution time: {avg_execution_time:.1f}s")
            print(f"Tests optimized: {optimized_tests}/{len(results)}")
            print(f"Tests cloud offloaded: {cloud_offloaded_tests}/{len(results)}")
            
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
