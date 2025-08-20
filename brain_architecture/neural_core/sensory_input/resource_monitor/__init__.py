#!/usr/bin/env python3
"""
🛡️ Resource Monitor Module
==========================

Ultimate Resource Authority for Mac M2 Max with intelligent cloud offloading.
This module provides supreme authority over system resources and automatically
offloads intensive operations to free cloud platforms when needed.

Features:
- Real-time resource monitoring for Mac M2 Max
- Automatic cloud offloading to Google Colab, Kaggle, etc.
- Predictive resource management
- Emergency safety controls
- Adaptive threshold optimization
- Integration with existing brain simulation systems

Author: Quark Resource Management Team
Created: 2025-01-21
"""

from ...........................................................ultimate_resource_authority import (
    UltimateResourceAuthority,
    ResourceLimits,
    ResourceMetrics,
    OptimizationAction,
    create_ultimate_authority
)

from ...........................................................cloud_offload_authority import (
    CloudOffloadAuthority,
    CloudProvider,
    CloudResource,
    OffloadJob,
    create_cloud_offload_authority
)

from ...........................................................integrated_resource_manager import (
    IntegratedResourceManager,
    IntegratedResourceConfig,
    create_integrated_resource_manager
)

# Simple interface for easy integration
def create_resource_monitor(enable_cloud_offload: bool = True, 
                          log_level: str = "INFO") -> IntegratedResourceManager:
    """
    Create a comprehensive resource monitor for Mac M2 Max.
    
    Args:
        enable_cloud_offload: Enable automatic cloud offloading
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        IntegratedResourceManager with ultimate authority over resources
    """
    if enable_cloud_offload:
        return create_integrated_resource_manager()
    else:
        # Just local resource management
        from ...........................................................ultimate_resource_authority import create_ultimate_authority
        return create_ultimate_authority()

# Convenience function for testing frameworks
def monitor_test_execution(test_function, 
                          enable_cloud_offload: bool = True,
                          auto_optimize: bool = True):
    """
    Monitor and optimize resource usage during test execution.
    
    Args:
        test_function: Function to execute with monitoring
        enable_cloud_offload: Enable cloud offloading for intensive tests
        auto_optimize: Automatically optimize parameters for resource constraints
    
    Returns:
        Test results with resource optimization information
    """
    manager = create_resource_monitor(enable_cloud_offload)
    
    try:
        with manager.integrated_management_context():
            # Get initial metrics
            initial_metrics = manager.resource_authority.get_current_metrics()
            
            # Execute test with monitoring
            start_time = time.time()
            result = test_function()
            execution_time = time.time() - start_time
            
            # Get final metrics
            final_metrics = manager.resource_authority.get_current_metrics()
            
            # Get optimization recommendations
            recommendations = manager.get_comprehensive_status()
            
            return {
                'test_result': result,
                'execution_time': execution_time,
                'resource_usage': {
                    'initial_memory_percent': initial_metrics.memory_percent,
                    'final_memory_percent': final_metrics.memory_percent,
                    'peak_memory_gb': max(initial_metrics.memory_used_gb, final_metrics.memory_used_gb),
                    'initial_cpu_percent': initial_metrics.cpu_percent,
                    'final_cpu_percent': final_metrics.cpu_percent
                },
                'optimizations_applied': len(manager.resource_authority.actions_history),
                'cloud_jobs_executed': len([j for j in manager.cloud_authority.completed_jobs if j.status == 'completed']),
                'recommendations': recommendations['recent_decisions'][-1] if recommendations['recent_decisions'] else None
            }
            
    except Exception as e:
        return {
            'test_result': None,
            'error': str(e),
            'execution_time': 0,
            'resource_usage': {},
            'optimizations_applied': 0,
            'cloud_jobs_executed': 0
        }

__all__ = [
    'UltimateResourceAuthority',
    'CloudOffloadAuthority', 
    'IntegratedResourceManager',
    'ResourceLimits',
    'ResourceMetrics',
    'OptimizationAction',
    'CloudProvider',
    'CloudResource',
    'OffloadJob',
    'IntegratedResourceConfig',
    'create_ultimate_authority',
    'create_cloud_offload_authority',
    'create_integrated_resource_manager',
    'create_resource_monitor',
    'monitor_test_execution'
]
