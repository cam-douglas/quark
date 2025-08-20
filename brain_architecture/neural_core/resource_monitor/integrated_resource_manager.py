#!/usr/bin/env python3
"""
🛡️ Integrated Resource Manager
==============================

This module combines the Ultimate Resource Authority with Cloud Offload Authority
to provide comprehensive resource management for Mac M2 Max with automatic cloud
offloading capabilities. It has supreme authority over all system resources.

AUTHORITY LEVEL: SUPREME - Can override any operation
INTEGRATION: Ultimate Resource Authority + Cloud Offload Authority
TARGET: Mac Silicon M2 Max with intelligent cloud backup

Author: Quark Integrated Resource Team
Created: 2025-01-21
"""

import os, sys
import time
import json
import logging
import threading
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Add quark root to path
QUARK_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(QUARK_ROOT))

# Import our components
from ................................................ultimate_resource_authority import UltimateResourceAuthority, ResourceLimits, ResourceMetrics, OptimizationAction
from ................................................cloud_offload_authority import CloudOffloadAuthority, CloudProvider, OffloadJob

@dataclass
class IntegratedResourceConfig:
    """Configuration for integrated resource management."""
    
    # Local resource limits (Mac M2 Max)
    max_memory_gb: float = 48.0
    warning_memory_gb: float = 40.0
    critical_memory_gb: float = 44.0
    
    max_cpu_percent: float = 85.0
    warning_cpu_percent: float = 70.0
    critical_cpu_percent: float = 80.0
    
    # Cloud offloading thresholds
    cloud_offload_memory_threshold: float = 70.0
    cloud_offload_cpu_threshold: float = 75.0
    
    # Intelligent routing parameters
    enable_predictive_offloading: bool = True
    enable_adaptive_thresholds: bool = True
    enable_emergency_shutdown: bool = True
    
    # Monitoring intervals
    monitoring_interval: float = 1.0
    cloud_status_check_interval: float = 30.0
    
    # Performance optimization
    max_concurrent_cloud_jobs: int = 5
    local_fallback_enabled: bool = True
    
class IntegratedResourceManager:
    """
    Supreme authority for resource management combining local optimization
    with intelligent cloud offloading for sustained performance.
    
    This system:
    1. Monitors all local resources continuously
    2. Predicts resource constraints before they occur
    3. Automatically offloads appropriate tasks to free cloud platforms
    4. Maintains emergency controls and safety limits
    5. Optimizes performance across local and cloud resources
    """
    
    def __init__(self, 
                 config: Optional[IntegratedResourceConfig] = None,
                 log_level: str = "INFO"):
        
        self.config = config or IntegratedResourceConfig()
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Initialize resource limits for Ultimate Authority
        resource_limits = ResourceLimits(
            max_memory_gb=self.config.max_memory_gb,
            warning_memory_gb=self.config.warning_memory_gb,
            critical_memory_gb=self.config.critical_memory_gb,
            max_cpu_percent=self.config.max_cpu_percent,
            warning_cpu_percent=self.config.warning_cpu_percent,
            critical_cpu_percent=self.config.critical_cpu_percent
        )
        
        # Initialize components
        self.resource_authority = UltimateResourceAuthority(
            limits=resource_limits,
            monitoring_interval=self.config.monitoring_interval,
            enable_emergency_shutdown=self.config.enable_emergency_shutdown,
            log_level=log_level
        )
        
        self.cloud_authority = CloudOffloadAuthority(
            enable_offloading=True,
            max_concurrent_jobs=self.config.max_concurrent_cloud_jobs,
            resource_threshold_memory_percent=self.config.cloud_offload_memory_threshold,
            resource_threshold_cpu_percent=self.config.cloud_offload_cpu_threshold,
            log_level=log_level
        )
        
        # Integration state
        self.is_managing = False
        self.management_thread = None
        self.resource_prediction_enabled = self.config.enable_predictive_offloading
        
        # Performance tracking
        self.decision_history: List[Dict] = []
        self.resource_predictions: List[Dict] = []
        self.optimization_effectiveness: Dict[str, float] = {}
        
        # Adaptive thresholds
        self.adaptive_thresholds_enabled = self.config.enable_adaptive_thresholds
        self.learned_thresholds: Dict[str, float] = {}
        
        # Integration callbacks
        self.resource_callbacks: Dict[str, Callable] = {}
        self.offload_callbacks: Dict[str, Callable] = {}
        
        # Register integration callbacks
        self._setup_integration_callbacks()
        
        self.logger.info("🛡️ Integrated Resource Manager initialized")
        self.logger.info(f"Local limits: {self.config.max_memory_gb}GB memory, {self.config.max_cpu_percent}% CPU")
        self.logger.info(f"Cloud thresholds: {self.config.cloud_offload_memory_threshold}% memory, {self.config.cloud_offload_cpu_threshold}% CPU")
        
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup integrated logging system."""
        logger = logging.getLogger('IntegratedResourceManager')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            log_dir = QUARK_ROOT / "logs" / "integrated_resource_management"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"integrated_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        
    def _setup_integration_callbacks(self):
        """Setup callbacks between resource authority and cloud authority."""
        
        # Resource authority callback for cloud offloading
        def resource_to_cloud_callback(metrics: ResourceMetrics) -> Dict[str, Any]:
            """Callback from resource authority to trigger cloud offloading."""
            try:
                # Determine if we should offload based on current metrics
                should_offload = self.cloud_authority.should_offload(
                    current_memory_percent=metrics.memory_percent,
                    current_cpu_percent=metrics.cpu_percent,
                    job_requirements={'memory_gb': 2.0, 'duration_seconds': 300}
                )
                
                if should_offload:
                    # Submit a resource optimization job to cloud
                    job_id = self.cloud_authority.submit_job(
                        job_type='resource_optimization',
                        parameters={
                            'current_memory_percent': metrics.memory_percent,
                            'current_cpu_percent': metrics.cpu_percent,
                            'optimization_type': 'emergency_offload'
                        },
                        priority=5  # Highest priority
                    )
                    
                    self.logger.info(f"Triggered emergency cloud offload: {job_id}")
                    return {'offload_job_id': job_id, 'offloaded': True}
                    
                return {'offloaded': False}
                
            except Exception as e:
                self.logger.error(f"Error in resource-to-cloud callback: {e}")
                return {'error': str(e)}
        
        # Cloud authority callback for resource optimization
        def cloud_to_resource_callback(job_result: Dict[str, Any]) -> Dict[str, Any]:
            """Callback from cloud authority to apply local optimizations."""
            try:
                if job_result.get('status') == 'completed':
                    # Apply optimizations based on cloud results
                    optimizations = job_result.get('optimizations', {})
                    
                    applied_optimizations = []
                    for opt_type, opt_params in optimizations.items():
                        if opt_type == 'memory_reduction':
                            # Apply memory optimizations
                            action = self.resource_authority._optimize_memory_usage()
                            if action:
                                applied_optimizations.append(action)
                        elif opt_type == 'cpu_reduction':
                            # Apply CPU optimizations
                            action = self.resource_authority._reduce_cpu_intensive_operations()
                            if action:
                                applied_optimizations.append(action)
                    
                    return {'applied_optimizations': len(applied_optimizations)}
                    
                return {'applied_optimizations': 0}
                
            except Exception as e:
                self.logger.error(f"Error in cloud-to-resource callback: {e}")
                return {'error': str(e)}
        
        # Register callbacks
        self.resource_authority.register_optimization_callback('cloud_offload', resource_to_cloud_callback)
        self.resource_authority.register_emergency_callback('cloud_emergency', resource_to_cloud_callback)
        
    def predict_resource_usage(self, 
                              current_metrics: ResourceMetrics,
                              prediction_window_seconds: float = 300.0) -> Dict[str, float]:
        """Predict future resource usage based on current trends."""
        
        if not self.resource_prediction_enabled:
            return {}
        
        if len(self.resource_authority.metrics_history) < 10:
            # Not enough history for prediction
            return {}
        
        try:
            # Get recent metrics for trend analysis
            recent_metrics = self.resource_authority.metrics_history[-10:]
            
            # Calculate trends
            memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
            cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
            
            # Predict future values
            predicted_memory = current_metrics.memory_percent + (memory_trend * prediction_window_seconds / 60.0)
            predicted_cpu = current_metrics.cpu_percent + (cpu_trend * prediction_window_seconds / 60.0)
            
            # Clamp predictions to reasonable ranges
            predicted_memory = max(0, min(100, predicted_memory))
            predicted_cpu = max(0, min(100, predicted_cpu))
            
            prediction = {
                'timestamp': datetime.now().isoformat(),
                'prediction_window_seconds': prediction_window_seconds,
                'predicted_memory_percent': predicted_memory,
                'predicted_cpu_percent': predicted_cpu,
                'memory_trend': memory_trend,
                'cpu_trend': cpu_trend,
                'confidence': self._calculate_prediction_confidence(recent_metrics)
            }
            
            self.resource_predictions.append(prediction)
            
            # Keep predictions history manageable
            if len(self.resource_predictions) > 100:
                self.resource_predictions = self.resource_predictions[-50:]
                
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting resource usage: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (rate of change) from a list of values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        if np.std(x) == 0:
            return 0.0
        
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        return slope if not np.isnan(slope) else 0.0
    
    def _calculate_prediction_confidence(self, recent_metrics: List[ResourceMetrics]) -> float:
        """Calculate confidence in resource predictions."""
        if len(recent_metrics) < 5:
            return 0.3  # Low confidence with little data
        
        # Calculate variability in recent metrics
        memory_values = [m.memory_percent for m in recent_metrics]
        cpu_values = [m.cpu_percent for m in recent_metrics]
        
        memory_variability = np.std(memory_values) / (np.mean(memory_values) + 1e-6)
        cpu_variability = np.std(cpu_values) / (np.mean(cpu_values) + 1e-6)
        
        # Lower variability = higher confidence
        avg_variability = (memory_variability + cpu_variability) / 2.0
        confidence = max(0.1, min(1.0, 1.0 - avg_variability))
        
        return confidence
    
    def make_intelligent_decision(self, 
                                 current_metrics: ResourceMetrics,
                                 task_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Make an intelligent decision about where to execute a task."""
        
        decision_start_time = time.time()
        
        # Get predictions if enabled
        predictions = self.predict_resource_usage(current_metrics)
        
        # Assess current local resource status
        local_severity, local_issues = self.resource_authority.assess_resource_status(current_metrics)
        
        # Check if cloud offloading is recommended
        should_offload = self.cloud_authority.should_offload(
            current_memory_percent=current_metrics.memory_percent,
            current_cpu_percent=current_metrics.cpu_percent,
            job_requirements=task_requirements
        )
        
        # Make decision based on multiple factors
        decision_factors = {
            'local_severity': local_severity,
            'current_memory_percent': current_metrics.memory_percent,
            'current_cpu_percent': current_metrics.cpu_percent,
            'predicted_memory_percent': predictions.get('predicted_memory_percent', current_metrics.memory_percent),
            'predicted_cpu_percent': predictions.get('predicted_cpu_percent', current_metrics.cpu_percent),
            'cloud_offload_recommended': should_offload,
            'task_memory_requirement': task_requirements.get('memory_gb', 0),
            'task_duration': task_requirements.get('duration_seconds', 0),
            'task_gpu_required': task_requirements.get('gpu_required', False)
        }
        
        # Decision logic
        execution_location = 'local'  # Default
        reasoning = []
        
        # Critical local resource issues
        if local_severity == 'critical':
            execution_location = 'cloud'
            reasoning.append('Critical local resource issues detected')
        
        # Predictive offloading
        elif (predictions.get('predicted_memory_percent', 0) > self.config.critical_memory_gb * 100 / 64 or
              predictions.get('predicted_cpu_percent', 0) > self.config.critical_cpu_percent):
            execution_location = 'cloud'
            reasoning.append('Predicted resource constraints')
        
        # Task-specific requirements
        elif (task_requirements.get('memory_gb', 0) > 8.0 or
              task_requirements.get('duration_seconds', 0) > 600 or
              task_requirements.get('gpu_required', False)):
            execution_location = 'cloud'
            reasoning.append('Task requirements exceed local comfort zone')
        
        # Resource pressure
        elif should_offload:
            execution_location = 'cloud'
            reasoning.append('Local resource pressure detected')
        
        else:
            reasoning.append('Local resources sufficient')
        
        # Final decision
        decision = {
            'timestamp': datetime.now().isoformat(),
            'execution_location': execution_location,
            'reasoning': reasoning,
            'decision_factors': decision_factors,
            'confidence': self._calculate_decision_confidence(decision_factors),
            'decision_time_ms': (time.time() - decision_start_time) * 1000
        }
        
        # Store decision for learning
        self.decision_history.append(decision)
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]
        
        self.logger.info(f"Decision: {execution_location} execution - {', '.join(reasoning)}")
        
        return decision
    
    def _calculate_decision_confidence(self, factors: Dict[str, Any]) -> float:
        """Calculate confidence in the execution decision."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on severity
        if factors['local_severity'] == 'critical':
            confidence += 0.2
        elif factors['local_severity'] == 'warning':
            confidence += 0.1
        
        # Adjust based on prediction confidence
        prediction_confidence = factors.get('prediction_confidence', 0.5)
        confidence = confidence * (0.5 + 0.5 * prediction_confidence)
        
        # Adjust based on resource headroom
        memory_headroom = 100 - factors['current_memory_percent']
        cpu_headroom = 100 - factors['current_cpu_percent']
        
        if memory_headroom < 20 or cpu_headroom < 20:
            confidence += 0.1  # More confident about offloading when low headroom
        
        return max(0.1, min(1.0, confidence))
    
    def execute_task_with_management(self, 
                                   task_type: str,
                                   parameters: Dict[str, Any],
                                   priority: int = 3) -> Dict[str, Any]:
        """Execute a task with integrated resource management."""
        
        try:
            # Get current metrics
            current_metrics = self.resource_authority.get_current_metrics()
            
            # Estimate task requirements
            task_requirements = self.cloud_authority._estimate_requirements(task_type, parameters)
            
            # Make intelligent decision
            decision = self.make_intelligent_decision(current_metrics, task_requirements)
            
            if decision['execution_location'] == 'cloud':
                # Execute on cloud
                self.logger.info(f"Executing {task_type} on cloud")
                
                job_id = self.cloud_authority.submit_job(
                    job_type=task_type,
                    parameters=parameters,
                    priority=priority,
                    resource_requirements=task_requirements
                )
                
                # Wait for completion
                result = self.cloud_authority.wait_for_job(job_id, timeout=600.0)
                
                if result and result['status'] == 'completed':
                    return {
                        'status': 'completed',
                        'execution_location': 'cloud',
                        'job_id': job_id,
                        'result': result.get('result'),
                        'decision': decision
                    }
                else:
                    # Cloud execution failed, try local fallback
                    self.logger.warning(f"Cloud execution failed for {task_type}, trying local fallback")
                    return self._execute_local_with_optimization(task_type, parameters, task_requirements)
                    
            else:
                # Execute locally
                self.logger.info(f"Executing {task_type} locally")
                return self._execute_local_with_optimization(task_type, parameters, task_requirements)
                
        except Exception as e:
            self.logger.error(f"Error executing task {task_type}: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_location': 'failed'
            }
    
    def _execute_local_with_optimization(self, 
                                       task_type: str,
                                       parameters: Dict[str, Any],
                                       task_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task locally with resource optimization."""
        
        start_time = time.time()
        
        # Apply local optimizations before execution
        current_metrics = self.resource_authority.get_current_metrics()
        local_severity, _ = self.resource_authority.assess_resource_status(current_metrics)
        
        if local_severity in ['warning', 'critical']:
            # Apply optimizations
            optimizations = self.resource_authority.optimize_system_resources(current_metrics, local_severity)
            self.logger.info(f"Applied {len(optimizations)} local optimizations before execution")
        
        # Reduce parameters if needed for local execution
        if task_requirements.get('memory_gb', 0) > 4.0 or task_requirements.get('duration_seconds', 0) > 300:
            parameters = self.cloud_authority._reduce_parameters_for_local(parameters)
            self.logger.info("Reduced parameters for local execution")
        
        # Simulate local execution
        execution_time = min(task_requirements.get('duration_seconds', 60), 30.0)
        time.sleep(execution_time)
        
        # Generate result based on task type
        if task_type == 'neural_training':
            result = {
                'training_loss': 0.2 * np.random.random(),  # Slightly higher loss for local
                'validation_accuracy': 0.7 + 0.2 * np.random.random(),
                'epochs_completed': parameters.get('num_epochs', 5),
                'execution_time': execution_time,
                'provider': 'local_optimized'
            }
        else:
            result = {
                'status': 'completed',
                'execution_time': execution_time,
                'provider': 'local_optimized'
            }
        
        return {
            'status': 'completed',
            'execution_location': 'local',
            'result': result,
            'total_execution_time': time.time() - start_time
        }
    
    def start_integrated_management(self):
        """Start integrated resource management."""
        if self.is_managing:
            self.logger.warning("Integrated management already running")
            return
        
        # Start resource authority monitoring
        self.resource_authority.start_monitoring()
        
        # Start cloud authority job processor
        self.cloud_authority.start_job_processor()
        
        # Start our integrated management
        self.is_managing = True
        self.management_thread = threading.Thread(target=self._management_loop, daemon=True)
        self.management_thread.start()
        
        self.logger.info("🛡️ Integrated resource management started")
    
    def stop_integrated_management(self):
        """Stop integrated resource management."""
        self.is_managing = False
        
        # Stop our management thread
        if self.management_thread and self.management_thread.is_alive():
            self.management_thread.join(timeout=10.0)
        
        # Stop components
        self.resource_authority.stop_monitoring()
        self.cloud_authority.stop_job_processor()
        
        self.logger.info("🛡️ Integrated resource management stopped")
    
    def _management_loop(self):
        """Main integrated management loop."""
        self.logger.info("Starting integrated management loop")
        
        while self.is_managing:
            try:
                # Periodic cloud status check
                cloud_status = self.cloud_authority.get_system_status()
                
                # Adaptive threshold adjustment
                if self.adaptive_thresholds_enabled:
                    self._adjust_adaptive_thresholds()
                
                # Health check and optimization
                self._perform_health_check()
                
                # Sleep for the cloud status check interval
                time.sleep(self.config.cloud_status_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in management loop: {e}")
                time.sleep(10)
        
        self.logger.info("Integrated management loop ended")
    
    def _adjust_adaptive_thresholds(self):
        """Adjust thresholds based on historical performance."""
        if len(self.decision_history) < 20:
            return  # Need more history
        
        try:
            recent_decisions = self.decision_history[-20:]
            
            # Analyze decision effectiveness
            cloud_decisions = [d for d in recent_decisions if d['execution_location'] == 'cloud']
            local_decisions = [d for d in recent_decisions if d['execution_location'] == 'local']
            
            # If too many cloud decisions, increase thresholds
            if len(cloud_decisions) / len(recent_decisions) > 0.7:
                self.config.cloud_offload_memory_threshold = min(90.0, self.config.cloud_offload_memory_threshold + 2.0)
                self.config.cloud_offload_cpu_threshold = min(95.0, self.config.cloud_offload_cpu_threshold + 2.0)
                self.logger.info("Increased cloud offload thresholds to reduce cloud usage")
            
            # If too few cloud decisions and we had resource issues, decrease thresholds
            elif len(cloud_decisions) / len(recent_decisions) < 0.2:
                recent_metrics = self.resource_authority.metrics_history[-20:] if self.resource_authority.metrics_history else []
                if any(m.memory_percent > 80 or m.cpu_percent > 80 for m in recent_metrics):
                    self.config.cloud_offload_memory_threshold = max(50.0, self.config.cloud_offload_memory_threshold - 2.0)
                    self.config.cloud_offload_cpu_threshold = max(60.0, self.config.cloud_offload_cpu_threshold - 2.0)
                    self.logger.info("Decreased cloud offload thresholds to increase cloud usage")
                    
        except Exception as e:
            self.logger.error(f"Error adjusting adaptive thresholds: {e}")
    
    def _perform_health_check(self):
        """Perform periodic health check and optimization."""
        try:
            # Check resource authority health
            if self.resource_authority.emergency_stop_triggered:
                self.logger.critical("Emergency stop triggered in resource authority!")
                
            # Check cloud authority health
            cloud_status = self.cloud_authority.get_system_status()
            if not cloud_status['job_processor_running']:
                self.logger.warning("Cloud job processor not running, restarting...")
                self.cloud_authority.start_job_processor()
            
            # Check for stuck jobs
            stuck_jobs = []
            for job_id, job in self.cloud_authority.active_jobs.items():
                if job.status == 'running':
                    time_running = (datetime.now() - job.created_at).total_seconds()
                    if time_running > job.estimated_duration * 3:  # 3x estimated time
                        stuck_jobs.append(job_id)
            
            # Cancel stuck jobs
            for job_id in stuck_jobs:
                self.cloud_authority.cancel_job(job_id)
                self.logger.warning(f"Cancelled stuck job: {job_id}")
                
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
    
    @contextmanager
    def integrated_management_context(self):
        """Context manager for integrated resource management."""
        try:
            self.start_integrated_management()
            yield self
        finally:
            self.stop_integrated_management()
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the integrated system."""
        
        # Get component statuses
        resource_status = self.resource_authority.get_optimization_recommendations()
        cloud_status = self.cloud_authority.get_system_status()
        
        # Get current metrics
        current_metrics = self.resource_authority.get_current_metrics()
        
        # Get predictions
        predictions = self.predict_resource_usage(current_metrics)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': {
                'integrated_management_running': self.is_managing,
                'resource_authority_active': self.resource_authority.is_monitoring,
                'cloud_authority_active': cloud_status['job_processor_running'],
                'emergency_stop_triggered': self.resource_authority.emergency_stop_triggered
            },
            'current_resources': {
                'memory_used_gb': current_metrics.memory_used_gb,
                'memory_percent': current_metrics.memory_percent,
                'cpu_percent': current_metrics.cpu_percent,
                'temperature_celsius': current_metrics.temperature_celsius
            },
            'predictions': predictions,
            'thresholds': {
                'local_warning_memory_gb': self.config.warning_memory_gb,
                'local_critical_memory_gb': self.config.critical_memory_gb,
                'cloud_offload_memory_percent': self.config.cloud_offload_memory_threshold,
                'cloud_offload_cpu_percent': self.config.cloud_offload_cpu_threshold
            },
            'cloud_providers': cloud_status['providers'],
            'active_cloud_jobs': cloud_status['active_jobs'],
            'recent_decisions': self.decision_history[-5:] if self.decision_history else [],
            'config': {
                'predictive_offloading': self.resource_prediction_enabled,
                'adaptive_thresholds': self.adaptive_thresholds_enabled,
                'max_concurrent_cloud_jobs': self.config.max_concurrent_cloud_jobs
            }
        }
    
    def export_comprehensive_report(self, output_file: Optional[Path] = None) -> Path:
        """Export comprehensive report of integrated system."""
        if not output_file:
            output_file = QUARK_ROOT / "logs" / "integrated_resource_management" / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Gather all data
        comprehensive_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'system_info': {
                    'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                    'cpu_count': psutil.cpu_count(),
                    'system': 'Mac M2 Max'
                }
            },
            'configuration': asdict(self.config),
            'current_status': self.get_comprehensive_status(),
            'resource_history': [asdict(m) for m in self.resource_authority.metrics_history[-100:]],
            'decision_history': self.decision_history[-100:],
            'prediction_history': self.resource_predictions[-50:],
            'resource_actions': [asdict(a) for a in self.resource_authority.actions_history[-50:]],
            'cloud_job_history': self.cloud_authority.completed_jobs[-50:] if hasattr(self.cloud_authority, 'completed_jobs') else []
        }
        
        with open(output_file, 'w') as f:
            json.dump(comprehensive_data, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive report exported to {output_file}")
        return output_file


def create_integrated_resource_manager() -> IntegratedResourceManager:
    """Create and configure the Integrated Resource Manager for Mac M2 Max."""
    
    config = IntegratedResourceConfig(
        # Mac M2 Max optimized limits
        max_memory_gb=48.0,
        warning_memory_gb=40.0,
        critical_memory_gb=44.0,
        
        max_cpu_percent=85.0,
        warning_cpu_percent=70.0,
        critical_cpu_percent=80.0,
        
        # Cloud offloading thresholds
        cloud_offload_memory_threshold=70.0,
        cloud_offload_cpu_threshold=75.0,
        
        # Advanced features
        enable_predictive_offloading=True,
        enable_adaptive_thresholds=True,
        enable_emergency_shutdown=True,
        
        # Performance settings
        monitoring_interval=1.0,
        cloud_status_check_interval=30.0,
        max_concurrent_cloud_jobs=5,
        local_fallback_enabled=True
    )
    
    manager = IntegratedResourceManager(config=config, log_level="INFO")
    
    return manager


# Example usage and testing
if __name__ == "__main__":
    # Create the integrated manager
    manager = create_integrated_resource_manager()
    
    print("🛡️ Integrated Resource Manager initialized")
    print("Features: Resource monitoring + Cloud offloading + Predictive management")
    
    try:
        with manager.integrated_management_context():
            print("\nStarted integrated management for 60 seconds...")
            
            # Test different task types
            test_tasks = [
                ('neural_training', {'population_size': 100, 'num_epochs': 5}),
                ('parameter_optimization', {'param_combinations': 15}),
                ('biological_validation', {'validation_type': 'basic'}),
                ('large_simulation', {'duration': 500, 'num_neurons': 200})
            ]
            
            for i, (task_type, params) in enumerate(test_tasks):
                print(f"\nExecuting test task {i+1}: {task_type}")
                
                result = manager.execute_task_with_management(
                    task_type=task_type,
                    parameters=params,
                    priority=3
                )
                
                print(f"Result: {result['status']} on {result['execution_location']}")
                
                # Wait between tasks
                time.sleep(5)
            
            # Show comprehensive status
            status = manager.get_comprehensive_status()
            print(f"\nFinal status:")
            print(f"  Memory: {status['current_resources']['memory_percent']:.1f}%")
            print(f"  CPU: {status['current_resources']['cpu_percent']:.1f}%")
            print(f"  Active cloud jobs: {status['active_cloud_jobs']}")
            print(f"  Recent decisions: {len(status['recent_decisions'])}")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # Export comprehensive report
    report_file = manager.export_comprehensive_report()
    print(f"\nComprehensive report saved to: {report_file}")
