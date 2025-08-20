#!/usr/bin/env python3
"""
🛡️ Ultimate Resource Authority Agent
====================================

This agent has ultimate authority over system resource management for Mac Silicon M2 Max.
It monitors and dynamically adjusts all operations to stay within optimal resource parameters.

AUTHORITY LEVEL: SUPREME (can override any operation for resource optimization)
TARGET HARDWARE: Mac Silicon M2 Max (64GB RAM, 12-core CPU)
SAFETY PRIORITY: HIGHEST

Author: Quark Resource Management Team
Created: 2025-01-21
"""

import os, sys
import psutil
import time
import logging
import threading
import numpy as np
import json
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass, asdict
from contextlib import contextmanager
import gc
import resource
import subprocess
import multiprocessing as mp
from queue import Queue, Empty
import warnings

# Add quark root to path
QUARK_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(QUARK_ROOT))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ResourceLimits:
    """Mac M2 Max optimal resource limits."""
    # Memory limits (64GB total, reserve 16GB for system)
    max_memory_gb: float = 48.0  # Maximum usable memory
    warning_memory_gb: float = 40.0  # Warning threshold
    critical_memory_gb: float = 44.0  # Critical threshold
    
    # CPU limits (12-core M2 Max, 8 performance + 4 efficiency cores)
    max_cpu_percent: float = 85.0  # Maximum CPU usage
    warning_cpu_percent: float = 70.0  # Warning threshold
    critical_cpu_percent: float = 80.0  # Critical threshold
    
    # Process limits
    max_concurrent_processes: int = 8  # Optimal for M2 Max
    max_threads_per_process: int = 4  # Balanced threading
    
    # GPU limits (if applicable)
    max_gpu_memory_percent: float = 90.0
    warning_gpu_memory_percent: float = 75.0
    
    # Disk I/O limits
    max_disk_io_mbps: float = 500.0  # MB/s
    warning_disk_io_mbps: float = 400.0
    
    # Temperature limits (Silicon thermal throttling)
    max_temperature_celsius: float = 85.0
    warning_temperature_celsius: float = 75.0

@dataclass
class ResourceMetrics:
    """Current system resource metrics."""
    timestamp: datetime
    memory_used_gb: float
    memory_percent: float
    cpu_percent: float
    cpu_cores_usage: List[float]
    disk_io_read_mbps: float
    disk_io_write_mbps: float
    network_io_mbps: float
    temperature_celsius: float
    active_processes: int
    gpu_memory_percent: float
    swap_used_gb: float
    
@dataclass
class OptimizationAction:
    """Action taken for resource optimization."""
    timestamp: datetime
    action_type: str
    severity: str  # 'info', 'warning', 'critical', 'emergency'
    description: str
    parameters_changed: Dict[str, Any]
    resource_impact: Dict[str, float]
    success: bool
    
class UltimateResourceAuthority:
    """
    Supreme authority for resource management on Mac M2 Max.
    
    This agent monitors all system resources and has ultimate power to:
    1. Terminate resource-intensive processes
    2. Adjust training parameters in real-time
    3. Scale down operations automatically
    4. Prevent system overload
    5. Optimize for sustained performance
    """
    
    def __init__(self, 
                 limits: Optional[ResourceLimits] = None,
                 monitoring_interval: float = 1.0,
                 enable_emergency_shutdown: bool = True,
                 log_level: str = "INFO"):
        """Initialize the Ultimate Resource Authority."""
        
        self.limits = limits or ResourceLimits()
        self.monitoring_interval = monitoring_interval
        self.enable_emergency_shutdown = enable_emergency_shutdown
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_history: List[ResourceMetrics] = []
        self.actions_history: List[OptimizationAction] = []
        
        # Resource tracking
        self.baseline_metrics = None
        self.peak_metrics = None
        self.process_registry: Dict[int, Dict] = {}
        
        # Optimization callbacks
        self.optimization_callbacks: Dict[str, Callable] = {}
        self.emergency_callbacks: Dict[str, Callable] = {}
        
        # Safety measures
        self.emergency_stop_triggered = False
        self.critical_resource_breaches = 0
        self.max_critical_breaches = 3
        
        # Performance optimization cache
        self.optimization_cache = {}
        self.last_optimization_time = time.time()
        
        self.logger.info("🛡️ Ultimate Resource Authority initialized for Mac M2 Max")
        self.logger.info(f"Memory limits: {self.limits.max_memory_gb}GB max, {self.limits.warning_memory_gb}GB warning")
        self.logger.info(f"CPU limits: {self.limits.max_cpu_percent}% max, {self.limits.warning_cpu_percent}% warning")
        
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup comprehensive logging system."""
        logger = logging.getLogger('UltimateResourceAuthority')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            # Create logs directory
            log_dir = QUARK_ROOT / "logs" / "resource_monitoring"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # File handler for detailed logs
            file_handler = logging.FileHandler(
                log_dir / f"resource_authority_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler for important messages
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
        return logger
        
    def get_current_metrics(self) -> ResourceMetrics:
        """Get comprehensive current system metrics."""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_percent = memory.percent
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_cores = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if hasattr(self, '_last_disk_io'):
                time_diff = time.time() - self._last_disk_time
                read_diff = disk_io.read_bytes - self._last_disk_io.read_bytes
                write_diff = disk_io.write_bytes - self._last_disk_io.write_bytes
                disk_io_read_mbps = (read_diff / time_diff) / (1024**2)
                disk_io_write_mbps = (write_diff / time_diff) / (1024**2)
            else:
                disk_io_read_mbps = 0.0
                disk_io_write_mbps = 0.0
            
            self._last_disk_io = disk_io
            self._last_disk_time = time.time()
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                time_diff = time.time() - self._last_net_time
                net_diff = (net_io.bytes_sent + net_io.bytes_recv) - (self._last_net_io.bytes_sent + self._last_net_io.bytes_recv)
                network_io_mbps = (net_diff / time_diff) / (1024**2)
            else:
                network_io_mbps = 0.0
                
            self._last_net_io = net_io
            self._last_net_time = time.time()
            
            # Temperature (Mac-specific)
            temperature = self._get_mac_temperature()
            
            # Process count
            active_processes = len(psutil.pids())
            
            # GPU memory (if available)
            gpu_memory_percent = self._get_gpu_memory_usage()
            
            # Swap memory
            swap = psutil.swap_memory()
            swap_used_gb = swap.used / (1024**3)
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                memory_used_gb=memory_used_gb,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                cpu_cores_usage=cpu_cores,
                disk_io_read_mbps=disk_io_read_mbps,
                disk_io_write_mbps=disk_io_write_mbps,
                network_io_mbps=network_io_mbps,
                temperature_celsius=temperature,
                active_processes=active_processes,
                gpu_memory_percent=gpu_memory_percent,
                swap_used_gb=swap_used_gb
            )
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            # Return safe defaults
            return ResourceMetrics(
                timestamp=datetime.now(),
                memory_used_gb=0.0, memory_percent=0.0, cpu_percent=0.0,
                cpu_cores_usage=[], disk_io_read_mbps=0.0, disk_io_write_mbps=0.0,
                network_io_mbps=0.0, temperature_celsius=50.0, active_processes=0,
                gpu_memory_percent=0.0, swap_used_gb=0.0
            )
    
    def _get_mac_temperature(self) -> float:
        """Get Mac temperature using system commands."""
        try:
            # Try powermetrics for temperature
            result = subprocess.run([
                'sudo', 'powermetrics', '--samplers', 'smc', '-n', '1', '-i', '100'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CPU die temperature' in line:
                        temp_str = line.split(':')[-1].strip().replace('C', '')
                        return float(temp_str)
                        
        except Exception:
            pass
            
        # Fallback: estimate based on CPU usage
        cpu_percent = psutil.cpu_percent()
        estimated_temp = 40 + (cpu_percent / 100) * 30  # Rough estimation
        return min(estimated_temp, 90.0)
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage (Metal Performance Shaders on M2 Max)."""
        try:
            # For M2 Max, unified memory architecture means GPU uses system RAM
            # We'll estimate based on graphics-intensive processes
            gpu_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
                try:
                    if any(gpu_keyword in proc.info['name'].lower() 
                          for gpu_keyword in ['gpu', 'metal', 'opengl', 'graphics']):
                        gpu_processes.append(proc.info['memory_percent'])
                except:
                    continue
                    
            return sum(gpu_processes) if gpu_processes else 0.0
            
        except Exception:
            return 0.0
    
    def assess_resource_status(self, metrics: ResourceMetrics) -> Tuple[str, List[str]]:
        """Assess current resource status and identify issues."""
        issues = []
        severity = 'normal'
        
        # Memory assessment
        if metrics.memory_used_gb >= self.limits.critical_memory_gb:
            issues.append(f"CRITICAL: Memory usage {metrics.memory_used_gb:.1f}GB exceeds critical limit {self.limits.critical_memory_gb}GB")
            severity = 'critical'
        elif metrics.memory_used_gb >= self.limits.warning_memory_gb:
            issues.append(f"WARNING: Memory usage {metrics.memory_used_gb:.1f}GB exceeds warning limit {self.limits.warning_memory_gb}GB")
            if severity != 'critical':
                severity = 'warning'
        
        # CPU assessment
        if metrics.cpu_percent >= self.limits.critical_cpu_percent:
            issues.append(f"CRITICAL: CPU usage {metrics.cpu_percent:.1f}% exceeds critical limit {self.limits.critical_cpu_percent}%")
            severity = 'critical'
        elif metrics.cpu_percent >= self.limits.warning_cpu_percent:
            issues.append(f"WARNING: CPU usage {metrics.cpu_percent:.1f}% exceeds warning limit {self.limits.warning_cpu_percent}%")
            if severity not in ['critical']:
                severity = 'warning'
        
        # Temperature assessment
        if metrics.temperature_celsius >= self.limits.max_temperature_celsius:
            issues.append(f"CRITICAL: Temperature {metrics.temperature_celsius:.1f}°C exceeds maximum {self.limits.max_temperature_celsius}°C")
            severity = 'critical'
        elif metrics.temperature_celsius >= self.limits.warning_temperature_celsius:
            issues.append(f"WARNING: Temperature {metrics.temperature_celsius:.1f}°C exceeds warning {self.limits.warning_temperature_celsius}°C")
            if severity not in ['critical']:
                severity = 'warning'
        
        # Disk I/O assessment
        total_disk_io = metrics.disk_io_read_mbps + metrics.disk_io_write_mbps
        if total_disk_io >= self.limits.max_disk_io_mbps:
            issues.append(f"WARNING: Disk I/O {total_disk_io:.1f} MB/s exceeds limit {self.limits.max_disk_io_mbps} MB/s")
            if severity not in ['critical']:
                severity = 'warning'
        
        # Swap usage assessment
        if metrics.swap_used_gb > 1.0:  # Any significant swap usage is concerning
            issues.append(f"WARNING: Swap usage {metrics.swap_used_gb:.1f}GB indicates memory pressure")
            if severity not in ['critical']:
                severity = 'warning'
        
        return severity, issues
    
    def optimize_system_resources(self, metrics: ResourceMetrics, severity: str) -> List[OptimizationAction]:
        """Perform system-wide resource optimization."""
        actions = []
        
        if severity == 'critical':
            actions.extend(self._critical_optimizations(metrics))
        elif severity == 'warning':
            actions.extend(self._warning_optimizations(metrics))
        else:
            actions.extend(self._preventive_optimizations(metrics))
            
        return actions
    
    def _critical_optimizations(self, metrics: ResourceMetrics) -> List[OptimizationAction]:
        """Critical resource optimizations with immediate effect."""
        actions = []
        
        # Force garbage collection
        action = self._force_garbage_collection()
        if action:
            actions.append(action)
        
        # Terminate high-memory processes
        if metrics.memory_used_gb >= self.limits.critical_memory_gb:
            action = self._terminate_memory_intensive_processes()
            if action:
                actions.append(action)
        
        # Reduce CPU-intensive operations
        if metrics.cpu_percent >= self.limits.critical_cpu_percent:
            action = self._reduce_cpu_intensive_operations()
            if action:
                actions.append(action)
        
        # Emergency training parameter reduction
        action = self._emergency_parameter_reduction(metrics)
        if action:
            actions.append(action)
        
        # Increase critical breach counter
        self.critical_resource_breaches += 1
        if self.critical_resource_breaches >= self.max_critical_breaches:
            action = self._trigger_emergency_shutdown()
            if action:
                actions.append(action)
        
        return actions
    
    def _warning_optimizations(self, metrics: ResourceMetrics) -> List[OptimizationAction]:
        """Warning-level optimizations for sustained performance."""
        actions = []
        
        # Optimize memory usage
        if metrics.memory_used_gb >= self.limits.warning_memory_gb:
            action = self._optimize_memory_usage()
            if action:
                actions.append(action)
        
        # Scale down concurrent operations
        if metrics.cpu_percent >= self.limits.warning_cpu_percent:
            action = self._scale_down_operations()
            if action:
                actions.append(action)
        
        # Adjust training parameters
        action = self._adjust_training_parameters(metrics)
        if action:
            actions.append(action)
        
        return actions
    
    def _preventive_optimizations(self, metrics: ResourceMetrics) -> List[OptimizationAction]:
        """Preventive optimizations for optimal performance."""
        actions = []
        
        # Optimize caching
        action = self._optimize_caching()
        if action:
            actions.append(action)
        
        # Balance load across cores
        action = self._balance_cpu_load()
        if action:
            actions.append(action)
        
        return actions
    
    def _force_garbage_collection(self) -> Optional[OptimizationAction]:
        """Force Python garbage collection."""
        try:
            memory_before = psutil.virtual_memory().used / (1024**3)
            
            # Force garbage collection
            collected = gc.collect()
            
            memory_after = psutil.virtual_memory().used / (1024**3)
            memory_freed = memory_before - memory_after
            
            return OptimizationAction(
                timestamp=datetime.now(),
                action_type="garbage_collection",
                severity="critical",
                description=f"Forced garbage collection, freed {memory_freed:.2f}GB",
                parameters_changed={"collected_objects": collected},
                resource_impact={"memory_freed_gb": memory_freed},
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in garbage collection: {e}")
            return None
    
    def _terminate_memory_intensive_processes(self) -> Optional[OptimizationAction]:
        """Terminate the most memory-intensive non-essential processes."""
        try:
            terminated_processes = []
            memory_freed = 0.0
            
            # Get all processes sorted by memory usage
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
                try:
                    memory_mb = proc.info['memory_info'].rss / (1024**2)
                    if memory_mb > 100:  # Only consider processes using >100MB
                        processes.append((proc, memory_mb))
                except:
                    continue
            
            # Sort by memory usage (highest first)
            processes.sort(key=lambda x: x[1], reverse=True)
            
            # Terminate non-essential high-memory processes
            for proc, memory_mb in processes[:5]:  # Check top 5 memory users
                try:
                    proc_name = proc.info['name']
                    
                    # Skip essential system processes
                    if any(essential in proc_name.lower() for essential in [
                        'kernel', 'launchd', 'windowserver', 'finder', 'dock',
                        'system', 'cursor', 'python'  # Don't kill our own process
                    ]):
                        continue
                    
                    # Terminate the process
                    proc.terminate()
                    terminated_processes.append(proc_name)
                    memory_freed += memory_mb / 1024  # Convert to GB
                    
                    if len(terminated_processes) >= 3:  # Limit terminations
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Could not terminate process: {e}")
                    continue
            
            if terminated_processes:
                return OptimizationAction(
                    timestamp=datetime.now(),
                    action_type="process_termination",
                    severity="critical",
                    description=f"Terminated {len(terminated_processes)} memory-intensive processes",
                    parameters_changed={"terminated_processes": terminated_processes},
                    resource_impact={"memory_freed_gb": memory_freed},
                    success=True
                )
            
        except Exception as e:
            self.logger.error(f"Error terminating processes: {e}")
            
        return None
    
    def _reduce_cpu_intensive_operations(self) -> Optional[OptimizationAction]:
        """Reduce CPU-intensive operations across the system."""
        try:
            reductions = {}
            
            # Reduce training batch sizes
            if hasattr(self, '_current_batch_size'):
                old_batch_size = self._current_batch_size
                new_batch_size = max(8, old_batch_size // 2)
                self._current_batch_size = new_batch_size
                reductions['batch_size'] = f"{old_batch_size} -> {new_batch_size}"
            
            # Reduce parallel processes
            if hasattr(self, '_max_workers'):
                old_workers = self._max_workers
                new_workers = max(2, old_workers // 2)
                self._max_workers = new_workers
                reductions['max_workers'] = f"{old_workers} -> {new_workers}"
            
            # Lower process priority for CPU-intensive tasks
            current_process = psutil.Process()
            try:
                current_process.nice(10)  # Lower priority
                reductions['process_priority'] = "lowered to nice=10"
            except:
                pass
            
            return OptimizationAction(
                timestamp=datetime.now(),
                action_type="cpu_reduction",
                severity="critical",
                description="Reduced CPU-intensive operations",
                parameters_changed=reductions,
                resource_impact={"cpu_reduction_percent": 30},
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error reducing CPU operations: {e}")
            return None
    
    def _emergency_parameter_reduction(self, metrics: ResourceMetrics) -> Optional[OptimizationAction]:
        """Emergency reduction of all training parameters."""
        try:
            # Global parameter reduction for all registered processes
            reductions = {}
            
            # Notify all optimization callbacks about emergency
            for callback_name, callback in self.emergency_callbacks.items():
                try:
                    result = callback(metrics)
                    reductions[callback_name] = result
                except Exception as e:
                    self.logger.warning(f"Emergency callback {callback_name} failed: {e}")
            
            return OptimizationAction(
                timestamp=datetime.now(),
                action_type="emergency_reduction",
                severity="critical",
                description="Emergency parameter reduction across all systems",
                parameters_changed=reductions,
                resource_impact={"emergency_reduction": True},
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in emergency parameter reduction: {e}")
            return None
    
    def _trigger_emergency_shutdown(self) -> Optional[OptimizationAction]:
        """Trigger emergency shutdown sequence."""
        if not self.enable_emergency_shutdown:
            return None
            
        try:
            self.emergency_stop_triggered = True
            
            # Save current state
            self._save_emergency_state()
            
            # Notify all systems of emergency shutdown
            for callback_name, callback in self.emergency_callbacks.items():
                try:
                    callback("EMERGENCY_SHUTDOWN")
                except:
                    pass
            
            self.logger.critical("🚨 EMERGENCY SHUTDOWN TRIGGERED - System resource limits exceeded critically")
            
            return OptimizationAction(
                timestamp=datetime.now(),
                action_type="emergency_shutdown",
                severity="critical",
                description="Emergency shutdown triggered due to repeated critical resource breaches",
                parameters_changed={"emergency_stop": True},
                resource_impact={"system_shutdown": True},
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in emergency shutdown: {e}")
            return None
    
    def _optimize_memory_usage(self) -> Optional[OptimizationAction]:
        """Optimize memory usage across the system."""
        try:
            optimizations = {}
            memory_freed = 0.0
            
            # Clear caches
            if hasattr(self, 'optimization_cache'):
                cache_size = len(self.optimization_cache)
                self.optimization_cache.clear()
                optimizations['cache_cleared'] = cache_size
            
            # Optimize data structures
            memory_before = psutil.virtual_memory().used / (1024**3)
            gc.collect()
            memory_after = psutil.virtual_memory().used / (1024**3)
            memory_freed = memory_before - memory_after
            
            return OptimizationAction(
                timestamp=datetime.now(),
                action_type="memory_optimization",
                severity="warning",
                description="Optimized memory usage",
                parameters_changed=optimizations,
                resource_impact={"memory_freed_gb": memory_freed},
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}")
            return None
    
    def _scale_down_operations(self) -> Optional[OptimizationAction]:
        """Scale down concurrent operations."""
        try:
            scaling_changes = {}
            
            # Reduce concurrent operations
            if hasattr(self, '_max_concurrent_operations'):
                old_ops = self._max_concurrent_operations
                new_ops = max(2, old_ops - 2)
                self._max_concurrent_operations = new_ops
                scaling_changes['concurrent_operations'] = f"{old_ops} -> {new_ops}"
            
            # Notify optimization callbacks
            for callback_name, callback in self.optimization_callbacks.items():
                try:
                    result = callback("scale_down")
                    scaling_changes[callback_name] = result
                except:
                    pass
            
            return OptimizationAction(
                timestamp=datetime.now(),
                action_type="operation_scaling",
                severity="warning",
                description="Scaled down concurrent operations",
                parameters_changed=scaling_changes,
                resource_impact={"operations_reduced": True},
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error scaling down operations: {e}")
            return None
    
    def _adjust_training_parameters(self, metrics: ResourceMetrics) -> Optional[OptimizationAction]:
        """Dynamically adjust training parameters based on current resources."""
        try:
            adjustments = {}
            
            # Calculate optimal parameters based on available resources
            available_memory_percent = (1.0 - metrics.memory_percent / 100.0) * 100
            available_cpu_percent = 100.0 - metrics.cpu_percent
            
            # Adjust batch size based on memory availability
            if available_memory_percent < 30:  # Less than 30% memory available
                batch_size_factor = 0.5
            elif available_memory_percent < 50:  # Less than 50% memory available
                batch_size_factor = 0.7
            else:
                batch_size_factor = 1.0
                
            adjustments['batch_size_factor'] = batch_size_factor
            
            # Adjust worker count based on CPU availability
            if available_cpu_percent < 20:  # Less than 20% CPU available
                worker_factor = 0.5
            elif available_cpu_percent < 40:  # Less than 40% CPU available
                worker_factor = 0.7
            else:
                worker_factor = 1.0
                
            adjustments['worker_factor'] = worker_factor
            
            return OptimizationAction(
                timestamp=datetime.now(),
                action_type="parameter_adjustment",
                severity="warning",
                description="Adjusted training parameters based on resource availability",
                parameters_changed=adjustments,
                resource_impact={
                    "batch_size_reduction": 1.0 - batch_size_factor,
                    "worker_reduction": 1.0 - worker_factor
                },
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error adjusting training parameters: {e}")
            return None
    
    def _optimize_caching(self) -> Optional[OptimizationAction]:
        """Optimize caching strategies."""
        try:
            cache_optimizations = {}
            
            # Implement intelligent cache pruning
            if hasattr(self, 'optimization_cache'):
                cache_size_before = len(self.optimization_cache)
                
                # Remove old cache entries (older than 1 hour)
                current_time = time.time()
                keys_to_remove = []
                for key, (timestamp, value) in self.optimization_cache.items():
                    if current_time - timestamp > 3600:  # 1 hour
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.optimization_cache[key]
                
                cache_size_after = len(self.optimization_cache)
                cache_optimizations['cache_entries_removed'] = cache_size_before - cache_size_after
            
            return OptimizationAction(
                timestamp=datetime.now(),
                action_type="cache_optimization",
                severity="info",
                description="Optimized caching strategies",
                parameters_changed=cache_optimizations,
                resource_impact={"cache_optimized": True},
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing cache: {e}")
            return None
    
    def _balance_cpu_load(self) -> Optional[OptimizationAction]:
        """Balance CPU load across available cores."""
        try:
            # Set CPU affinity for better load distribution
            current_process = psutil.Process()
            cpu_count = psutil.cpu_count()
            
            # Use performance cores for main process (cores 0-7 on M2 Max)
            performance_cores = list(range(min(8, cpu_count)))
            
            try:
                current_process.cpu_affinity(performance_cores)
                affinity_set = True
            except:
                affinity_set = False
            
            return OptimizationAction(
                timestamp=datetime.now(),
                action_type="cpu_balancing",
                severity="info",
                description="Balanced CPU load across cores",
                parameters_changed={"cpu_affinity": performance_cores, "affinity_set": affinity_set},
                resource_impact={"cpu_balanced": True},
                success=affinity_set
            )
            
        except Exception as e:
            self.logger.error(f"Error balancing CPU load: {e}")
            return None
    
    def _save_emergency_state(self):
        """Save current system state during emergency."""
        try:
            emergency_dir = QUARK_ROOT / "logs" / "emergency_states"
            emergency_dir.mkdir(parents=True, exist_ok=True)
            
            emergency_state = {
                "timestamp": datetime.now().isoformat(),
                "metrics_history": [asdict(m) for m in self.metrics_history[-10:]],
                "actions_history": [asdict(a) for a in self.actions_history[-20:]],
                "resource_limits": asdict(self.limits),
                "critical_breaches": self.critical_resource_breaches
            }
            
            emergency_file = emergency_dir / f"emergency_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(emergency_file, 'w') as f:
                json.dump(emergency_state, f, indent=2, default=str)
                
            self.logger.info(f"Emergency state saved to {emergency_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving emergency state: {e}")
    
    def register_optimization_callback(self, name: str, callback: Callable) -> None:
        """Register a callback for optimization notifications."""
        self.optimization_callbacks[name] = callback
        self.logger.info(f"Registered optimization callback: {name}")
    
    def register_emergency_callback(self, name: str, callback: Callable) -> None:
        """Register a callback for emergency notifications."""
        self.emergency_callbacks[name] = callback
        self.logger.info(f"Registered emergency callback: {name}")
    
    def start_monitoring(self) -> None:
        """Start continuous resource monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self.emergency_stop_triggered = False
        self.critical_resource_breaches = 0
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("🛡️ Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("🛡️ Resource monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info("Starting resource monitoring loop")
        
        while self.is_monitoring and not self.emergency_stop_triggered:
            try:
                # Get current metrics
                current_metrics = self.get_current_metrics()
                self.metrics_history.append(current_metrics)
                
                # Keep history manageable
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # Assess resource status
                severity, issues = self.assess_resource_status(current_metrics)
                
                # Log issues
                if issues:
                    for issue in issues:
                        if severity == 'critical':
                            self.logger.critical(issue)
                        elif severity == 'warning':
                            self.logger.warning(issue)
                        else:
                            self.logger.info(issue)
                
                # Perform optimizations
                actions = self.optimize_system_resources(current_metrics, severity)
                self.actions_history.extend(actions)
                
                # Keep actions history manageable
                if len(self.actions_history) > 500:
                    self.actions_history = self.actions_history[-250:]
                
                # Reset critical breach counter if we're back to normal
                if severity == 'normal' and self.critical_resource_breaches > 0:
                    self.critical_resource_breaches = max(0, self.critical_resource_breaches - 1)
                    self.logger.info(f"Critical breach counter reset to {self.critical_resource_breaches}")
                
                # Update peak metrics
                if not self.peak_metrics or current_metrics.memory_used_gb > self.peak_metrics.memory_used_gb:
                    self.peak_metrics = current_metrics
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
        
        self.logger.info("Resource monitoring loop ended")
    
    @contextmanager
    def resource_monitoring_context(self):
        """Context manager for automatic resource monitoring."""
        try:
            self.start_monitoring()
            yield self
        finally:
            self.stop_monitoring()
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get current optimization recommendations."""
        if not self.metrics_history:
            return {"recommendations": "No metrics available yet"}
        
        current_metrics = self.metrics_history[-1]
        recommendations = []
        
        # Memory recommendations
        if current_metrics.memory_percent > 80:
            recommendations.append({
                "type": "memory",
                "severity": "high",
                "description": "Consider reducing batch sizes or enabling memory optimization",
                "action": "Reduce memory usage"
            })
        
        # CPU recommendations
        if current_metrics.cpu_percent > 85:
            recommendations.append({
                "type": "cpu",
                "severity": "high", 
                "description": "Consider reducing parallel processes or lowering CPU-intensive operations",
                "action": "Reduce CPU load"
            })
        
        # Temperature recommendations
        if current_metrics.temperature_celsius > 75:
            recommendations.append({
                "type": "thermal",
                "severity": "medium",
                "description": "System running hot, consider reducing workload intensity",
                "action": "Cool down system"
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": asdict(current_metrics),
            "recommendations": recommendations,
            "recent_actions": [asdict(a) for a in self.actions_history[-5:]]
        }
    
    def export_monitoring_report(self, output_file: Optional[Path] = None) -> Path:
        """Export comprehensive monitoring report."""
        if not output_file:
            output_file = QUARK_ROOT / "logs" / "resource_monitoring" / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "monitoring_duration": len(self.metrics_history) * self.monitoring_interval,
                "system_info": {
                    "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                    "cpu_count": psutil.cpu_count(),
                    "system": "Mac M2 Max"
                }
            },
            "resource_limits": asdict(self.limits),
            "metrics_summary": {
                "total_metrics": len(self.metrics_history),
                "peak_memory_gb": self.peak_metrics.memory_used_gb if self.peak_metrics else 0,
                "peak_cpu_percent": max([m.cpu_percent for m in self.metrics_history]) if self.metrics_history else 0
            },
            "actions_summary": {
                "total_actions": len(self.actions_history),
                "critical_actions": len([a for a in self.actions_history if a.severity == 'critical']),
                "warning_actions": len([a for a in self.actions_history if a.severity == 'warning'])
            },
            "emergency_info": {
                "emergency_triggered": self.emergency_stop_triggered,
                "critical_breaches": self.critical_resource_breaches
            },
            "recent_metrics": [asdict(m) for m in self.metrics_history[-50:]],
            "recent_actions": [asdict(a) for a in self.actions_history[-50:]]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring report exported to {output_file}")
        return output_file

def create_ultimate_authority() -> UltimateResourceAuthority:
    """Create and configure the Ultimate Resource Authority for Mac M2 Max."""
    
    # Optimized limits for Mac M2 Max with 64GB RAM
    limits = ResourceLimits(
        max_memory_gb=48.0,          # Use up to 48GB, reserve 16GB for system
        warning_memory_gb=40.0,       # Warning at 40GB
        critical_memory_gb=44.0,      # Critical at 44GB
        max_cpu_percent=85.0,         # Max 85% CPU usage
        warning_cpu_percent=70.0,     # Warning at 70%
        critical_cpu_percent=80.0,    # Critical at 80%
        max_concurrent_processes=8,   # Optimal for 12-core processor
        max_threads_per_process=4,    # Balanced threading
        max_temperature_celsius=85.0, # Silicon thermal limit
        warning_temperature_celsius=75.0
    )
    
    authority = UltimateResourceAuthority(
        limits=limits,
        monitoring_interval=1.0,      # Monitor every second
        enable_emergency_shutdown=True,
        log_level="INFO"
    )
    
    return authority

# Example usage and testing
if __name__ == "__main__":
    # Create the authority
    authority = create_ultimate_authority()
    
    print("🛡️ Ultimate Resource Authority initialized for Mac M2 Max")
    print(f"Memory limits: {authority.limits.max_memory_gb}GB max")
    print(f"CPU limits: {authority.limits.max_cpu_percent}% max")
    print("\nStarting resource monitoring for 30 seconds...")
    
    try:
        with authority.resource_monitoring_context():
            # Simulate some work
            for i in range(30):
                time.sleep(1)
                if i % 10 == 0:
                    metrics = authority.get_current_metrics()
                    print(f"Memory: {metrics.memory_used_gb:.1f}GB ({metrics.memory_percent:.1f}%), CPU: {metrics.cpu_percent:.1f}%")
                    
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    
    # Export final report
    report_file = authority.export_monitoring_report()
    print(f"\nMonitoring report saved to: {report_file}")
    
    # Show recommendations
    recommendations = authority.get_optimization_recommendations()
    print(f"\nCurrent recommendations: {len(recommendations.get('recommendations', []))}")
