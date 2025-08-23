#!/usr/bin/env python3
"""
Performance Monitor for Live Streaming System.
Tracks metrics, performance, and system health.
"""

import time
import threading
import psutil
import json
from collections import defaultdict, deque
from typing import Dict, List, Any
from datetime import datetime

from testing.visualizations.visual_utils import live_series


class PerformanceMonitor:
    """Monitors performance of the live streaming system."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = defaultdict(list)
        self.performance_data = deque(maxlen=1000)
        self.monitoring = False
        self.monitor_thread = None
        
        # Performance counters
        self.message_count = 0
        self.client_count = 0
        self.error_count = 0
        self.latency_sum = 0
        self.latency_count = 0
        
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("📊 Performance monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._collect_system_metrics()
                self._collect_performance_metrics()
                time.sleep(1)  # Collect metrics every second
            except Exception as e:
                print(f"⚠️ Performance monitoring error: {e}")
                
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            live_series("perf_cpu_usage", cpu_percent, int(time.time() - self.start_time))
            
            # Memory usage
            memory = psutil.virtual_memory()
            live_series("perf_memory_usage", memory.percent, int(time.time() - self.start_time))
            live_series("perf_memory_available", memory.available / (1024**3), int(time.time() - self.start_time))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            live_series("perf_disk_usage", disk.percent, int(time.time() - self.start_time))
            
        except Exception as e:
            print(f"⚠️ Error collecting system metrics: {e}")
            
    def _collect_performance_metrics(self):
        """Collect performance metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Message throughput
        if self.latency_count > 0:
            avg_latency = self.latency_sum / self.latency_count
            live_series("perf_avg_latency", avg_latency, int(uptime))
        
        # Message rate
        live_series("perf_message_count", self.message_count, int(uptime))
        live_series("perf_client_count", self.client_count, int(uptime))
        live_series("perf_error_count", self.error_count, int(uptime))
        
        # Performance summary
        live_series("perf_summary", {
            "uptime": uptime,
            "messages": self.message_count,
            "clients": self.client_count,
            "errors": self.error_count,
            "avg_latency": self.latency_sum / max(self.latency_count, 1)
        }, int(uptime))
        
    def record_message(self, latency_ms: float = None):
        """Record a message being sent."""
        self.message_count += 1
        if latency_ms is not None:
            self.latency_sum += latency_ms
            self.latency_count += 1
            
    def record_client_connection(self, connected: bool = True):
        """Record client connection/disconnection."""
        if connected:
            self.client_count += 1
        else:
            self.client_count = max(0, self.client_count - 1)
            
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "message_count": self.message_count,
            "client_count": self.client_count,
            "error_count": self.error_count,
            "avg_latency_ms": self.latency_sum / max(self.latency_count, 1),
            "message_rate": self.message_count / max(uptime, 1),
            "error_rate": self.error_count / max(uptime, 1)
        }
        
    def export_metrics(self, filepath: str = None):
        """Export metrics to file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"testing/visualizations/outputs/performance_metrics_{timestamp}.json"
            
        report = self.get_performance_report()
        
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
                
            print(f"📊 Performance metrics exported to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error exporting metrics: {e}")
            return None


# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor():
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def start_performance_monitoring():
    """Start performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()
    return monitor

def stop_performance_monitoring():
    """Stop performance monitoring."""
    if _performance_monitor:
        _performance_monitor.stop_monitoring()

def record_performance_event(event_type: str, **kwargs):
    """Record a performance event."""
    monitor = get_performance_monitor()
    
    if event_type == "message":
        monitor.record_message(kwargs.get("latency_ms"))
    elif event_type == "client":
        monitor.record_client_connection(kwargs.get("connected", True))
    elif event_type == "error":
        monitor.record_error()
    else:
        print(f"⚠️ Unknown performance event type: {event_type}")


if __name__ == "__main__":
    # Test the performance monitor
    print("🚀 Performance Monitor Test")
    print("=" * 40)
    
    monitor = start_performance_monitoring()
    
    # Simulate some activity
    for i in range(10):
        record_performance_event("message", latency_ms=10 + i)
        record_performance_event("client", connected=True)
        time.sleep(0.5)
    
    # Generate report
    report = monitor.get_performance_report()
    print(f"\n📊 Performance Report:")
    for key, value in report.items():
        print(f"   {key}: {value}")
    
    # Export metrics
    monitor.export_metrics()
    
    # Stop monitoring
    stop_performance_monitoring()
    print("\n✅ Performance monitor test completed")
