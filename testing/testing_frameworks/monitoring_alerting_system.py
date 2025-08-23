#!/usr/bin/env python3
"""
🔔 Monitoring and Alerting System
Deploys monitoring and alerting systems for QUARK's experimentation protocols
"""

import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class SystemMetric:
    """System metric for monitoring"""
    name: str
    value: float
    unit: str
    timestamp: float
    status: str  # "normal", "warning", "critical"

@dataclass
class Alert:
    """System alert"""
    id: str
    severity: str  # "info", "warning", "critical"
    message: str
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False

class MonitoringAlertingSystem:
    """Monitoring and alerting system for QUARK"""
    
    def __init__(self):
        self.metrics: List[SystemMetric] = []
        self.alerts: List[Alert] = []
        self.alert_count = 0
        
        # Thresholds for alerts
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "response_time": {"warning": 2.0, "critical": 5.0}
        }
        
        print("🔔 Monitoring and Alerting System initialized")
    
    def collect_system_metrics(self) -> List[SystemMetric]:
        """Collect current system metrics"""
        current_time = time.time()
        metrics = []
        
        # Simulate system metrics collection
        # In a real implementation, these would come from system monitoring tools
        
        # CPU usage (simulated)
        cpu_usage = 45.2  # Simulated value
        cpu_status = self._get_metric_status("cpu_usage", cpu_usage)
        metrics.append(SystemMetric(
            name="CPU Usage",
            value=cpu_usage,
            unit="%",
            timestamp=current_time,
            status=cpu_status
        ))
        
        # Memory usage (simulated)
        memory_usage = 62.8  # Simulated value
        memory_status = self._get_metric_status("memory_usage", memory_usage)
        metrics.append(SystemMetric(
            name="Memory Usage",
            value=memory_usage,
            unit="%",
            timestamp=current_time,
            status=memory_status
        ))
        
        # Disk usage (simulated)
        disk_usage = 78.5  # Simulated value
        disk_status = self._get_metric_status("disk_usage", disk_usage)
        metrics.append(SystemMetric(
            name="Disk Usage",
            value=disk_usage,
            unit="%",
            timestamp=current_time,
            status=disk_status
        ))
        
        # Response time (simulated)
        response_time = 1.2  # Simulated value in seconds
        response_status = self._get_metric_status("response_time", response_time)
        metrics.append(SystemMetric(
            name="Response Time",
            value=response_time,
            unit="s",
            timestamp=current_time,
            status=response_status
        ))
        
        self.metrics = metrics
        return metrics
    
    def _get_metric_status(self, metric_name: str, value: float) -> str:
        """Determine metric status based on thresholds"""
        if metric_name not in self.thresholds:
            return "normal"
        
        thresholds = self.thresholds[metric_name]
        
        if value >= thresholds["critical"]:
            return "critical"
        elif value >= thresholds["warning"]:
            return "warning"
        else:
            return "normal"
    
    def check_alerts(self) -> List[Alert]:
        """Check for new alerts based on current metrics"""
        new_alerts = []
        
        for metric in self.metrics:
            if metric.status in ["warning", "critical"]:
                # Check if we already have an alert for this metric
                existing_alert = next(
                    (a for a in self.alerts if a.message.startswith(f"{metric.name}") and not a.resolved),
                    None
                )
                
                if not existing_alert:
                    # Create new alert
                    alert = Alert(
                        id=f"alert_{self.alert_count}",
                        severity=metric.status,
                        message=f"{metric.name}: {metric.value}{metric.unit} ({metric.status})",
                        timestamp=time.time()
                    )
                    
                    new_alerts.append(alert)
                    self.alerts.append(alert)
                    self.alert_count += 1
                    
                    print(f"🚨 New {metric.status} alert: {alert.message}")
        
        return new_alerts
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.metrics:
            self.collect_system_metrics()
        
        # Count metrics by status
        normal_count = len([m for m in self.metrics if m.status == "normal"])
        warning_count = len([m for m in self.metrics if m.status == "warning"])
        critical_count = len([m for m in self.metrics if m.status == "critical"])
        
        # Determine overall health
        if critical_count > 0:
            overall_health = "critical"
        elif warning_count > 0:
            overall_health = "warning"
        else:
            overall_health = "healthy"
        
        # Calculate health score (0-100)
        total_metrics = len(self.metrics)
        health_score = (normal_count / total_metrics) * 100 if total_metrics > 0 else 0
        
        return {
            "overall_health": overall_health,
            "health_score": health_score,
            "metrics_summary": {
                "normal": normal_count,
                "warning": warning_count,
                "critical": critical_count,
                "total": total_metrics
            },
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "timestamp": time.time()
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                print(f"✅ Alert {alert_id} acknowledged")
                return True
        
        print(f"❌ Alert {alert_id} not found")
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                print(f"✅ Alert {alert_id} resolved")
                return True
        
        print(f"❌ Alert {alert_id} not found")
        return False
    
    def generate_health_report(self) -> str:
        """Generate a human-readable health report"""
        health = self.get_system_health()
        
        report = []
        report.append("# 🔔 QUARK System Health Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall health
        report.append(f"## 📊 Overall Health: {health['overall_health'].upper()}")
        report.append(f"**Health Score**: {health['health_score']:.1f}%")
        report.append(f"**Active Alerts**: {health['active_alerts']}")
        report.append("")
        
        # Metrics summary
        report.append("## 📈 Metrics Summary")
        summary = health['metrics_summary']
        report.append(f"- **Normal**: {summary['normal']}")
        report.append(f"- **Warning**: {summary['warning']}")
        report.append(f"- **Critical**: {summary['critical']}")
        report.append(f"- **Total**: {summary['total']}")
        report.append("")
        
        # Current metrics
        report.append("## 🔍 Current Metrics")
        for metric in self.metrics:
            status_emoji = {"normal": "✅", "warning": "⚠️", "critical": "🚨"}
            report.append(f"- **{metric.name}**: {metric.value}{metric.unit} {status_emoji.get(metric.status, '❓')} ({metric.status})")
        
        # Active alerts
        if health['active_alerts'] > 0:
            report.append("")
            report.append("## 🚨 Active Alerts")
            for alert in self.alerts:
                if not alert.resolved:
                    severity_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
                    report.append(f"- **{alert.severity.upper()}**: {alert.message}")
        
        return "\n".join(report)
    
    def save_health_report(self, filename: str = "system_health_report.md"):
        """Save health report to file"""
        report = self.generate_health_report()
        
        try:
            with open(filename, 'w') as f:
                f.write(report)
            print(f"✅ Health report saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving health report: {e}")
    
    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run a complete monitoring cycle"""
        print("🔄 Running monitoring cycle...")
        
        # Collect metrics
        metrics = self.collect_system_metrics()
        
        # Check for alerts
        new_alerts = self.check_alerts()
        
        # Get system health
        health = self.get_system_health()
        
        # Generate report
        report = self.generate_health_report()
        
        cycle_result = {
            "metrics_collected": len(metrics),
            "new_alerts": len(new_alerts),
            "system_health": health,
            "report_generated": bool(report)
        }
        
        print(f"✅ Monitoring cycle completed: {health['overall_health']} health, {len(new_alerts)} new alerts")
        return cycle_result

def main():
    """Main function to demonstrate monitoring system"""
    print("🔔 QUARK Monitoring and Alerting System")
    print("=" * 50)
    
    # Initialize system
    monitoring = MonitoringAlertingSystem()
    
    # Run monitoring cycle
    print("\n🔄 Running monitoring cycle...")
    result = monitoring.run_monitoring_cycle()
    
    # Display results
    print(f"\n📊 Monitoring Results:")
    print(f"   Metrics collected: {result['metrics_collected']}")
    print(f"   New alerts: {result['new_alerts']}")
    print(f"   System health: {result['system_health']['overall_health']}")
    print(f"   Health score: {result['system_health']['health_score']:.1f}%")
    
    # Generate and save health report
    print("\n📄 Generating health report...")
    monitoring.save_health_report()
    
    print("\n✅ Monitoring system demonstration complete!")
    return monitoring

if __name__ == "__main__":
    try:
        monitoring = main()
    except Exception as e:
        print(f"❌ Monitoring system failed: {e}")
        import traceback
        traceback.print_exc()
