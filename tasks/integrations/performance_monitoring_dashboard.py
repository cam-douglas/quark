#!/usr/bin/env python3
"""
📊 Performance Monitoring Dashboard

This module provides real-time monitoring of the biological brain agent integration
performance, health metrics, and system status.
"""

import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors performance of biological brain agent integration"""
    
    def __init__(self, brain_agent_path: str = "../../brain_architecture/neural_core/biological_brain_agent.py",
                 task_system_path: str = "../../tasks"):
        self.brain_agent_path = Path(brain_agent_path)
        self.task_system_path = Path(task_system_path)
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance metrics
        self.performance_metrics = {
            "start_time": datetime.now(),
            "total_checks": 0,
            "successful_checks": 0,
            "failed_checks": 0,
            "last_check": None,
            "check_interval": 30,  # seconds
            "health_history": [],
            "performance_history": [],
            "constraint_violations": []
        }
        
        # Health thresholds
        self.health_thresholds = {
            "max_response_time": 5.0,  # seconds
            "min_success_rate": 0.95,  # 95%
            "max_cognitive_load": 0.8,
            "min_working_memory": 0.3,
            "min_energy_level": 0.4
        }
        
        logger.info("📊 Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("🛑 Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Perform health check
                health_status = self.perform_health_check()
                
                # Record health status
                self.performance_metrics["health_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "status": health_status
                })
                
                # Keep only last 100 health records
                if len(self.performance_metrics["health_history"]) > 100:
                    self.performance_metrics["health_history"] = self.performance_metrics["health_history"][-100:]
                
                # Update metrics
                self.performance_metrics["total_checks"] += 1
                if health_status["overall_health"] == "healthy":
                    self.performance_metrics["successful_checks"] += 1
                else:
                    self.performance_metrics["failed_checks"] += 1
                
                self.performance_metrics["last_check"] = datetime.now().isoformat()
                
                # Sleep until next check
                time.sleep(self.performance_metrics["check_interval"])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.performance_metrics["check_interval"])
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        start_time = time.time()
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "issues": [],
            "recommendations": [],
            "metrics": {},
            "constraints": {}
        }
        
        try:
            # Check 1: Brain agent file accessibility
            if not self.brain_agent_path.exists():
                health_status["issues"].append("Biological brain agent file not found")
                health_status["overall_health"] = "critical"
            else:
                # Check 2: File modification time
                file_age = datetime.now() - datetime.fromtimestamp(self.brain_agent_path.stat().st_mtime)
                if file_age > timedelta(hours=24):
                    health_status["issues"].append("Brain agent file is older than 24 hours")
                    health_status["recommendations"].append("Consider updating brain agent")
                
                # Check 3: File size and content
                file_size = self.brain_agent_path.stat().st_size
                if file_size < 1000:  # Less than 1KB
                    health_status["issues"].append("Brain agent file seems too small")
                    health_status["overall_health"] = "degraded"
                
                health_status["metrics"]["file_size"] = file_size
                health_status["metrics"]["file_age_hours"] = file_age.total_seconds() / 3600
            
            # Check 4: Task system accessibility
            if not self.task_system_path.exists():
                health_status["issues"].append("Task system directory not found")
                health_status["overall_health"] = "critical"
            else:
                # Check key task system files
                key_files = ["TASK_STATUS.md", "goals/README.md", "active_tasks/README.md"]
                missing_files = []
                
                for file_name in key_files:
                    file_path = self.task_system_path / file_name
                    if not file_path.exists():
                        missing_files.append(file_name)
                
                if missing_files:
                    health_status["issues"].append(f"Missing task system files: {', '.join(missing_files)}")
                    health_status["overall_health"] = "degraded"
                
                health_status["metrics"]["task_files_accessible"] = len(key_files) - len(missing_files)
                health_status["metrics"]["total_task_files"] = len(key_files)
            
            # Check 5: Integration files
            integration_files = [
                "BIOLOGICAL_BRAIN_INTEGRATION_SUMMARY.md",
                "test_biological_integration.py"
            ]
            
            missing_integration_files = []
            for file_name in integration_files:
                file_path = Path(file_name)
                if not file_path.exists():
                    missing_integration_files.append(file_name)
            
            if missing_integration_files:
                health_status["issues"].append(f"Missing integration files: {', '.join(missing_integration_files)}")
                health_status["overall_health"] = "degraded"
            
            # Check 6: Performance metrics
            success_rate = self.performance_metrics["successful_checks"] / max(self.performance_metrics["total_checks"], 1)
            if success_rate < self.health_thresholds["min_success_rate"]:
                health_status["issues"].append(f"Success rate below threshold: {success_rate:.2%}")
                health_status["overall_health"] = "degraded"
            
            health_status["metrics"]["success_rate"] = success_rate
            health_status["metrics"]["total_checks"] = self.performance_metrics["total_checks"]
            health_status["metrics"]["failed_checks"] = self.performance_metrics["failed_checks"]
            
            # Check 7: Response time
            response_time = time.time() - start_time
            if response_time > self.health_thresholds["max_response_time"]:
                health_status["issues"].append(f"Health check response time too slow: {response_time:.2f}s")
                health_status["overall_health"] = "degraded"
            
            health_status["metrics"]["response_time"] = response_time
            
            # Check 8: Biological constraints simulation
            # Simulate brain state for constraint checking
            simulated_brain_state = self._simulate_brain_state()
            health_status["constraints"] = simulated_brain_state
            
            # Check constraint violations
            constraint_violations = []
            if simulated_brain_state["cognitive_load"] > self.health_thresholds["max_cognitive_load"]:
                constraint_violations.append("Cognitive load exceeds maximum")
            
            if simulated_brain_state["working_memory"] < self.health_thresholds["min_working_memory"]:
                constraint_violations.append("Working memory below minimum")
            
            if simulated_brain_state["energy_level"] < self.health_thresholds["min_energy_level"]:
                constraint_violations.append("Energy level below minimum")
            
            if constraint_violations:
                health_status["issues"].extend(constraint_violations)
                health_status["overall_health"] = "degraded"
                self.performance_metrics["constraint_violations"].append({
                    "timestamp": datetime.now().isoformat(),
                    "violations": constraint_violations
                })
            
            # Generate recommendations
            if not health_status["recommendations"]:
                if health_status["overall_health"] == "healthy":
                    health_status["recommendations"].append("System is healthy - continue monitoring")
                else:
                    health_status["recommendations"].append("Review and resolve identified issues")
                    health_status["recommendations"].append("Check system logs for additional details")
            
        except Exception as e:
            health_status["issues"].append(f"Health check error: {str(e)}")
            health_status["overall_health"] = "critical"
            health_status["recommendations"].append("Investigate health check failures")
        
        return health_status
    
    def _simulate_brain_state(self) -> Dict[str, float]:
        """Simulate current brain state for constraint checking"""
        # This is a simplified simulation - in a real system, this would read actual brain state
        
        # Simulate varying brain state based on time and load
        current_hour = datetime.now().hour
        base_load = 0.3 + (0.2 * (current_hour % 8) / 8)  # Varies throughout the day
        
        # Add some randomness to simulate real conditions
        import random
        random.seed(int(time.time() / 300))  # Change every 5 minutes
        
        cognitive_load = min(1.0, base_load + random.uniform(-0.1, 0.1))
        working_memory = max(0.0, 1.0 - cognitive_load + random.uniform(-0.1, 0.1))
        energy_level = max(0.0, 1.0 - (current_hour / 24) + random.uniform(-0.1, 0.1))
        
        return {
            "cognitive_load": round(cognitive_load, 3),
            "working_memory": round(working_memory, 3),
            "energy_level": round(energy_level, 3),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        current_health = self.perform_health_check()
        
        # Calculate performance statistics
        total_checks = self.performance_metrics["total_checks"]
        success_rate = self.performance_metrics["successful_checks"] / max(total_checks, 1)
        
        # Calculate uptime
        uptime = datetime.now() - self.performance_metrics["start_time"]
        
        # Get recent constraint violations
        recent_violations = self.performance_metrics["constraint_violations"][-10:]  # Last 10
        
        return {
            "monitoring_status": {
                "active": self.monitoring_active,
                "started": self.performance_metrics["start_time"].isoformat(),
                "uptime_seconds": uptime.total_seconds(),
                "check_interval": self.performance_metrics["check_interval"]
            },
            "performance_metrics": {
                "total_checks": total_checks,
                "successful_checks": self.performance_metrics["successful_checks"],
                "failed_checks": self.performance_metrics["failed_checks"],
                "success_rate": round(success_rate, 3),
                "last_check": self.performance_metrics["last_check"]
            },
            "current_health": current_health,
            "constraint_violations": {
                "total": len(self.performance_metrics["constraint_violations"]),
                "recent": recent_violations
            },
            "health_thresholds": self.health_thresholds
        }
    
    def generate_performance_report(self) -> str:
        """Generate human-readable performance report"""
        summary = self.get_performance_summary()
        
        report = f"""# 📊 Performance Monitoring Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Monitoring Status**: {'🟢 ACTIVE' if summary['monitoring_status']['active'] else '🔴 INACTIVE'}

## 📈 Performance Overview

**Uptime**: {summary['monitoring_status']['uptime_seconds']:.0f} seconds
**Total Checks**: {summary['performance_metrics']['total_checks']}
**Success Rate**: {summary['performance_metrics']['success_rate']:.1%}
**Last Check**: {summary['performance_metrics']['last_check'] or 'Never'}

## 🏥 Current Health Status

**Overall Health**: {summary['current_health']['overall_health'].upper()}

### Issues Found
"""
        
        if summary['current_health']['issues']:
            for issue in summary['current_health']['issues']:
                report += f"- ❌ {issue}\n"
        else:
            report += "- ✅ No issues found\n"
        
        report += f"""
### Recommendations
"""
        
        for rec in summary['current_health']['recommendations']:
            report += f"- 💡 {rec}\n"
        
        report += f"""
## 🧠 Brain State Simulation

**Cognitive Load**: {summary['current_health']['constraints']['cognitive_load']:.3f}
**Working Memory**: {summary['current_health']['constraints']['working_memory']:.3f}
**Energy Level**: {summary['current_health']['constraints']['energy_level']:.3f}

## 🚨 Constraint Violations

**Total Violations**: {summary['constraint_violations']['total']}
**Recent Violations**: {len(summary['constraint_violations']['recent'])}

"""
        
        if summary['constraint_violations']['recent']:
            for violation in summary['constraint_violations']['recent']:
                report += f"- **{violation['timestamp']}**: {', '.join(violation['violations'])}\n"
        else:
            report += "- ✅ No recent violations\n"
        
        report += f"""
## ⚙️ Health Thresholds

- **Max Response Time**: {summary['health_thresholds']['max_response_time']}s
- **Min Success Rate**: {summary['health_thresholds']['min_success_rate']:.1%}
- **Max Cognitive Load**: {summary['health_thresholds']['max_cognitive_load']:.1f}
- **Min Working Memory**: {summary['health_thresholds']['min_working_memory']:.1f}
- **Min Energy Level**: {summary['health_thresholds']['min_energy_level']:.1f}

---
*Generated by Performance Monitoring Dashboard*
"""
        
        return report
    
    def save_performance_report(self, output_path: str = "performance_report.md"):
        """Save performance report to file"""
        try:
            report_content = self.generate_performance_report()
            
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Performance report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
            return False

def main():
    """Main function to demonstrate performance monitoring"""
    print("📊 Performance Monitoring Dashboard")
    print("=" * 50)
    
    # Create monitor
    monitor = PerformanceMonitor()
    
    # Start monitoring
    print("🚀 Starting performance monitoring...")
    monitor.start_monitoring()
    
    try:
        # Let it run for a few cycles
        print("📈 Monitoring for 2 minutes...")
        time.sleep(120)
        
        # Generate and display report
        print("\n📊 Generating performance report...")
        report = monitor.generate_performance_report()
        print(report)
        
        # Save report
        monitor.save_performance_report("performance_report.md")
        print("\n💾 Performance report saved to performance_report.md")
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring interrupted by user")
    
    finally:
        # Stop monitoring
        print("🛑 Stopping performance monitoring...")
        monitor.stop_monitoring()
        
        # Final summary
        summary = monitor.get_performance_summary()
        print(f"\n📊 Final Summary:")
        print(f"   Total Checks: {summary['performance_metrics']['total_checks']}")
        print(f"   Success Rate: {summary['performance_metrics']['success_rate']:.1%}")
        print(f"   Uptime: {summary['monitoring_status']['uptime_seconds']:.0f} seconds")
        
        print("\n✅ Performance monitoring demonstration complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        import traceback
        traceback.print_exc()
