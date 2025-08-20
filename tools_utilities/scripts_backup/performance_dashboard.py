"""
Performance Monitoring Dashboard
Purpose: Monitor and visualize performance of integrated consciousness-brain system
Inputs: Performance metrics, system status, real-time data
Outputs: Visual dashboard, performance reports, alerts
Seeds: Performance thresholds, monitoring patterns
Dependencies: matplotlib, numpy, integrated_main_consciousness, connect_brain_simulation
"""

import os, sys
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle, Circle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Install with: pip install matplotlib")

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Performance data storage
        self.performance_history = {
            'timestamps': [],
            'consciousness_levels': [],
            'neural_activity': [],
            'brain_step_times': [],
            'integration_latency': [],
            'memory_usage': [],
            'pfc_firing_rates': [],
            'loop_stability': []
        }
        
        # Performance thresholds
        self.thresholds = {
            'consciousness_level': {'warning': 0.3, 'critical': 0.1},
            'neural_activity': {'warning': 0.2, 'critical': 0.05},
            'brain_step_time': {'warning': 0.5, 'critical': 1.0},
            'integration_latency': {'warning': 0.1, 'critical': 0.3}
        }
        
        # Alert system
        self.alerts = []
        self.alert_history = []
        
        # Dashboard state
        self.dashboard_state = {
            'last_update': None,
            'total_updates': 0,
            'system_health': 'unknown',
            'performance_score': 0.0
        }
        
        print("📊 Initializing Performance Dashboard...")
    
    def start_monitoring(self, integrated_consciousness=None, brain_connector=None):
        """Start performance monitoring"""
        if self.monitoring_active:
            return
        
        self.integrated_consciousness = integrated_consciousness
        self.brain_connector = brain_connector
        
        print("🚀 Starting performance monitoring...")
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        print("✅ Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        print("🛑 Stopping performance monitoring...")
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        print("✅ Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect performance data
                self._collect_performance_data()
                
                # Update dashboard state
                self._update_dashboard_state()
                
                # Check performance thresholds
                self._check_performance_thresholds()
                
                # Update timestamp
                self.dashboard_state['last_update'] = datetime.now().isoformat()
                self.dashboard_state['total_updates'] += 1
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"⚠️  Performance monitoring error: {e}")
                time.sleep(5)
    
    def _collect_performance_data(self):
        """Collect performance data from all systems"""
        current_time = datetime.now()
        
        # Collect consciousness data
        if self.integrated_consciousness:
            try:
                report = self.integrated_consciousness.get_integrated_report()
                consciousness_level = report['unified_state']['consciousness_level']
                neural_activity = report['unified_state'].get('neural_activity', 0.0)
                
                self.performance_history['timestamps'].append(current_time)
                self.performance_history['consciousness_levels'].append(consciousness_level)
                self.performance_history['neural_activity'].append(neural_activity)
                
            except Exception as e:
                print(f"⚠️  Consciousness data collection error: {e}")
        
        # Collect brain simulation data
        if self.brain_connector:
            try:
                status = self.brain_connector.get_connection_status()
                
                brain_step_time = status['performance_metrics'].get('brain_step_time', 0.0)
                integration_latency = status['performance_metrics'].get('integration_latency', 0.0)
                memory_usage = status['performance_metrics'].get('memory_usage', 0.0)
                
                self.performance_history['brain_step_times'].append(brain_step_time)
                self.performance_history['integration_latency'].append(integration_latency)
                self.performance_history['memory_usage'].append(memory_usage)
                
                # Brain metrics
                brain_metrics = status.get('brain_metrics', {})
                pfc_rate = brain_metrics.get('pfc_firing_rate', 0.0)
                loop_stability = brain_metrics.get('loop_stability', 0.0)
                
                self.performance_history['pfc_firing_rates'].append(pfc_rate)
                self.performance_history['loop_stability'].append(loop_stability)
                
            except Exception as e:
                print(f"⚠️  Brain simulation data collection error: {e}")
        
        # Limit history size
        max_history = 1000
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key] = self.performance_history[key][-max_history:]
    
    def _update_dashboard_state(self):
        """Update dashboard state based on performance data"""
        if not self.performance_history['timestamps']:
            return
        
        # Calculate performance score
        consciousness_score = np.mean(self.performance_history['consciousness_levels'][-10:]) if self.performance_history['consciousness_levels'] else 0.0
        neural_score = np.mean(self.performance_history['neural_activity'][-10:]) if self.performance_history['neural_activity'] else 0.0
        step_time_score = 1.0 - min(1.0, np.mean(self.performance_history['brain_step_times'][-10:]) if self.performance_history['brain_step_times'] else 0.0)
        
        # Weighted performance score
        self.dashboard_state['performance_score'] = (
            consciousness_score * 0.4 + 
            neural_score * 0.3 + 
            step_time_score * 0.3
        )
        
        # Determine system health
        if self.dashboard_state['performance_score'] > 0.8:
            self.dashboard_state['system_health'] = 'excellent'
        elif self.dashboard_state['performance_score'] > 0.6:
            self.dashboard_state['system_health'] = 'good'
        elif self.dashboard_state['performance_score'] > 0.4:
            self.dashboard_state['system_health'] = 'fair'
        elif self.dashboard_state['performance_score'] > 0.2:
            self.dashboard_state['system_health'] = 'poor'
        else:
            self.dashboard_state['system_health'] = 'critical'
    
    def _check_performance_thresholds(self):
        """Check performance against thresholds and generate alerts"""
        if not self.performance_history['timestamps']:
            return
        
        current_time = datetime.now()
        
        # Check consciousness level
        if self.performance_history['consciousness_levels']:
            current_consciousness = self.performance_history['consciousness_levels'][-1]
            
            if current_consciousness < self.thresholds['consciousness_level']['critical']:
                self._generate_alert('CRITICAL', 'Consciousness level critically low', current_consciousness)
            elif current_consciousness < self.thresholds['consciousness_level']['warning']:
                self._generate_alert('WARNING', 'Consciousness level below warning threshold', current_consciousness)
        
        # Check brain step time
        if self.performance_history['brain_step_times']:
            current_step_time = self.performance_history['brain_step_times'][-1]
            
            if current_step_time > self.thresholds['brain_step_time']['critical']:
                self._generate_alert('CRITICAL', 'Brain step time critically high', current_step_time)
            elif current_step_time > self.thresholds['brain_step_time']['warning']:
                self._generate_alert('WARNING', 'Brain step time above warning threshold', current_step_time)
        
        # Check integration latency
        if self.performance_history['integration_latency']:
            current_latency = self.performance_history['integration_latency'][-1]
            
            if current_latency > self.thresholds['integration_latency']['critical']:
                self._generate_alert('CRITICAL', 'Integration latency critically high', current_latency)
            elif current_latency > self.thresholds['integration_latency']['warning']:
                self._generate_alert('WARNING', 'Integration latency above warning threshold', current_latency)
    
    def _generate_alert(self, level: str, message: str, value: float):
        """Generate performance alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'value': value,
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        # Print alert
        alert_icon = "🚨" if level == "CRITICAL" else "⚠️"
        print(f"{alert_icon} {level}: {message} (Value: {value:.3f})")
        
        # Limit alert history
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dashboard_state': self.dashboard_state.copy(),
            'current_metrics': {},
            'performance_trends': {},
            'alerts': {
                'active': len([a for a in self.alerts if not a['acknowledged']]),
                'total': len(self.alert_history),
                'critical': len([a for a in self.alert_history if a['level'] == 'CRITICAL']),
                'warning': len([a for a in self.alert_history if a['level'] == 'WARNING'])
            }
        }
        
        # Current metrics
        if self.performance_history['consciousness_levels']:
            summary['current_metrics']['consciousness_level'] = self.performance_history['consciousness_levels'][-1]
        
        if self.performance_history['neural_activity']:
            summary['current_metrics']['neural_activity'] = self.performance_history['neural_activity'][-1]
        
        if self.performance_history['pfc_firing_rates']:
            summary['current_metrics']['pfc_firing_rate'] = self.performance_history['pfc_firing_rates'][-1]
        
        if self.performance_history['loop_stability']:
            summary['current_metrics']['loop_stability'] = self.performance_history['loop_stability'][-1]
        
        # Performance trends (last 10 values)
        for metric in ['consciousness_levels', 'neural_activity', 'brain_step_times']:
            if self.performance_history[metric]:
                recent_values = self.performance_history[metric][-10:]
                if len(recent_values) >= 2:
                    trend = 'increasing' if recent_values[-1] > recent_values[0] else 'decreasing'
                    summary['performance_trends'][metric] = trend
        
        return summary
    
    def acknowledge_alert(self, alert_index: int):
        """Acknowledge an alert"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index]['acknowledged'] = True
            print(f"✅ Alert acknowledged: {self.alerts[alert_index]['message']}")
    
    def clear_acknowledged_alerts(self):
        """Clear acknowledged alerts"""
        self.alerts = [a for a in self.alerts if not a['acknowledged']]
        print(f"🧹 Cleared acknowledged alerts. {len(self.alerts)} active alerts remaining.")
    
    def run_interactive_dashboard(self):
        """Run interactive dashboard mode"""
        print("📊 Interactive Performance Dashboard")
        print("Commands: status, summary, alerts, clear, quit")
        
        while True:
            try:
                command = input("\nDashboard command: ").lower().strip()
                
                if command == 'quit':
                    break
                elif command == 'status':
                    self._print_dashboard_status()
                elif command == 'summary':
                    summary = self.get_performance_summary()
                    print("\n📊 Performance Summary:")
                    print(json.dumps(summary, indent=2))
                elif command == 'alerts':
                    self._print_alerts()
                elif command == 'clear':
                    self.clear_acknowledged_alerts()
                else:
                    print("Unknown command. Use: status, summary, alerts, clear, quit")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Command error: {e}")
    
    def _print_dashboard_status(self):
        """Print current dashboard status"""
        print(f"\n📊 Dashboard Status:")
        print(f"  Monitoring Active: {'✅ Yes' if self.monitoring_active else '❌ No'}")
        print(f"  System Health: {self.dashboard_state['system_health'].upper()}")
        print(f"  Performance Score: {self.dashboard_state['performance_score']:.3f}")
        print(f"  Total Updates: {self.dashboard_state['total_updates']}")
        print(f"  Last Update: {self.dashboard_state['last_update'] or 'Never'}")
        
        # Current metrics
        if self.performance_history['consciousness_levels']:
            print(f"  Current Consciousness: {self.performance_history['consciousness_levels'][-1]:.3f}")
        
        if self.performance_history['pfc_firing_rates']:
            print(f"  Current PFC Firing: {self.performance_history['pfc_firing_rates'][-1]:.1f} Hz")
        
        if self.performance_history['loop_stability']:
            print(f"  Current Loop Stability: {self.performance_history['loop_stability'][-1]:.3f}")
    
    def _print_alerts(self):
        """Print current alerts"""
        if not self.alerts:
            print("✅ No active alerts")
            return
        
        print(f"\n🚨 Active Alerts ({len(self.alerts)}):")
        for i, alert in enumerate(self.alerts):
            if not alert['acknowledged']:
                level_icon = "🚨" if alert['level'] == "CRITICAL" else "⚠️"
                print(f"  {i}: {level_icon} {alert['level']} - {alert['message']}")
                print(f"      Value: {alert['value']:.3f}, Time: {alert['timestamp']}")
        
        print("\nTo acknowledge an alert, use: acknowledge <index>")
        print("To clear all acknowledged alerts, use: clear")

def main():
    """Main function for performance dashboard"""
    print("📊 Performance Monitoring Dashboard")
    print("=" * 50)
    
    # Create dashboard
    dashboard = PerformanceDashboard()
    
    try:
        print("Dashboard initialized. Use commands to monitor performance:")
        print("  start - Start monitoring (requires integrated consciousness)")
        print("  status - Show current status")
        print("  summary - Show performance summary")
        print("  alerts - Show active alerts")
        print("  quit - Exit dashboard")
        
        while True:
            command = input("\nDashboard> ").lower().strip()
            
            if command == 'quit':
                break
            elif command == 'start':
                print("To start monitoring, call:")
                print("  dashboard.start_monitoring(integrated_consciousness, brain_connector)")
            elif command == 'status':
                dashboard._print_dashboard_status()
            elif command == 'summary':
                summary = dashboard.get_performance_summary()
                print("\n📊 Performance Summary:")
                print(json.dumps(summary, indent=2))
            elif command == 'alerts':
                dashboard._print_alerts()
            else:
                print("Unknown command. Use: start, status, summary, alerts, quit")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        dashboard.stop_monitoring()
        print("Performance dashboard completed!")

if __name__ == "__main__":
    main()
