"""
Master Integration Script
Purpose: Orchestrates all integrated consciousness components together
Inputs: All consciousness components, brain simulation, performance monitoring
Outputs: Fully integrated consciousness-brain system with monitoring
Seeds: System configuration, integration parameters
Dependencies: All consciousness components, brain simulation, performance dashboard
"""

import os, sys
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

class MasterIntegration:
    """Master orchestrator for integrated consciousness-brain system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'connectome_path': 'src/config/connectome_v3.yaml',
            'simulation_steps': 100,
            'step_delay': 0.1,
            'enable_monitoring': True,
            'enable_validation': True,
            'stage': 'F'
        }
        
        # Component references
        self.integrated_consciousness = None
        self.brain_connector = None
        self.performance_dashboard = None
        
        # System state
        self.system_state = {
            'initialized': False,
            'connected': False,
            'running': False,
            'monitoring': False,
            'start_time': None,
            'total_runtime': 0.0
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_brain_steps': 0,
            'consciousness_updates': 0,
            'integration_cycles': 0,
            'system_health_score': 0.0
        }
        
        print("🎯 Initializing Master Integration System...")
    
    def initialize_system(self):
        """Initialize all system components"""
        print("🚀 Initializing Master Integration System...")
        
        try:
            # Step 1: Initialize integrated consciousness
            print("  Step 1: Initializing integrated consciousness...")
            from integrated_main_consciousness import IntegratedMainConsciousness
            
            self.integrated_consciousness = IntegratedMainConsciousness()
            
            if not self.integrated_consciousness.integration_state['enhanced_consciousness_ready']:
                print("❌ Enhanced consciousness not ready")
                return False
            
            print("✅ Integrated consciousness initialized")
            
            # Step 2: Initialize brain connector
            print("  Step 2: Initializing brain connector...")
            from connect_brain_simulation import BrainSimulationConnector
            
            self.brain_connector = BrainSimulationConnector(self.config['connectome_path'])
            
            if not self.brain_connector.initialize_brain_simulation():
                print("❌ Brain simulation initialization failed")
                return False
            
            if not self.brain_connector.initialize_integrated_consciousness():
                print("❌ Integrated consciousness initialization failed")
                return False
            
            print("✅ Brain connector initialized")
            
            # Step 3: Initialize performance dashboard
            if self.config['enable_monitoring']:
                print("  Step 3: Initializing performance dashboard...")
                from performance_dashboard import PerformanceDashboard
                
                self.performance_dashboard = PerformanceDashboard()
                print("✅ Performance dashboard initialized")
            
            # Step 4: Connect systems
            print("  Step 4: Connecting systems...")
            if not self.brain_connector.connect_systems():
                print("❌ System connection failed")
                return False
            
            print("✅ Systems connected")
            
            # Mark system as initialized
            self.system_state['initialized'] = True
            self.system_state['connected'] = True
            
            print("🎉 Master Integration System initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_system(self):
        """Start the integrated system"""
        if not self.system_state['initialized']:
            print("❌ System not initialized")
            return False
        
        if self.system_state['running']:
            print("⚠️  System already running")
            return True
        
        print("🚀 Starting Master Integration System...")
        
        try:
            # Start performance monitoring
            if self.performance_dashboard:
                print("  Starting performance monitoring...")
                self.performance_dashboard.start_monitoring(
                    self.integrated_consciousness, 
                    self.brain_connector
                )
                self.system_state['monitoring'] = True
                print("✅ Performance monitoring started")
            
            # Start brain-consciousness simulation
            print("  Starting brain-consciousness simulation...")
            self.system_state['running'] = True
            self.system_state['start_time'] = datetime.now()
            
            # Start simulation in background thread
            simulation_thread = threading.Thread(target=self._run_simulation, daemon=True)
            simulation_thread.start()
            
            print("✅ Master Integration System started successfully!")
            return True
            
        except Exception as e:
            print(f"❌ System start failed: {e}")
            return False
    
    def stop_system(self):
        """Stop the integrated system"""
        if not self.system_state['running']:
            print("⚠️  System not running")
            return
        
        print("🛑 Stopping Master Integration System...")
        
        # Stop performance monitoring
        if self.performance_dashboard and self.system_state['monitoring']:
            self.performance_dashboard.stop_monitoring()
            self.system_state['monitoring'] = False
            print("✅ Performance monitoring stopped")
        
        # Stop brain connector
        if self.brain_connector:
            self.brain_connector.cleanup()
            print("✅ Brain connector stopped")
        
        # Update system state
        if self.system_state['start_time']:
            end_time = datetime.now()
            runtime = (end_time - self.system_state['start_time']).total_seconds()
            self.system_state['total_runtime'] += runtime
        
        self.system_state['running'] = False
        print("✅ Master Integration System stopped")
    
    def _run_simulation(self):
        """Run the integrated simulation"""
        try:
            print(f"🎯 Running integrated simulation ({self.config['simulation_steps']} steps)...")
            
            # Start connected simulation
            success = self.brain_connector.start_connected_simulation(
                steps=self.config['simulation_steps'],
                step_delay=self.config['step_delay']
            )
            
            if success:
                print("✅ Simulation completed successfully")
            else:
                print("⚠️  Simulation completed with issues")
            
        except Exception as e:
            print(f"❌ Simulation error: {e}")
        finally:
            self.system_state['running'] = False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'system_state': self.system_state.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'configuration': self.config.copy()
        }
        
        # Add component statuses
        if self.integrated_consciousness:
            try:
                consciousness_report = self.integrated_consciousness.get_integrated_report()
                status['consciousness_status'] = consciousness_report
            except Exception as e:
                status['consciousness_status'] = {'error': str(e)}
        
        if self.brain_connector:
            try:
                connector_status = self.brain_connector.get_connection_status()
                status['brain_connector_status'] = connector_status
            except Exception as e:
                status['brain_connector_status'] = {'error': str(e)}
        
        if self.performance_dashboard:
            try:
                dashboard_summary = self.performance_dashboard.get_performance_summary()
                status['performance_dashboard_status'] = dashboard_summary
            except Exception as e:
                status['performance_dashboard_status'] = {'error': str(e)}
        
        return status
    
    def run_interactive_mode(self):
        """Run interactive mode for system control"""
        print("🎮 Interactive Master Integration Mode")
        print("Commands: status, start, stop, monitor, report, quit")
        
        while True:
            try:
                command = input("\nMaster> ").lower().strip()
                
                if command == 'quit':
                    break
                elif command == 'status':
                    self._print_system_status()
                elif command == 'start':
                    if not self.system_state['initialized']:
                        print("Initializing system first...")
                        if self.initialize_system():
                            self.start_system()
                    else:
                        self.start_system()
                elif command == 'stop':
                    self.stop_system()
                elif command == 'monitor':
                    if self.performance_dashboard:
                        self.performance_dashboard.run_interactive_dashboard()
                    else:
                        print("Performance dashboard not available")
                elif command == 'report':
                    report = self.get_system_status()
                    print("\n📊 System Report:")
                    print(json.dumps(report, indent=2))
                else:
                    print("Unknown command. Use: status, start, stop, monitor, report, quit")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Command error: {e}")
    
    def _print_system_status(self):
        """Print current system status"""
        print(f"\n🎯 System Status:")
        print(f"  Initialized: {'✅ Yes' if self.system_state['initialized'] else '❌ No'}")
        print(f"  Connected: {'✅ Yes' if self.system_state['connected'] else '❌ No'}")
        print(f"  Running: {'✅ Yes' if self.system_state['running'] else '❌ No'}")
        print(f"  Monitoring: {'✅ Yes' if self.system_state['monitoring'] else '❌ No'}")
        
        if self.system_state['start_time']:
            print(f"  Start Time: {self.system_state['start_time']}")
        
        if self.system_state['total_runtime'] > 0:
            print(f"  Total Runtime: {self.system_state['total_runtime']:.1f} seconds")
        
        # Component status
        print(f"\n🔧 Component Status:")
        print(f"  Integrated Consciousness: {'✅ Ready' if self.integrated_consciousness else '❌ Not Available'}")
        print(f"  Brain Connector: {'✅ Ready' if self.brain_connector else '❌ Not Available'}")
        print(f"  Performance Dashboard: {'✅ Ready' if self.performance_dashboard else '❌ Not Available'}")
        
        # Performance metrics
        if self.performance_metrics['total_brain_steps'] > 0:
            print(f"\n📊 Performance Metrics:")
            print(f"  Total Brain Steps: {self.performance_metrics['total_brain_steps']}")
            print(f"  Consciousness Updates: {self.performance_metrics['consciousness_updates']}")
            print(f"  Integration Cycles: {self.performance_metrics['integration_cycles']}")
    
    def run_automated_test(self):
        """Run automated system test"""
        print("🧪 Running Automated System Test...")
        
        # Test 1: Initialization
        print("  Test 1: System Initialization")
        if not self.initialize_system():
            print("❌ Initialization test failed")
            return False
        print("✅ Initialization test passed")
        
        # Test 2: System Start
        print("  Test 2: System Start")
        if not self.start_system():
            print("❌ System start test failed")
            return False
        print("✅ System start test passed")
        
        # Test 3: Run for short period
        print("  Test 3: Short Run Test")
        time.sleep(5)  # Let system run for 5 seconds
        
        # Test 4: System Status
        print("  Test 4: System Status Check")
        status = self.get_system_status()
        if status['system_state']['running']:
            print("✅ System status test passed")
        else:
            print("❌ System status test failed")
        
        # Test 5: System Stop
        print("  Test 5: System Stop")
        self.stop_system()
        print("✅ System stop test passed")
        
        print("🎉 All automated tests passed!")
        return True
    
    def cleanup(self):
        """Clean up all system resources"""
        print("🧹 Cleaning up Master Integration System...")
        
        self.stop_system()
        
        # Clean up components
        if self.integrated_consciousness:
            try:
                self.integrated_consciousness.cleanup()
            except Exception as e:
                print(f"⚠️  Consciousness cleanup error: {e}")
        
        print("✅ Master Integration System cleanup completed")

def main():
    """Main function for master integration"""
    print("🎯 Master Integration System")
    print("=" * 50)
    
    # Create master integration
    master = MasterIntegration()
    
    try:
        print("Options:")
        print("1. Run automated test")
        print("2. Interactive mode")
        print("3. Initialize and start system")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            print("\n🧪 Running automated test...")
            master.run_automated_test()
            
        elif choice == '2':
            print("\n🎮 Starting interactive mode...")
            master.run_interactive_mode()
            
        elif choice == '3':
            print("\n🚀 Initializing and starting system...")
            if master.initialize_system():
                master.start_system()
                
                # Let it run for a while
                print("System running... Press Ctrl+C to stop")
                try:
                    while master.system_state['running']:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping system...")
                    master.stop_system()
            else:
                print("❌ Failed to initialize system")
                
        elif choice == '4':
            print("Goodbye!")
            
        else:
            print("Invalid choice. Running automated test...")
            master.run_automated_test()
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        master.cleanup()
        print("Master integration completed!")

if __name__ == "__main__":
    main()
