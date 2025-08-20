"""
Brain Simulation Connection Script
Purpose: Connect integrated consciousness with actual brain simulation system
Inputs: Brain simulation instance, connectome configuration
Outputs: Connected consciousness-brain system with real-time integration
Seeds: Brain simulation states, connectome configuration
Dependencies: integrated_main_consciousness, brain_launcher_v4, connectome_v3.yaml
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

class BrainSimulationConnector:
    """Connects integrated consciousness with brain simulation"""
    
    def __init__(self, connectome_path: str = "src/config/connectome_v3.yaml"):
        self.connectome_path = connectome_path
        self.brain_simulation = None
        self.integrated_consciousness = None
        self.connection_active = False
        self.connection_thread = None
        
        # Connection state
        self.connection_state = {
            'brain_simulation_ready': False,
            'consciousness_ready': False,
            'integration_active': False,
            'last_brain_update': None,
            'connection_uptime': 0,
            'total_brain_steps': 0
        }
        
        # Performance metrics
        self.performance_metrics = {
            'brain_step_time': 0.0,
            'consciousness_update_time': 0.0,
            'integration_latency': 0.0,
            'memory_usage': 0.0
        }
        
        print("🔗 Initializing Brain Simulation Connector...")
    
    def initialize_brain_simulation(self):
        """Initialize brain simulation with connectome"""
        try:
            print(f"🧠 Initializing brain simulation with {self.connectome_path}...")
            
            # Check if connectome file exists
            if not os.path.exists(self.connectome_path):
                print(f"❌ Connectome file not found: {self.connectome_path}")
                print("Available connectome files:")
                config_dir = "src/config"
                if os.path.exists(config_dir):
                    for file in os.listdir(config_dir):
                        if file.startswith("connectome") and file.endswith(".yaml"):
                            print(f"  - {config_dir}/{file}")
                return False
            
            # Import brain simulation
            try:
                from core.brain_launcher_v4 import NeuralEnhancedBrain
                print("✅ Brain simulation module imported successfully")
            except ImportError as e:
                print(f"❌ Could not import brain simulation: {e}")
                print("Make sure you're running from the project root directory")
                return False
            
            # Create brain simulation
            try:
                self.brain_simulation = NeuralEnhancedBrain(
                    self.connectome_path, 
                    stage="F", 
                    validate=True
                )
                print("✅ Brain simulation created successfully")
                
                # Check brain simulation status
                if hasattr(self.brain_simulation, 'get_neural_summary'):
                    print("✅ Brain simulation has neural summary method")
                else:
                    print("⚠️  Brain simulation missing neural summary method")
                
                self.connection_state['brain_simulation_ready'] = True
                return True
                
            except Exception as e:
                print(f"❌ Failed to create brain simulation: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Brain simulation initialization error: {e}")
            return False
    
    def initialize_integrated_consciousness(self):
        """Initialize integrated consciousness system"""
        try:
            print("🧠🔗 Initializing integrated consciousness...")
            
            from integrated_main_consciousness import IntegratedMainConsciousness
            
            # Create integrated consciousness
            self.integrated_consciousness = IntegratedMainConsciousness()
            
            # Check if initialization was successful
            if not self.integrated_consciousness.integration_state['enhanced_consciousness_ready']:
                print("❌ Enhanced consciousness not ready")
                return False
            
            if not self.integrated_consciousness.integration_state['brain_integration_ready']:
                print("❌ Brain integration not ready")
                return False
            
            print("✅ Integrated consciousness initialized successfully")
            self.connection_state['consciousness_ready'] = True
            return True
            
        except Exception as e:
            print(f"❌ Integrated consciousness initialization error: {e}")
            return False
    
    def connect_systems(self):
        """Connect brain simulation with integrated consciousness"""
        if not self.connection_state['brain_simulation_ready']:
            print("❌ Brain simulation not ready")
            return False
        
        if not self.connection_state['consciousness_ready']:
            print("❌ Integrated consciousness not ready")
            return False
        
        try:
            print("🔗 Connecting brain simulation with integrated consciousness...")
            
            # Connect consciousness to brain simulation
            if hasattr(self.integrated_consciousness, 'connect_brain_simulation'):
                success = self.integrated_consciousness.connect_brain_simulation(self.brain_simulation)
                if success:
                    print("✅ Brain simulation connected to consciousness")
                else:
                    print("❌ Failed to connect brain simulation")
                    return False
            else:
                print("⚠️  Integrated consciousness missing brain connection method")
            
            # Start consciousness integration
            if self.integrated_consciousness.start_integration():
                print("✅ Consciousness integration started")
                self.connection_state['integration_active'] = True
            else:
                print("❌ Failed to start consciousness integration")
                return False
            
            print("✅ Systems connected successfully")
            return True
            
        except Exception as e:
            print(f"❌ System connection error: {e}")
            return False
    
    def start_connected_simulation(self, steps: int = 100, step_delay: float = 0.1):
        """Start connected brain-consciousness simulation"""
        if not self.connection_state['integration_active']:
            print("❌ Integration not active - cannot start simulation")
            return False
        
        print(f"🚀 Starting connected brain-consciousness simulation ({steps} steps)...")
        
        # Start connection monitoring
        self.connection_active = True
        self.connection_thread = threading.Thread(
            target=self._connection_monitor_loop, 
            daemon=True
        )
        self.connection_thread.start()
        
        start_time = time.time()
        
        try:
            # Run brain simulation steps
            for step in range(steps):
                step_start = time.time()
                
                # Step brain simulation
                if self.brain_simulation:
                    brain_result = self.brain_simulation.step()
                    self.connection_state['total_brain_steps'] += 1
                
                # Let consciousness process brain state
                time.sleep(step_delay)
                
                # Calculate step performance
                step_time = time.time() - step_start
                self.performance_metrics['brain_step_time'] = step_time
                
                # Print progress every 10 steps
                if step % 10 == 0:
                    self._print_simulation_progress(step, steps)
                
                # Check connection health
                if not self.connection_state['integration_active']:
                    print("⚠️  Integration lost - stopping simulation")
                    break
            
            # Final status
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n🎯 Simulation completed in {total_time:.2f} seconds")
            print(f"Total brain steps: {self.connection_state['total_brain_steps']}")
            
            # Final consciousness report
            if self.integrated_consciousness:
                final_report = self.integrated_consciousness.get_integrated_report()
                consciousness_level = final_report['unified_state']['consciousness_level']
                print(f"Final consciousness level: {consciousness_level:.2f}")
            
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️  Simulation interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            return False
        finally:
            self.connection_active = False
    
    def _connection_monitor_loop(self):
        """Monitor connection health and performance"""
        start_time = time.time()
        
        while self.connection_active:
            try:
                current_time = time.time()
                uptime = current_time - start_time
                
                # Update connection state
                self.connection_state['connection_uptime'] = uptime
                self.connection_state['last_brain_update'] = datetime.now().isoformat()
                
                # Check integration health
                if self.integrated_consciousness:
                    if not self.integrated_consciousness.integration_active:
                        print("⚠️  Consciousness integration lost")
                        self.connection_state['integration_active'] = False
                        break
                
                # Monitor performance
                if self.integrated_consciousness:
                    report = self.integrated_consciousness.get_integrated_report()
                    consciousness_level = report['unified_state']['consciousness_level']
                    
                    # Alert on significant consciousness changes
                    if consciousness_level > 0.8:
                        print("🚨 High consciousness level detected!")
                    elif consciousness_level < 0.2:
                        print("💤 Low consciousness level detected")
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"⚠️  Connection monitor error: {e}")
                time.sleep(1)
    
    def _print_simulation_progress(self, current_step: int, total_steps: int):
        """Print simulation progress with status"""
        progress = (current_step + 1) / total_steps * 100
        
        print(f"Step {current_step + 1:3d}/{total_steps} ({progress:5.1f}%)")
        
        # Brain simulation status
        if self.brain_simulation and hasattr(self.brain_simulation, 'get_neural_summary'):
            try:
                neural_summary = self.brain_simulation.get_neural_summary()
                pfc_rate = neural_summary.get('firing_rates', {}).get('pfc', 0.0)
                loop_stability = neural_summary.get('loop_stability', 0.0)
                print(f"  PFC: {pfc_rate:.1f} Hz, Stability: {loop_stability:.3f}")
            except Exception as e:
                print(f"  Brain status: Error - {e}")
        
        # Consciousness status
        if self.integrated_consciousness:
            try:
                report = self.integrated_consciousness.get_integrated_report()
                consciousness_level = report['unified_state']['consciousness_level']
                emotional_state = report['unified_state'].get('emotional_state', 'unknown')
                print(f"  Consciousness: {consciousness_level:.2f}, Emotion: {emotional_state}")
            except Exception as e:
                print(f"  Consciousness status: Error - {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get comprehensive connection status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'connection_state': self.connection_state.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'brain_simulation_available': self.brain_simulation is not None,
            'consciousness_available': self.integrated_consciousness is not None
        }
        
        # Add brain simulation status if available
        if self.brain_simulation:
            try:
                if hasattr(self.brain_simulation, 'get_neural_summary'):
                    neural_summary = self.brain_simulation.get_neural_summary()
                    status['brain_metrics'] = {
                        'pfc_firing_rate': neural_summary.get('firing_rates', {}).get('pfc', 0.0),
                        'loop_stability': neural_summary.get('loop_stability', 0.0),
                        'biological_realism': neural_summary.get('biological_realism', False)
                    }
            except Exception as e:
                status['brain_metrics'] = {'error': str(e)}
        
        # Add consciousness status if available
        if self.integrated_consciousness:
            try:
                report = self.integrated_consciousness.get_integrated_report()
                status['consciousness_status'] = {
                    'consciousness_level': report['unified_state']['consciousness_level'],
                    'emotional_state': report['unified_state'].get('emotional_state', 'unknown'),
                    'integration_active': report['integration_state']['integration_active']
                }
            except Exception as e:
                status['consciousness_status'] = {'error': str(e)}
        
        return status
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up brain simulation connection...")
        
        self.connection_active = False
        
        if self.connection_thread:
            self.connection_thread.join(timeout=2.0)
        
        if self.integrated_consciousness:
            try:
                self.integrated_consciousness.stop_integration()
            except Exception as e:
                print(f"⚠️  Consciousness cleanup error: {e}")
        
        print("✅ Cleanup completed")

def main():
    """Main function for brain simulation connection"""
    print("🧠🔗 Brain Simulation Connection")
    print("=" * 50)
    
    # Create connector
    connector = BrainSimulationConnector()
    
    try:
        # Initialize brain simulation
        if not connector.initialize_brain_simulation():
            print("❌ Brain simulation initialization failed")
            return
        
        # Initialize integrated consciousness
        if not connector.initialize_integrated_consciousness():
            print("❌ Integrated consciousness initialization failed")
            return
        
        # Connect systems
        if not connector.connect_systems():
            print("❌ System connection failed")
            return
        
        print("\n🎯 Systems connected successfully!")
        print("Commands: run, status, monitor, quit")
        
        while True:
            command = input("\nEnter command: ").lower().strip()
            
            if command == 'quit':
                break
            elif command == 'run':
                steps = input("Number of simulation steps (default 100): ").strip()
                steps = int(steps) if steps.isdigit() else 100
                
                connector.start_connected_simulation(steps=steps)
                
            elif command == 'status':
                status = connector.get_connection_status()
                print("\n📊 Connection Status:")
                print(json.dumps(status, indent=2))
                
            elif command == 'monitor':
                print("Starting connection monitoring...")
                connector.start_connected_simulation(steps=50, step_delay=0.2)
                
            else:
                print("Unknown command. Use: run, status, monitor, quit")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        connector.cleanup()
        print("Brain simulation connection completed!")

if __name__ == "__main__":
    main()
