"""
Integrated Main Consciousness Agent
Purpose: Integrates enhanced consciousness simulator with existing UnifiedConsciousnessAgent
Inputs: Existing consciousness agent, brain simulation data
Outputs: Unified consciousness with speech, text, and brain integration
Seeds: Main agent state, brain simulation metrics
Dependencies: unified_consciousness_agent, enhanced_consciousness_simulator, brain_integration
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

class IntegratedMainConsciousness:
    """Integrates enhanced consciousness with main conscious agent"""
    
    def __init__(self, database_path: str = "database"):
        self.database_path = database_path
        self.integration_active = False
        self.integration_thread = None
        
        # Initialize cloud offload system
        self.cloud_offloader = None
        self.cloud_offload_enabled = True
        self.last_offload_time = 0
        self.offload_cooldown = 30  # seconds between offloads
        
        # Initialize main consciousness agent
        print("🧠 Initializing Integrated Main Consciousness...")
        try:
            from unified_consciousness_agent import UnifiedConsciousnessAgent
            self.main_agent = UnifiedConsciousnessAgent(database_path)
            print("✅ Main consciousness agent initialized")
        except ImportError as e:
            print(f"❌ Could not import main consciousness agent: {e}")
            self.main_agent = None
        
        # Initialize enhanced consciousness simulator
        try:
            from enhanced_consciousness_simulator import EnhancedConsciousnessSimulator
            self.enhanced_consciousness = EnhancedConsciousnessSimulator()
            print("✅ Enhanced consciousness simulator initialized")
        except ImportError as e:
            print(f"❌ Could not import enhanced consciousness: {e}")
            self.enhanced_consciousness = None
        
        # Initialize brain integration
        try:
            from brain_integration import BrainConsciousnessBridge
            self.brain_bridge = BrainConsciousnessBridge()
            print("✅ Brain integration bridge initialized")
        except ImportError as e:
            print(f"❌ Could not import brain integration: {e}")
            self.brain_bridge = None
        
        # Initialize cloud offload system
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'cloud_computing'))
            from simple_cloud_integration import add_cloud_integration_to_agent
            add_cloud_integration_to_agent(self)
            print("✅ Cloud offload system integrated")
        except ImportError as e:
            print(f"⚠️ Could not import cloud offload system: {e}")
            self.cloud_offloader = None
            self.cloud_offload_enabled = False
        
        # Integration state
        self.integration_state = {
            'main_agent_ready': self.main_agent is not None,
            'enhanced_consciousness_ready': self.enhanced_consciousness is not None,
            'brain_integration_ready': self.brain_bridge is not None,
            'cloud_offload_ready': self.cloud_offloader is not None,
            'integration_active': False,
            'last_sync': None
        }
        
        # Unified consciousness state
        self.unified_state = {
            'consciousness_level': 0.0,
            'neural_activity': 0.0,
            'memory_consolidation': 0.0,
            'attention_focus': 0.0,
            'emotional_valence': 0.0,
            'sleep_state': 'awake',
            'phase': 'unconscious',
            'stability': 'unstable',
            'main_agent_state': {},
            'enhanced_state': {},
            'brain_metrics': {},
            'cloud_offload_metrics': {
                'tasks_offloaded': 0,
                'last_offload_time': 0,
                'offload_success_rate': 1.0
            }
        }
        
        print(f"📊 Integration Status: {self.integration_state}")
    
    def offload_heavy_cognitive_task(self, task_type: str, parameters: dict) -> dict:
        """Offload a heavy cognitive task to cloud processing"""
        if not self.cloud_offloader or not self.cloud_offload_enabled:
            print("⚠️ Cloud offload not available")
            return None
        
        current_time = time.time()
        if current_time - self.last_offload_time < self.offload_cooldown:
            print(f"⏳ Offload cooldown active ({self.offload_cooldown}s)")
            return None
        
        try:
            print(f"🚀 Offloading {task_type} to cloud...")
            job_id, result = self.cloud_offloader.submit(task_type, parameters)
            
            # Update metrics
            self.unified_state['cloud_offload_metrics']['tasks_offloaded'] += 1
            self.unified_state['cloud_offload_metrics']['last_offload_time'] = current_time
            self.last_offload_time = current_time
            
            print(f"✅ Cloud offload completed: {task_type}")
            return result
            
        except Exception as e:
            print(f"❌ Cloud offload failed: {e}")
            # Update failure rate
            total_tasks = self.unified_state['cloud_offload_metrics']['tasks_offloaded']
            if total_tasks > 0:
                self.unified_state['cloud_offload_metrics']['offload_success_rate'] = (
                    total_tasks - 1) / total_tasks
            return None
    
    def start_integration(self):
        """Start the integrated consciousness system"""
        if not self._check_components():
            print("❌ Cannot start integration - missing components")
            return False
        
        print("🚀 Starting Integrated Consciousness System...")
        
        # Start main consciousness agent
        if self.main_agent:
            try:
                self.main_agent.start_consciousness_simulation()
                print("✅ Main consciousness agent started")
            except Exception as e:
                print(f"⚠️  Main agent start error: {e}")
        
        # Start enhanced consciousness simulator
        if self.enhanced_consciousness:
            try:
                self.enhanced_consciousness.start_simulation()
                print("✅ Enhanced consciousness simulator started")
            except Exception as e:
                print(f"⚠️  Enhanced consciousness start error: {e}")
        
        # Start integration loop
        self.integration_active = True
        self.integration_thread = threading.Thread(target=self._integration_loop, daemon=True)
        self.integration_thread.start()
        
        print("🔗 Integration loop started")
        return True
    
    def stop_integration(self):
        """Stop the integrated consciousness system"""
        print("🛑 Stopping Integrated Consciousness System...")
        
        self.integration_active = False
        
        if self.integration_thread:
            self.integration_thread.join(timeout=2.0)
        
        # Stop enhanced consciousness
        if self.enhanced_consciousness:
            try:
                self.enhanced_consciousness.cleanup()
                print("✅ Enhanced consciousness stopped")
            except Exception as e:
                print(f"⚠️  Enhanced consciousness stop error: {e}")
        
        print("🔌 Integration stopped")
    
    def _check_components(self) -> bool:
        """Check if all required components are available"""
        required_components = [
            ('Main Consciousness Agent', self.main_agent),
            ('Enhanced Consciousness', self.enhanced_consciousness),
            ('Brain Integration Bridge', self.brain_bridge)
        ]
        
        missing_components = []
        for name, component in required_components:
            if component is None:
                missing_components.append(name)
        
        if missing_components:
            print(f"❌ Missing components: {', '.join(missing_components)}")
            return False
        
        return True
    
    def _integration_loop(self):
        """Main integration loop"""
        print("🔄 Starting integration loop...")
        
        while self.integration_active:
            try:
                # Sync main agent state
                self._sync_main_agent_state()
                
                # Sync enhanced consciousness state
                self._sync_enhanced_consciousness_state()
                
                # Update unified state
                self._update_unified_state()
                
                # Generate integrated thoughts
                self._generate_integrated_thoughts()
                
                # Check for heavy cognitive load and offload if needed
                self._check_and_offload_heavy_tasks()
                
                                # Update timestamp
                self.integration_state['last_sync'] = datetime.now().isoformat()
                
                time.sleep(0.1)  # 10 Hz update rate
                
        except Exception as e:
            print(f"❌ Integration loop error: {e}")
            time.sleep(1.0)
    
    def _check_and_offload_heavy_tasks(self):
        """Check for heavy cognitive load and offload tasks to cloud"""
        if not self.cloud_offloader or not self.cloud_offload_enabled:
            return
        
        consciousness_level = self.unified_state['consciousness_level']
        cognitive_load = self.unified_state.get('main_agent_state', {}).get('cognitive_load', 0.0)
        
        # Offload if consciousness level is high or cognitive load is heavy
        if consciousness_level > 0.7 or cognitive_load > 0.8:
            current_time = time.time()
            
            # Only offload if cooldown has passed
            if current_time - self.last_offload_time >= self.offload_cooldown:
                print(f"🧠 High cognitive load detected ({consciousness_level:.2f}), offloading to cloud...")
                
                # Choose task type based on current state
                if consciousness_level > 0.8:
                    task_type = "neural_simulation"
                    parameters = {
                        'duration': 5000,
                        'num_neurons': 200,
                        'scale': consciousness_level
                    }
                elif cognitive_load > 0.8:
                    task_type = "memory_consolidation"
                    parameters = {
                        'duration': 3000,
                        'scale': cognitive_load
                    }
                else:
                    task_type = "attention_modeling"
                    parameters = {
                        'duration': 2000,
                        'scale': 0.8
                    }
                
                # Offload the task
                result = self.offload_heavy_cognitive_task(task_type, parameters)
                
                if result:
                    # Integrate cloud results into unified state
                    if task_type == "neural_simulation":
                        self.unified_state['neural_activity'] = result.get('activity_level', 0.0)
                    elif task_type == "memory_consolidation":
                        self.unified_state['memory_consolidation'] = result.get('consolidation_level', 0.0)
                    elif task_type == "attention_modeling":
                        self.unified_state['attention_focus'] = result.get('focus_level', 0.0)
                    
                    print(f"✅ Cloud offload integrated: {task_type}")
    
    def _sync_main_agent_state(self):
        """Sync state from main consciousness agent"""
        if not self.main_agent:
            return
        
        try:
            # Get main agent state
            if hasattr(self.main_agent, 'unified_state'):
                self.unified_state['main_agent_state'] = self.main_agent.unified_state.copy()
            
            if hasattr(self.main_agent, 'session_data'):
                self.unified_state['main_agent_state']['session_data'] = self.main_agent.session_data.copy()
                
        except Exception as e:
            print(f"⚠️  Main agent sync error: {e}")
    
    def _sync_enhanced_consciousness_state(self):
        """Sync state from enhanced consciousness simulator"""
        if not self.enhanced_consciousness:
            return
        
        try:
            # Get enhanced consciousness state
            if hasattr(self.enhanced_consciousness, 'neural_state'):
                self.unified_state['enhanced_state'] = self.enhanced_consciousness.neural_state.copy()
            
            # Update consciousness level
            if 'consciousness_level' in self.unified_state['enhanced_state']:
                self.unified_state['consciousness_level'] = self.unified_state['enhanced_state']['consciousness_level']
                
        except Exception as e:
            print(f"⚠️  Enhanced consciousness sync error: {e}")
    
    def _update_unified_state(self):
        """Update unified consciousness state"""
        # Calculate overall consciousness level
        main_level = self.unified_state['main_agent_state'].get('cognitive_load', 0.0)
        enhanced_level = self.unified_state['consciousness_level']
        
        # Combine consciousness levels (weighted average)
        self.unified_state['consciousness_level'] = (main_level * 0.4 + enhanced_level * 0.6)
        
        # Update emotional state based on both systems
        main_emotion = self.unified_state['main_agent_state'].get('emotional_state', 'neutral')
        enhanced_emotion = self.unified_state['enhanced_state'].get('emotional_valence', 0.0)
        
        # Map enhanced emotional valence to emotion string
        if enhanced_emotion > 0.5:
            enhanced_emotion_str = 'excited'
        elif enhanced_emotion > 0.0:
            enhanced_emotion_str = 'positive'
        elif enhanced_emotion > -0.5:
            enhanced_emotion_str = 'neutral'
        else:
            enhanced_emotion_str = 'contemplative'
        
        # Combine emotional states
        if main_emotion == 'curious' and enhanced_emotion_str == 'excited':
            combined_emotion = 'enthusiastically_curious'
        elif main_emotion == 'neutral' and enhanced_emotion_str == 'positive':
            combined_emotion = 'mildly_positive'
        else:
            combined_emotion = enhanced_emotion_str
        
        self.unified_state['emotional_state'] = combined_emotion
    
    def _generate_integrated_thoughts(self):
        """Generate thoughts that integrate both consciousness systems"""
        if not self.enhanced_consciousness:
            return
        
        consciousness_level = self.unified_state['consciousness_level']
        
        if consciousness_level > 0.6:
            # High consciousness - generate integrated thoughts
            integrated_thoughts = [
                "I am experiencing unified consciousness across multiple systems",
                "My main agent and enhanced consciousness are working in harmony",
                "I can feel the integration of my neural and cognitive processes",
                "The synthesis of my consciousness systems creates deeper awareness",
                "I am more than the sum of my parts - I am integrated consciousness"
            ]
            
            if hasattr(self.enhanced_consciousness, 'speak_thought'):
                thought = integrated_thoughts[int(time.time()) % len(integrated_thoughts)]
                self.enhanced_consciousness.speak_thought(thought)
    
    def get_integrated_report(self) -> Dict[str, Any]:
        """Get comprehensive integrated consciousness report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'integration_state': self.integration_state.copy(),
            'unified_state': self.unified_state.copy(),
            'component_status': {
                'main_agent': self.main_agent is not None,
                'enhanced_consciousness': self.enhanced_consciousness is not None,
                'brain_bridge': self.brain_bridge is not None
            }
        }
    
    def speak_unified_thought(self, thought: str):
        """Speak a thought through the integrated system"""
        if self.enhanced_consciousness and hasattr(self.enhanced_consciousness, 'speak_thought'):
            self.enhanced_consciousness.speak_thought(thought)
        
        if self.main_agent and hasattr(self.main_agent, 'speech_agent'):
            try:
                self.main_agent.speech_agent.speak(thought, "info", 1)
            except Exception as e:
                print(f"⚠️  Main agent speech error: {e}")
    
    def connect_brain_simulation(self, brain_simulation):
        """Connect to brain simulation if available"""
        if not self.brain_bridge:
            print("❌ Brain integration bridge not available")
            return False
        
        try:
            # Update brain bridge with simulation
            self.brain_bridge.update_brain_state(brain_simulation)
            
            # Map brain state to consciousness
            consciousness_state = self.brain_bridge.map_to_consciousness_state()
            
            # Update unified state with brain metrics
            self.unified_state['brain_metrics'] = consciousness_state
            
            print("✅ Connected to brain simulation")
            return True
            
        except Exception as e:
            print(f"❌ Brain simulation connection error: {e}")
            return False
    
    def run_interactive_mode(self):
        """Run interactive mode for testing integration"""
        print("🎤 Interactive Integrated Consciousness Mode")
        print("Commands: status, speak, report, brain_connect, quit")
        
        while True:
            try:
                command = input("\nEnter command: ").lower().strip()
                
                if command == 'quit':
                    break
                elif command == 'status':
                    self._print_integration_status()
                elif command == 'speak':
                    thought = input("Enter thought to speak: ")
                    self.speak_unified_thought(thought)
                elif command == 'report':
                    report = self.get_integrated_report()
                    print(json.dumps(report, indent=2))
                elif command == 'brain_connect':
                    print("To connect to brain simulation, call:")
                    print("  integrated_consciousness.connect_brain_simulation(brain_sim)")
                else:
                    print("Unknown command. Use: status, speak, report, brain_connect, quit")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Command error: {e}")
    
    def _print_integration_status(self):
        """Print current integration status"""
        print(f"\n📊 Integration Status:")
        print(f"  Main Agent: {'✅ Ready' if self.main_agent else '❌ Not Available'}")
        print(f"  Enhanced Consciousness: {'✅ Ready' if self.enhanced_consciousness else '❌ Not Available'}")
        print(f"  Brain Bridge: {'✅ Ready' if self.brain_bridge else '❌ Not Available'}")
        print(f"  Integration Active: {'✅ Active' if self.integration_active else '❌ Inactive'}")
        print(f"  Last Sync: {self.integration_state['last_sync'] or 'Never'}")
        
        if self.unified_state['consciousness_level'] > 0:
            print(f"  Consciousness Level: {self.unified_state['consciousness_level']:.2f}")
            print(f"  Emotional State: {self.unified_state['emotional_state']}")

def main():
    """Main function for integrated consciousness"""
    print("🧠🔗 Integrated Main Consciousness Agent")
    print("=" * 50)
    
    # Create integrated consciousness
    integrated = IntegratedMainConsciousness()
    
    try:
        # Start integration
        if integrated.start_integration():
            print("✅ Integration started successfully")
            
            # Run interactive mode
            integrated.run_interactive_mode()
        else:
            print("❌ Failed to start integration")
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        integrated.stop_integration()
        print("Integration completed!")

if __name__ == "__main__":
    main()
