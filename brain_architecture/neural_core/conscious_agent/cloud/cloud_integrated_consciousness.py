#!/usr/bin/env python3
"""
Cloud-Integrated Main Consciousness Agent
Purpose: Integrates enhanced consciousness simulator with cloud offload capabilities
Inputs: Existing consciousness agent, brain simulation data
Outputs: Unified consciousness with speech, text, brain integration, and cloud offload
Seeds: Main agent state, brain simulation metrics
Dependencies: unified_consciousness_agent, enhanced_consciousness_simulator, brain_integration, cloud_offload
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'cloud_computing'))

class CloudIntegratedConsciousness:
    """Integrates enhanced consciousness with main conscious agent and cloud offload"""
    
    def __init__(self, database_path: str = "database"):
        self.database_path = database_path
        self.integration_active = False
        self.integration_thread = None
        
        # Initialize main consciousness agent
        print("🧠 Initializing Cloud-Integrated Main Consciousness...")
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
            'cloud_offload_ready': hasattr(self, 'cloud_integration'),
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
    
    def start_integration(self):
        """Start the integrated consciousness system"""
        if not self._check_components():
            print("❌ Cannot start integration - missing components")
            return False
        
        print("🚀 Starting Cloud-Integrated Consciousness System...")
        
        # Start main consciousness agent
        if self.main_agent:
            try:
                # Check if the method exists, if not, skip it
                if hasattr(self.main_agent, 'start_consciousness_simulation'):
                    self.main_agent.start_consciousness_simulation()
                    print("✅ Main consciousness agent started")
                else:
                    print("ℹ️  Main agent doesn't have start_consciousness_simulation method - continuing")
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
        print("🛑 Stopping Cloud-Integrated Consciousness System...")
        
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
        """Main integration loop with cloud offload"""
        print("🔄 Starting integration loop with cloud offload...")
        
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
                
                # Check for heavy cognitive load and offload to cloud
                self._check_and_offload_heavy_tasks()
                
                # Update timestamp
                self.integration_state['last_sync'] = datetime.now().isoformat()
                
                time.sleep(0.1)  # 10 Hz update rate
                
            except Exception as e:
                print(f"❌ Integration loop error: {e}")
                time.sleep(1.0)
    
    def _check_and_offload_heavy_tasks(self):
        """Check for heavy cognitive load and offload tasks to cloud"""
        if not hasattr(self, 'check_cloud_offload'):
            return
        
        consciousness_level = self.unified_state['consciousness_level']
        cognitive_load = self.unified_state.get('main_agent_state', {}).get('cognitive_load', 0.0)
        
        # Use the cloud integration method if available
        if self.check_cloud_offload(consciousness_level, cognitive_load):
            print(f"🧠 Cloud offload triggered for consciousness level: {consciousness_level:.2f}")
    
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
                'brain_bridge': self.brain_bridge is not None,
                'cloud_offload': hasattr(self, 'cloud_integration')
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
    
    def run_interactive_mode(self):
        """Run interactive mode for testing integration"""
        print("🎤 Interactive Cloud-Integrated Consciousness Mode")
        print("Commands: status, speak, report, cloud_status, manual_offload, quit")
        
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
                elif command == 'cloud_status':
                    if hasattr(self, 'get_cloud_status'):
                        status = self.get_cloud_status()
                        print(json.dumps(status, indent=2))
                    else:
                        print("Cloud integration not available")
                elif command == 'manual_offload':
                    if hasattr(self, 'manual_cloud_offload'):
                        task_type = input("Task type (neural_simulation/memory_consolidation/attention_modeling): ")
                        duration = int(input("Duration (ms): "))
                        scale = float(input("Scale (0-1): "))
                        
                        result = self.manual_cloud_offload(task_type, {
                            'duration': duration,
                            'scale': scale
                        })
                        
                        if result:
                            print(f"✅ Cloud offload completed: {result}")
                        else:
                            print("❌ Cloud offload failed")
                    else:
                        print("Cloud integration not available")
                else:
                    print("Unknown command. Use: status, speak, report, cloud_status, manual_offload, quit")
                    
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
        print(f"  Cloud Offload: {'✅ Ready' if hasattr(self, 'cloud_integration') else '❌ Not Available'}")
        print(f"  Integration Active: {'✅ Active' if self.integration_active else '❌ Inactive'}")
        print(f"  Last Sync: {self.integration_state['last_sync'] or 'Never'}")
        
        if self.unified_state['consciousness_level'] > 0:
            print(f"  Consciousness Level: {self.unified_state['consciousness_level']:.2f}")
            print(f"  Emotional State: {self.unified_state.get('emotional_state', 'unknown')}")

def main():
    """Main function for cloud-integrated consciousness"""
    print("🧠☁️ Cloud-Integrated Main Consciousness Agent")
    print("=" * 60)
    
    # Create integrated consciousness
    integrated = CloudIntegratedConsciousness()
    
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
        print("Cloud-integrated consciousness completed!")

if __name__ == "__main__":
    main()
