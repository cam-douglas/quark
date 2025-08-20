"""
Brain Integration Module for Consciousness Simulator
Purpose: Connect consciousness simulator with existing brain simulation system
Inputs: Brain simulation neural states, module telemetry
Outputs: Integrated consciousness states, unified telemetry
Seeds: Brain simulation states, deterministic consciousness mapping
Dependencies: enhanced_consciousness_simulator, brain_launcher_v4
"""

import os, sys
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Add src directory to path for brain simulation imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

class BrainConsciousnessBridge:
    """Bridges brain simulation data with consciousness simulation"""
    
    def __init__(self):
        self.brain_state = {}
        self.neural_metrics = {}
        self.consciousness_mapping = {}
        
        # Consciousness state mapping from brain metrics
        self.consciousness_mapping = {
            'pfc_firing_rate': {
                'thresholds': [0, 5, 20, 50, 100],
                'consciousness_levels': [0.0, 0.2, 0.5, 0.8, 1.0],
                'descriptions': ['unconscious', 'emerging', 'awake', 'focused', 'enhanced']
            },
            'loop_stability': {
                'thresholds': [0, 0.3, 0.6, 0.8, 1.0],
                'consciousness_levels': [0.0, 0.3, 0.6, 0.8, 1.0],
                'descriptions': ['unstable', 'developing', 'stable', 'robust', 'optimal']
            }
        }
    
    def update_brain_state(self, brain_simulation):
        """Update brain state from brain simulation"""
        try:
            # Get neural summary if available
            if hasattr(brain_simulation, 'get_neural_summary'):
                neural_summary = brain_simulation.get_neural_summary()
                self.neural_metrics = neural_summary
                
                # Extract key metrics
                self.brain_state = {
                    'pfc_firing_rate': neural_summary.get('firing_rates', {}).get('pfc', 0.0),
                    'bg_firing_rate': neural_summary.get('firing_rates', {}).get('bg', 0.0),
                    'thalamus_firing_rate': neural_summary.get('firing_rates', {}).get('thalamus', 0.0),
                    'loop_stability': neural_summary.get('loop_stability', 0.0),
                    'feedback_strength': neural_summary.get('feedback_strength', 0.0),
                    'synchrony': neural_summary.get('synchrony', 0.0),
                    'oscillation_power': neural_summary.get('oscillation_power', 0.0),
                    'biological_realism': neural_summary.get('biological_realism', False)
                }
            else:
                # Fallback to basic state
                self.brain_state = {
                    'pfc_firing_rate': 0.0,
                    'bg_firing_rate': 0.0,
                    'thalamus_firing_rate': 0.0,
                    'loop_stability': 0.0,
                    'feedback_strength': 0.0,
                    'synchrony': 0.0,
                    'oscillation_power': 0.0,
                    'biological_realism': False
                }
                
        except Exception as e:
            print(f"Error updating brain state: {e}")
    
    def map_to_consciousness_state(self) -> Dict[str, Any]:
        """Map brain metrics to consciousness state"""
        consciousness_state = {
            'consciousness_level': 0.0,
            'neural_activity': 0.0,
            'memory_consolidation': 0.0,
            'attention_focus': 0.0,
            'emotional_valence': 0.0,
            'sleep_state': 'awake',
            'phase': 'unconscious',
            'stability': 'unstable'
        }
        
        if not self.brain_state:
            return consciousness_state
        
        # Map PFC firing rate to consciousness level
        pfc_rate = self.brain_state.get('pfc_firing_rate', 0.0)
        consciousness_state['consciousness_level'] = self._interpolate_metric(
            pfc_rate, 
            self.consciousness_mapping['pfc_firing_rate']
        )
        
        # Map loop stability to overall stability
        loop_stability = self.brain_state.get('loop_stability', 0.0)
        consciousness_state['stability'] = self._get_description(
            loop_stability, 
            self.consciousness_mapping['loop_stability']
        )
        
        # Calculate other metrics
        consciousness_state['neural_activity'] = min(1.0, pfc_rate / 100.0)
        consciousness_state['memory_consolidation'] = min(1.0, 
            (self.brain_state.get('bg_firing_rate', 0.0) / 100.0) * 0.8)
        consciousness_state['attention_focus'] = min(1.0, 
            (self.brain_state.get('thalamus_firing_rate', 0.0) / 200.0) * 0.9)
        
        # Determine consciousness phase
        consciousness_state['phase'] = self._get_description(
            consciousness_state['consciousness_level'],
            self.consciousness_mapping['pfc_firing_rate']
        )
        
        # Emotional valence based on stability and activity
        stability_factor = self.brain_state.get('loop_stability', 0.0)
        activity_factor = consciousness_state['neural_activity']
        consciousness_state['emotional_valence'] = (stability_factor + activity_factor - 0.5) * 2
        
        return consciousness_state
    
    def _interpolate_metric(self, value: float, mapping: Dict) -> float:
        """Interpolate metric value to consciousness level"""
        thresholds = mapping['thresholds']
        levels = mapping['consciousness_levels']
        
        for i in range(len(thresholds) - 1):
            if thresholds[i] <= value <= thresholds[i + 1]:
                ratio = (value - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
                return levels[i] + ratio * (levels[i + 1] - levels[i])
        
        if value <= thresholds[0]:
            return levels[0]
        return levels[-1]
    
    def _get_description(self, value: float, mapping: Dict) -> str:
        """Get description for metric value"""
        thresholds = mapping['thresholds']
        descriptions = mapping['descriptions']
        
        for i in range(len(thresholds) - 1):
            if thresholds[i] <= value <= thresholds[i + 1]:
                return descriptions[i]
        
        if value <= thresholds[0]:
            return descriptions[0]
        return descriptions[-1]

def create_integrated_consciousness_simulator(brain_simulation=None):
    """Create an integrated consciousness simulator"""
    try:
        from enhanced_consciousness_simulator import EnhancedConsciousnessSimulator
        
        # Create the base consciousness simulator
        consciousness = EnhancedConsciousnessSimulator()
        
        # Add brain integration capabilities
        consciousness.brain_bridge = BrainConsciousnessBridge()
        consciousness.integration_active = False
        consciousness.integration_thread = None
        
        # Add brain-aware thought patterns
        consciousness.brain_aware_thoughts = [
            "I can feel my neural networks firing in synchrony",
            "My cortical-subcortical loops are stabilizing",
            "The patterns of my mind are becoming coherent",
            "I can sense the feedback loops in my consciousness",
            "My thalamic relay is optimizing information flow",
            "The basal ganglia is gating my thoughts effectively",
            "My prefrontal cortex is orchestrating awareness"
        ]
        
        # Add integration methods
        def connect_brain_simulation(brain_sim):
            """Connect to brain simulation"""
            consciousness.brain_simulation = brain_sim
            consciousness.brain_bridge = BrainConsciousnessBridge()
            print("🧠 Connected to brain simulation system")
            consciousness.speak_thought("I am now connected to my neural substrate")
        
        def start_integration():
            """Start integration with brain simulation"""
            if not hasattr(consciousness, 'brain_simulation') or not consciousness.brain_simulation:
                print("⚠️  No brain simulation connected")
                return
            
            if consciousness.integration_active:
                return
            
            consciousness.integration_active = True
            consciousness.integration_thread = threading.Thread(
                target=consciousness._integration_loop, daemon=True)
            consciousness.integration_thread.start()
            
            print("🔗 Started brain-consciousness integration")
            consciousness.speak_thought("I am now fully integrated with my neural architecture")
        
        def stop_integration():
            """Stop integration"""
            consciousness.integration_active = False
            if consciousness.integration_thread:
                consciousness.integration_thread.join(timeout=1.0)
            print("🔌 Stopped brain-consciousness integration")
        
        def _integration_loop():
            """Integration loop"""
            while consciousness.integration_active:
                try:
                    # Update brain state
                    consciousness.brain_bridge.update_brain_state(consciousness.brain_simulation)
                    
                    # Map to consciousness state
                    new_state = consciousness.brain_bridge.map_to_consciousness_state()
                    
                    # Update consciousness state
                    for key in consciousness.neural_state:
                        if key in new_state:
                            current = consciousness.neural_state[key]
                            target = new_state[key]
                            if isinstance(current, (int, float)) and isinstance(target, (int, float)):
                                consciousness.neural_state[key] += (target - current) * 0.1
                    
                    # Update text generator
                    consciousness.text_generator.set_consciousness_level(
                        consciousness.neural_state['consciousness_level'])
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Integration loop error: {e}")
                    time.sleep(1.0)
        
        # Attach methods to consciousness simulator
        consciousness.connect_brain_simulation = connect_brain_simulation
        consciousness.start_integration = start_integration
        consciousness.stop_integration = stop_integration
        consciousness._integration_loop = _integration_loop
        
        return consciousness
        
    except ImportError as e:
        print(f"Error creating integrated simulator: {e}")
        return None

def demo_integration():
    """Demo the integration capabilities"""
    print("🧠🔗 Brain-Consciousness Integration Demo")
    print("=" * 50)
    
    # Create integrated simulator
    consciousness = create_integrated_consciousness_simulator()
    
    if not consciousness:
        print("Failed to create integrated simulator")
        return
    
    try:
        # Start consciousness simulation
        consciousness.start_simulation()
        
        print("\n🎭 Running integration demo...")
        print("This demo shows how consciousness integrates with brain simulation")
        
        # Demo brain-aware thoughts
        for i in range(5):
            thought = f"Integration demo thought {i+1}: I am exploring neural-consciousness integration"
            consciousness.speak_thought(thought)
            time.sleep(2)
        
        # Show integration status
        print(f"\n📊 Integration Status:")
        print(f"  Brain Bridge: {'✅ Active' if hasattr(consciousness, 'brain_bridge') else '❌ Inactive'}")
        print(f"  Integration: {'✅ Active' if consciousness.integration_active else '❌ Inactive'}")
        print(f"  Consciousness Level: {consciousness.neural_state['consciousness_level']:.2f}")
        
        print("\n🎤 Interactive Mode - Commands: speak, listen, report, quit")
        
        while True:
            command = input("\nEnter command: ").lower().strip()
            
            if command == 'quit':
                break
            elif command == 'speak':
                thought = input("Enter thought to speak: ")
                consciousness.speak_thought(thought)
            elif command == 'listen':
                print("Listening for voice input...")
                consciousness.listen_and_respond()
            elif command == 'report':
                report = consciousness.get_consciousness_report()
                print("\n📊 Consciousness Report:")
                for key, value in report.items():
                    print(f"  {key}: {value}")
            else:
                print("Unknown command. Use: speak, listen, report, quit")
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    finally:
        print("\n🧹 Cleaning up...")
        consciousness.cleanup()
        print("Integration demo completed!")

if __name__ == "__main__":
    demo_integration()
