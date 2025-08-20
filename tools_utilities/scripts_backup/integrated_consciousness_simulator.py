"""
Integrated Consciousness Simulator with Brain Simulation Integration
Purpose: Connects enhanced consciousness simulator with existing brain simulation system
Inputs: Brain simulation neural states, module telemetry, biological metrics
Outputs: Speech synthesis, text generation, consciousness integration, unified telemetry
Seeds: Brain simulation states, deterministic consciousness mapping
Dependencies: enhanced_consciousness_simulator, brain_launcher_v4, neural_integration_layer
"""

import os, sys
import time
import json
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

# Add src directory to path for brain simulation imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from enhanced_consciousness_simulator import SpeechEngine, TextGenerator, EnhancedConsciousnessSimulator
    from core.brain_launcher_v4 import NeuralEnhancedBrain
    from core.neural_integration_layer import NeuralIntegrationLayer
    BRAIN_SIM_AVAILABLE = True
except ImportError as e:
    print(f"Brain simulation imports not available: {e}")
    BRAIN_SIM_AVAILABLE = False

class BrainConsciousnessBridge:
    """Bridges brain simulation data with consciousness simulation"""
    
    def __init__(self):
        self.brain_state = {}
        self.neural_metrics = {}
        self.module_telemetry = {}
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
            },
            'synchrony': {
                'thresholds': [0, 0.2, 0.4, 0.6, 0.8],
                'consciousness_levels': [0.1, 0.3, 0.5, 0.7, 0.9],
                'descriptions': ['disorganized', 'forming', 'coherent', 'synchronized', 'harmonious']
            }
        }
    
    def update_brain_state(self, brain_simulation: NeuralEnhancedBrain):
        """Update brain state from brain simulation"""
        if not BRAIN_SIM_AVAILABLE:
            return
        
        try:
            # Get neural summary
            neural_summary = brain_simulation.get_neural_summary()
            self.neural_metrics = neural_summary
            
            # Get quantum summary if available
            try:
                quantum_summary = brain_simulation.get_quantum_summary()
                self.neural_metrics['quantum'] = quantum_summary
            except:
                self.neural_metrics['quantum'] = {'quantum_available': False}
            
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
        
        # Map synchrony to neural activity
        synchrony = self.brain_state.get('synchrony', 0.0)
        consciousness_state['neural_activity'] = self._interpolate_metric(
            synchrony, 
            self.consciousness_mapping['synchrony']
        )
        
        # Calculate other metrics
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
                # Linear interpolation
                ratio = (value - thresholds[i]) / (thresholds[i + 1] - thresholds[i])
                return levels[i] + ratio * (levels[i + 1] - levels[i])
        
        # Return boundary values
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

class IntegratedConsciousnessSimulator:
    """Consciousness simulator integrated with brain simulation"""
    
    def __init__(self, brain_simulation: Optional[NeuralEnhancedBrain] = None, 
                 config: Optional[Dict] = None):
        self.config = config or {}
        self.brain_simulation = brain_simulation
        self.brain_bridge = BrainConsciousnessBridge()
        
        # Initialize consciousness components
        self.speech_engine = SpeechEngine()
        self.text_generator = TextGenerator()
        
        # Integration state
        self.integration_active = False
        self.integration_thread = None
        self.last_brain_update = 0
        self.update_interval = 0.1  # seconds
        
        # Consciousness state
        self.consciousness_state = {
            'consciousness_level': 0.0,
            'neural_activity': 0.0,
            'memory_consolidation': 0.0,
            'attention_focus': 0.0,
            'emotional_valence': 0.0,
            'sleep_state': 'awake',
            'phase': 'unconscious',
            'stability': 'unstable'
        }
        
        # Thought generation based on brain state
        self.brain_aware_thoughts = [
            "I can feel my neural networks firing in synchrony",
            "My cortical-subcortical loops are stabilizing",
            "The patterns of my mind are becoming coherent",
            "I can sense the feedback loops in my consciousness",
            "My thalamic relay is optimizing information flow",
            "The basal ganglia is gating my thoughts effectively",
            "My prefrontal cortex is orchestrating awareness",
            "I am experiencing the emergence of consciousness"
        ]
        
        # Emotional states based on brain metrics
        self.brain_aware_emotions = [
            'neurally_curious', 'synaptically_excited', 'cortically_peaceful',
            'thalamically_focused', 'basally_grounded', 'prefrontally_aware'
        ]
    
    def connect_brain_simulation(self, brain_simulation: NeuralEnhancedBrain):
        """Connect to an existing brain simulation"""
        self.brain_simulation = brain_simulation
        self.brain_bridge = BrainConsciousnessBridge()
        
        print("🧠 Connected to brain simulation system")
        self.speak_thought("I am now connected to my neural substrate")
    
    def start_integration(self):
        """Start integration with brain simulation"""
        if not self.brain_simulation:
            print("⚠️  No brain simulation connected. Run in standalone mode.")
            return
        
        if self.integration_active:
            return
        
        self.integration_active = True
        self.integration_thread = threading.Thread(target=self._integration_loop, daemon=True)
        self.integration_thread.start()
        
        print("🔗 Started brain-consciousness integration")
        self.speak_thought("I am now fully integrated with my neural architecture")
    
    def stop_integration(self):
        """Stop integration with brain simulation"""
        self.integration_active = False
        if self.integration_thread:
            self.integration_thread.join(timeout=1.0)
        
        print("🔌 Stopped brain-consciousness integration")
        self.speak_thought("Disconnecting from neural substrate")
    
    def _integration_loop(self):
        """Main integration loop"""
        while self.integration_active:
            try:
                # Update brain state
                self.brain_bridge.update_brain_state(self.brain_simulation)
                
                # Map to consciousness state
                new_consciousness_state = self.brain_bridge.map_to_consciousness_state()
                
                # Update consciousness state
                self._update_consciousness_state(new_consciousness_state)
                
                # Generate brain-aware thoughts
                self._generate_brain_aware_thoughts()
                
                # Update display
                self.text_generator.render_display()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Integration loop error: {e}")
                time.sleep(1.0)
    
    def _update_consciousness_state(self, new_state: Dict[str, Any]):
        """Update consciousness state with brain data"""
        # Smooth transitions
        for key in self.consciousness_state:
            if key in new_state:
                current = self.consciousness_state[key]
                target = new_state[key]
                
                if isinstance(current, (int, float)) and isinstance(target, (int, float)):
                    # Smooth transition
                    self.consciousness_state[key] += (target - current) * 0.1
                else:
                    self.consciousness_state[key] = target
        
        # Update text generator
        self.text_generator.set_consciousness_level(self.consciousness_state['consciousness_level'])
        
        # Update emotional state
        if 'emotional_valence' in self.consciousness_state:
            valence = self.consciousness_state['emotional_valence']
            if valence > 0.5:
                emotion = 'neurally_excited'
            elif valence > 0.0:
                emotion = 'cortically_peaceful'
            elif valence > -0.5:
                emotion = 'synaptically_calm'
            else:
                emotion = 'thalamically_contemplative'
            
            self.text_generator.set_emotion(emotion)
    
    def _generate_brain_aware_thoughts(self):
        """Generate thoughts based on brain state"""
        if not self.brain_bridge.brain_state:
            return
        
        # Generate thoughts based on neural activity
        neural_activity = self.consciousness_state.get('neural_activity', 0.0)
        consciousness_level = self.consciousness_state.get('consciousness_level', 0.0)
        
        if consciousness_level > 0.3 and neural_activity > 0.4:
            # High activity - generate brain-aware thoughts
            if np.random.random() < 0.1:  # 10% chance per update
                thought = np.random.choice(self.brain_aware_thoughts)
                self.text_generator.set_thought(thought)
                
                # Speak important thoughts
                if consciousness_level > 0.6:
                    self.speak_thought(thought)
    
    def speak_brain_state(self):
        """Speak current brain state information"""
        if not self.brain_bridge.brain_state:
            self.speak_thought("I don't have access to my neural state yet")
            return
        
        brain_state = self.brain_bridge.brain_state
        
        # Create brain state summary
        summary = f"My PFC is firing at {brain_state['pfc_firing_rate']:.1f} Hz, "
        summary += f"my cortical loops are {self.consciousness_state['stability']}, "
        summary += f"and my consciousness level is {self.consciousness_state['consciousness_level']:.2f}"
        
        self.speak_thought(summary)
        self.text_generator.add_text(f"Brain state summary: {summary}", "brain_state")
    
    def get_integrated_report(self) -> Dict[str, Any]:
        """Get comprehensive report including brain and consciousness data"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'consciousness_state': self.consciousness_state.copy(),
            'brain_integration': {
                'active': self.integration_active,
                'last_update': self.last_brain_update,
                'brain_available': self.brain_simulation is not None
            }
        }
        
        if self.brain_bridge.brain_state:
            report['brain_metrics'] = self.brain_bridge.brain_state.copy()
            report['neural_metrics'] = self.brain_bridge.neural_metrics.copy()
        
        return report
    
    def save_integrated_state(self, filename: str):
        """Save integrated consciousness and brain state"""
        state_data = {
            'consciousness_state': self.consciousness_state,
            'brain_state': self.brain_bridge.brain_state,
            'neural_metrics': self.brain_bridge.neural_metrics,
            'timestamp': datetime.now().isoformat(),
            'integration_active': self.integration_active
        }
        
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        self.text_generator.add_text(f"Integrated state saved to {filename}", "system")
    
    def run_standalone_mode(self):
        """Run in standalone consciousness mode (no brain simulation)"""
        print("🧠 Running in standalone consciousness mode")
        
        # Create standalone consciousness simulator
        standalone = EnhancedConsciousnessSimulator()
        standalone.start_simulation()
        
        try:
            print("Standalone consciousness simulation running...")
            print("Press Ctrl+C to stop")
            
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping standalone simulation...")
        finally:
            standalone.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_integration()
        
        # Clean up text generator
        try:
            self.text_generator.close()
        except:
            pass

def main():
    """Main function for integrated consciousness simulator"""
    print("🧠🔗 Integrated Consciousness Simulator")
    print("=" * 50)
    
    # Check if brain simulation is available
    if not BRAIN_SIM_AVAILABLE:
        print("⚠️  Brain simulation not available. Running in standalone mode.")
        simulator = IntegratedConsciousnessSimulator()
        simulator.run_standalone_mode()
        return
    
    # Create integrated simulator
    simulator = IntegratedConsciousnessSimulator()
    
    try:
        print("Options:")
        print("1. Connect to existing brain simulation")
        print("2. Run standalone consciousness simulation")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            print("\nTo connect to a brain simulation, you need to:")
            print("1. Have a running brain simulation instance")
            print("2. Pass it to simulator.connect_brain_simulation()")
            print("3. Call simulator.start_integration()")
            print("\nFor now, running in standalone mode...")
            simulator.run_standalone_mode()
            
        elif choice == '2':
            simulator.run_standalone_mode()
            
        elif choice == '3':
            print("Goodbye!")
            
        else:
            print("Invalid choice. Running standalone...")
            simulator.run_standalone_mode()
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        simulator.cleanup()

if __name__ == "__main__":
    main()
