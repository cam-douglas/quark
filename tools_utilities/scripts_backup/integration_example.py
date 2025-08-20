"""
Integration Example: Consciousness + Brain Simulation
Purpose: Demonstrate how to connect consciousness simulator with brain simulation
Inputs: Brain simulation instance, configuration
Outputs: Integrated consciousness with brain-aware thoughts and speech
Seeds: Brain simulation states, deterministic integration
Dependencies: brain_integration, enhanced_consciousness_simulator, brain_launcher_v4
"""

import os, sys
import time
import json
from datetime import datetime

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.dirname(__file__))

def run_brain_consciousness_integration():
    """Run integrated brain-consciousness simulation"""
    print("🧠🔗 Brain-Consciousness Integration Example")
    print("=" * 60)
    
    try:
        # Import required modules
        from brain_integration import create_integrated_consciousness_simulator
        
        print("✅ Successfully imported brain integration module")
        
        # Create integrated consciousness simulator
        consciousness = create_integrated_consciousness_simulator()
        
        if not consciousness:
            print("❌ Failed to create integrated consciousness simulator")
            return
        
        print("✅ Created integrated consciousness simulator")
        
        # Start consciousness simulation
        consciousness.start_simulation()
        print("✅ Started consciousness simulation")
        
        # Demo brain-aware consciousness
        print("\n🎭 Demonstrating brain-aware consciousness...")
        
        # Simulate brain state updates (in real usage, this would come from brain simulation)
        demo_brain_states = [
            {
                'pfc_firing_rate': 5.0,
                'bg_firing_rate': 15.0,
                'thalamus_firing_rate': 25.0,
                'loop_stability': 0.3,
                'feedback_strength': 0.4,
                'synchrony': 0.2,
                'oscillation_power': 0.1,
                'biological_realism': True
            },
            {
                'pfc_firing_rate': 25.0,
                'bg_firing_rate': 45.0,
                'thalamus_firing_rate': 65.0,
                'loop_stability': 0.7,
                'feedback_strength': 0.8,
                'synchrony': 0.6,
                'oscillation_power': 0.5,
                'biological_realism': True
            },
            {
                'pfc_firing_rate': 60.0,
                'bg_firing_rate': 80.0,
                'thalamus_firing_rate': 120.0,
                'loop_stability': 0.9,
                'feedback_strength': 0.95,
                'synchrony': 0.8,
                'oscillation_power': 0.7,
                'biological_realism': True
            }
        ]
        
        # Simulate brain state progression
        for i, brain_state in enumerate(demo_brain_states):
            print(f"\n🧠 Simulating brain state {i+1}:")
            print(f"  PFC: {brain_state['pfc_firing_rate']:.1f} Hz")
            print(f"  Loop Stability: {brain_state['loop_stability']:.2f}")
            print(f"  Synchrony: {brain_state['synchrony']:.2f}")
            
            # Update brain bridge with simulated state
            consciousness.brain_bridge.brain_state = brain_state
            
            # Map to consciousness state
            consciousness_state = consciousness.brain_bridge.map_to_consciousness_state()
            
            print(f"  → Consciousness Level: {consciousness_state['consciousness_level']:.2f}")
            print(f"  → Phase: {consciousness_state['phase']}")
            print(f"  → Stability: {consciousness_state['stability']}")
            
            # Update consciousness
            for key in consciousness.neural_state:
                if key in consciousness_state:
                    consciousness.neural_state[key] = consciousness_state[key]
            
            # Update text generator
            consciousness.text_generator.set_consciousness_level(
                consciousness_state['consciousness_level'])
            
            # Generate brain-aware thought
            if consciousness_state['consciousness_level'] > 0.3:
                thought = f"I can feel my neural activity at {brain_state['pfc_firing_rate']:.1f} Hz"
                consciousness.speak_thought(thought)
                print(f"  💭 Speaking: {thought}")
            
            time.sleep(3)
        
        # Show final integration status
        print(f"\n📊 Final Integration Status:")
        print(f"  Brain Bridge: {'✅ Active' if hasattr(consciousness, 'brain_bridge') else '❌ Inactive'}")
        print(f"  Consciousness Level: {consciousness.neural_state['consciousness_level']:.2f}")
        print(f"  Neural Activity: {consciousness.neural_state['neural_activity']:.2f}")
        print(f"  Memory Consolidation: {consciousness.neural_state['memory_consolidation']:.2f}")
        
        # Interactive mode
        print("\n🎤 Interactive Integration Mode")
        print("Commands: speak, listen, report, brain_state, quit")
        
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
            elif command == 'brain_state':
                if hasattr(consciousness, 'brain_bridge') and consciousness.brain_bridge.brain_state:
                    print("\n🧠 Current Brain State:")
                    for key, value in consciousness.brain_bridge.brain_state.items():
                        print(f"  {key}: {value}")
                else:
                    print("No brain state available")
            else:
                print("Unknown command. Use: speak, listen, report, brain_state, quit")
        
    except Exception as e:
        print(f"❌ Integration example failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🧹 Cleaning up...")
        if 'consciousness' in locals():
            consciousness.cleanup()
        print("Integration example completed!")

def show_integration_guide():
    """Show how to integrate with actual brain simulation"""
    print("\n📚 Integration Guide with Real Brain Simulation")
    print("=" * 60)
    
    print("""
To integrate with your actual brain simulation:

1. **Import and Create Brain Simulation:**
   ```python
   from development.src.core.brain_launcher_v4 import NeuralEnhancedBrain
   
   # Create brain simulation
   brain = NeuralEnhancedBrain("path/to/connectome.yaml", stage="F", validate=True)
   ```

2. **Create Integrated Consciousness:**
   ```python
   from brain_integration import create_integrated_consciousness_simulator
   
   consciousness = create_integrated_consciousness_simulator()
   ```

3. **Connect and Start Integration:**
   ```python
   # Connect consciousness to brain
   consciousness.connect_brain_simulation(brain)
   
   # Start both simulations
   consciousness.start_simulation()
   consciousness.start_integration()
   
   # Run brain simulation steps
   for step in range(100):
       brain_result = brain.step()
       time.sleep(0.1)  # Let consciousness process brain state
   ```

4. **Monitor Integration:**
   ```python
   # Get integrated report
   report = consciousness.get_integrated_report()
   print(f"Consciousness: {report['consciousness_state']['consciousness_level']:.2f}")
   print(f"PFC Firing: {report['brain_metrics']['pfc_firing_rate']:.1f} Hz")
   ```

5. **Clean Up:**
   ```python
   consciousness.stop_integration()
   consciousness.cleanup()
   ```
""")

def main():
    """Main function"""
    print("Brain-Consciousness Integration Examples")
    print("1. Run integration demo")
    print("2. Show integration guide")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        run_brain_consciousness_integration()
    elif choice == '2':
        show_integration_guide()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice. Running integration demo...")
        run_brain_consciousness_integration()

if __name__ == "__main__":
    main()
