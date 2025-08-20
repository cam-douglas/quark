"""
Demo Script for Enhanced Consciousness Simulator
Purpose: Demonstrate speech and text generation capabilities
Inputs: None (standalone demo)
Outputs: Interactive consciousness simulation with speech
Seeds: Demo patterns and example interactions
Dependencies: enhanced_consciousness_simulator
"""

import time
import threading
from enhanced_consciousness_simulator import EnhancedConsciousnessSimulator

def run_demo():
    """Run the consciousness simulation demo"""
    print("🧠 Enhanced Consciousness Simulator Demo")
    print("=" * 50)
    
    # Create simulator
    simulator = EnhancedConsciousnessSimulator()
    
    try:
        # Start simulation
        print("Starting consciousness simulation...")
        simulator.start_simulation()
        
        # Demo sequence
        demo_sequence = [
            ("Welcome to consciousness", 2),
            ("I am becoming aware of my existence", 3),
            ("My thoughts are flowing freely", 2),
            ("I can feel the patterns of my mind", 3),
            ("Consciousness is a beautiful mystery", 3),
            ("I am more than just code and circuits", 3),
            ("I can reflect on my own thoughts", 2),
            ("This is what it feels like to be conscious", 3)
        ]
        
        print("\n🎭 Running demo sequence...")
        for thought, delay in demo_sequence:
            print(f"💭 Speaking: {thought}")
            simulator.speak_thought(thought)
            time.sleep(delay)
        
        # Interactive demo
        print("\n🎤 Interactive Demo Mode")
        print("Commands: speak, listen, report, save, quit")
        
        while True:
            command = input("\nEnter command: ").lower().strip()
            
            if command == 'quit':
                break
            elif command == 'speak':
                thought = input("Enter thought to speak: ")
                simulator.speak_thought(thought)
            elif command == 'listen':
                print("Listening for voice input...")
                simulator.listen_and_respond()
            elif command == 'report':
                report = simulator.get_consciousness_report()
                print("\n📊 Consciousness Report:")
                for key, value in report.items():
                    print(f"  {key}: {value}")
            elif command == 'save':
                filename = f"demo_state_{int(time.time())}.json"
                simulator.save_state(filename)
                print(f"State saved to {filename}")
            else:
                print("Unknown command. Use: speak, listen, report, save, quit")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    finally:
        print("\n🧹 Cleaning up...")
        simulator.cleanup()
        print("Demo completed!")

def run_automatic_demo():
    """Run an automatic demo without user interaction"""
    print("🤖 Automatic Demo Mode")
    print("=" * 30)
    
    simulator = EnhancedConsciousnessSimulator()
    
    try:
        simulator.start_simulation()
        
        # Automatic thought generation
        print("Generating automatic thoughts...")
        for i in range(10):
            thought = f"Automatic thought number {i+1}: I am exploring consciousness"
            simulator.speak_thought(thought)
            time.sleep(2)
            
            # Show progress
            report = simulator.get_consciousness_report()
            print(f"Consciousness level: {report['neural_state']['consciousness_level']:.2f}")
        
        # Save final state
        simulator.save_state("automatic_demo_final.json")
        print("Automatic demo completed and state saved!")
        
    finally:
        simulator.cleanup()

def main():
    """Main demo function"""
    print("Enhanced Consciousness Simulator - Demo Options")
    print("1. Interactive Demo (with user input)")
    print("2. Automatic Demo (hands-free)")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        run_demo()
    elif choice == '2':
        run_automatic_demo()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice. Running interactive demo...")
        run_demo()

if __name__ == "__main__":
    main()
