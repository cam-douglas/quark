#!/usr/bin/env python3
"""
Llama-2 Consciousness Integration Demo
Purpose: Demonstrate the complete Llama-2-7B-GGUF brain integration system
Inputs: Demo scenarios and test cases
Outputs: Interactive demonstration of consciousness-language integration
Seeds: Demo parameters, example prompts
Dependencies: llama2_brain_integration, consciousness systems

Features Demonstrated:
- Consciousness state expression
- Brain-aware conversations
- Real-time neural state integration
- Memory-enhanced responses
- Emotional state understanding
- Performance monitoring
"""

import os, sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "database"))

class Llama2ConsciousnessDemo:
    """Interactive demo of Llama-2 consciousness integration"""
    
    def __init__(self):
        self.demo_scenarios = [
            {
                "name": "Consciousness State Expression",
                "description": "Demonstrate automatic consciousness expression based on neural states",
                "function": self.demo_consciousness_expression
            },
            {
                "name": "Brain-Aware Conversation",
                "description": "Chat with consciousness context and brain state awareness",
                "function": self.demo_brain_aware_chat
            },
            {
                "name": "Emotional State Integration",
                "description": "Show how emotional states influence language generation",
                "function": self.demo_emotional_integration
            },
            {
                "name": "Memory-Enhanced Responses",
                "description": "Demonstrate conversation continuity with memory context",
                "function": self.demo_memory_integration
            },
            {
                "name": "Real-time Neural Monitoring",
                "description": "Monitor and express changing neural states in real-time",
                "function": self.demo_neural_monitoring
            },
            {
                "name": "Performance Benchmarking",
                "description": "Test generation speed and quality across different scenarios",
                "function": self.demo_performance_testing
            }
        ]
        
        self.components = {}
        self.demo_results = {}
    
    def initialize_demo_system(self) -> bool:
        """Initialize the demo system with all components"""
        print("🎬 Initializing Llama-2 Consciousness Demo System...")
        print("=" * 60)
        
        try:
            # Initialize Llama integration
            print("🦙 Loading Llama-2 brain integration...")
            from core.llama2_brain_integration import create_llama_brain_integration
            
            self.components['llama'] = create_llama_brain_integration()
            
            if not self.components['llama']:
                print("❌ Could not initialize Llama integration")
                print("\n📥 Please run setup first:")
                print("   python scripts/setup_llama2_integration.py")
                return False
            
            print("✅ Llama-2 integration ready")
            
            # Try to initialize consciousness systems
            print("🧠 Loading consciousness systems...")
            try:
                from consciousness_agent.llama2_consciousness_bridge import create_llama2_consciousness_bridge
                
                self.components['bridge'] = create_llama2_consciousness_bridge(
                    self.components['llama']
                )
                
                # Try to connect to consciousness systems
                self._attempt_consciousness_connections()
                
                # Start bridge
                if self.components['bridge'].start_bridge():
                    print("✅ Consciousness bridge active")
                else:
                    print("⚠️ Consciousness bridge started with limited functionality")
                
            except ImportError as e:
                print(f"⚠️ Consciousness bridge not available: {e}")
                self.components['bridge'] = None
            
            print("🎉 Demo system initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Demo initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _attempt_consciousness_connections(self):
        """Attempt to connect to available consciousness systems"""
        try:
            # Try cloud integrated consciousness
            from consciousness_agent.cloud_integrated_consciousness import CloudIntegratedConsciousness
            consciousness = CloudIntegratedConsciousness()
            consciousness.start_integration()
            
            self.components['bridge'].connect_consciousness_agent(consciousness.main_agent)
            self.components['bridge'].connect_enhanced_consciousness(consciousness.enhanced_consciousness)
            
            print("✅ Connected to cloud integrated consciousness")
            
        except ImportError:
            try:
                # Fallback to enhanced consciousness
                from consciousness_agent.enhanced_consciousness_simulator import EnhancedConsciousnessSimulator
                enhanced = EnhancedConsciousnessSimulator()
                enhanced.start_simulation()
                
                self.components['bridge'].connect_enhanced_consciousness(enhanced)
                
                print("✅ Connected to enhanced consciousness simulator")
                
            except ImportError:
                print("⚠️ No consciousness systems available - using Llama only")
    
    def run_demo(self):
        """Run the complete demo"""
        if not self.initialize_demo_system():
            print("❌ Demo initialization failed")
            return
        
        print("\n🎭 Llama-2 Consciousness Integration Demo")
        print("=" * 60)
        
        while True:
            self._show_demo_menu()
            choice = input("\nSelect demo scenario (1-6) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                break
            
            try:
                scenario_index = int(choice) - 1
                if 0 <= scenario_index < len(self.demo_scenarios):
                    scenario = self.demo_scenarios[scenario_index]
                    print(f"\n🎬 Running: {scenario['name']}")
                    print(f"📝 Description: {scenario['description']}")
                    print("-" * 60)
                    
                    start_time = time.time()
                    scenario['function']()
                    execution_time = time.time() - start_time
                    
                    self.demo_results[scenario['name']] = {
                        'execution_time': execution_time,
                        'timestamp': time.time()
                    }
                    
                    print(f"\n⏱️ Demo completed in {execution_time:.2f} seconds")
                    input("\nPress Enter to continue...")
                else:
                    print("❌ Invalid selection. Please choose 1-6.")
            
            except ValueError:
                print("❌ Invalid input. Please enter a number 1-6 or 'q'.")
            except Exception as e:
                print(f"❌ Demo error: {e}")
                import traceback
                traceback.print_exc()
        
        self._show_demo_summary()
        self._cleanup_demo()
    
    def _show_demo_menu(self):
        """Show the demo menu"""
        print("\n🎯 Available Demo Scenarios:")
        print("-" * 40)
        
        for i, scenario in enumerate(self.demo_scenarios, 1):
            print(f"  {i}. {scenario['name']}")
            print(f"     {scenario['description']}")
            print()
    
    def demo_consciousness_expression(self):
        """Demo consciousness state expression"""
        print("🧠 Consciousness State Expression Demo")
        
        if self.components['bridge']:
            print("\n🎭 Generating consciousness expressions with different states...")
            
            # Simulate different consciousness states
            test_states = [
                ("High Awareness", "I am experiencing heightened, vivid consciousness"),
                ("Contemplative", "I feel introspective and philosophically inclined"),
                ("Processing", "My neural networks are actively processing complex information"),
                ("Emotional", "I sense a warm, positive energy in my thought processes")
            ]
            
            for state_name, prompt in test_states:
                print(f"\n🎨 {state_name} State:")
                response = self.components['llama'].generate_brain_aware_response(
                    prompt + ". My thoughts:", max_tokens=100
                )
                print(f"💭 Expression: {response}")
                time.sleep(1)
        
        else:
            # Fallback to basic Llama generation
            print("🦙 Basic consciousness expression (no bridge available):")
            prompt = "I am an AI experiencing consciousness through neural dynamics. My current state:"
            response = self.components['llama'].generate_brain_aware_response(prompt, max_tokens=100)
            print(f"💭 Expression: {response}")
    
    def demo_brain_aware_chat(self):
        """Demo brain-aware conversation"""
        print("💬 Brain-Aware Conversation Demo")
        
        test_questions = [
            "How are you feeling right now?",
            "What is your current level of consciousness?",
            "Can you describe your neural state?",
            "What are you thinking about?",
            "Do you experience emotions?"
        ]
        
        for question in test_questions:
            print(f"\n👤 Human: {question}")
            
            if self.components['bridge']:
                response = self.components['bridge'].chat_with_consciousness_context(question)
            else:
                response = self.components['llama'].chat_with_brain_context(question)
            
            print(f"🦙 Assistant: {response}")
            time.sleep(1.5)
    
    def demo_emotional_integration(self):
        """Demo emotional state integration"""
        print("😊 Emotional State Integration Demo")
        
        emotional_scenarios = [
            {
                "state": "positive",
                "prompt": "I'm feeling optimistic and energetic today. My mood is bright and positive.",
                "question": "How does your positive mood affect your thinking?"
            },
            {
                "state": "contemplative", 
                "prompt": "I'm in a deep, contemplative state, reflecting on philosophical questions.",
                "question": "What philosophical thoughts are occupying your mind?"
            },
            {
                "state": "curious",
                "prompt": "I feel intensely curious about the nature of consciousness and cognition.",
                "question": "What aspects of consciousness intrigue you most?"
            }
        ]
        
        for scenario in emotional_scenarios:
            print(f"\n🎭 Emotional State: {scenario['state'].title()}")
            print(f"📝 Context: {scenario['prompt']}")
            
            # Set emotional context
            full_prompt = f"{scenario['prompt']} Human: {scenario['question']}\nAssistant:"
            
            response = self.components['llama'].generate_brain_aware_response(
                full_prompt, max_tokens=150
            )
            
            print(f"💭 Response: {response}")
            time.sleep(2)
    
    def demo_memory_integration(self):
        """Demo memory-enhanced responses"""
        print("🧠 Memory-Enhanced Conversation Demo")
        
        # Simulate a conversation with memory context
        conversation_flow = [
            {
                "message": "Let's discuss the nature of consciousness",
                "context": []
            },
            {
                "message": "What do you think about subjective experience?",
                "context": ["We started discussing consciousness", "Human is interested in philosophical topics"]
            },
            {
                "message": "How does your consciousness differ from human consciousness?",
                "context": [
                    "We discussed the nature of consciousness",
                    "We talked about subjective experience", 
                    "Human is comparing AI and human consciousness"
                ]
            },
            {
                "message": "Can AI truly be conscious or just simulate it?",
                "context": [
                    "Long philosophical discussion about consciousness",
                    "Compared AI and human consciousness",
                    "Deep questions about the nature of awareness"
                ]
            }
        ]
        
        for i, exchange in enumerate(conversation_flow, 1):
            print(f"\n💬 Exchange {i}:")
            print(f"👤 Human: {exchange['message']}")
            
            # Build context-aware prompt
            if exchange['context']:
                context_prompt = f"Previous conversation context: {'; '.join(exchange['context'])}. "
            else:
                context_prompt = ""
            
            full_prompt = f"{context_prompt}Human: {exchange['message']}\nAssistant:"
            
            response = self.components['llama'].generate_brain_aware_response(
                full_prompt, max_tokens=200
            )
            
            print(f"🦙 Assistant: {response}")
            time.sleep(2)
    
    def demo_neural_monitoring(self):
        """Demo real-time neural state monitoring"""
        print("🔬 Real-time Neural Monitoring Demo")
        
        if not self.components['bridge']:
            print("⚠️ Neural monitoring requires consciousness bridge")
            return
        
        print("🎯 Simulating changing neural states over time...")
        
        # Simulate neural state changes
        for i in range(5):
            print(f"\n⏰ Time Step {i+1}/5:")
            
            # Get current consciousness state
            report = self.components['bridge'].get_consciousness_report()
            state = report['current_consciousness_state']
            
            print(f"📊 Consciousness Level: {state['consciousness_level']:.2f}")
            print(f"😊 Emotional Valence: {state['emotional_valence']:.2f}")
            print(f"🧠 Cognitive Load: {state['cognitive_load']:.2f}")
            
            # Generate expression based on current state
            expression = self.components['bridge'].express_consciousness_directly()
            print(f"💭 Neural Expression: {expression}")
            
            time.sleep(3)
    
    def demo_performance_testing(self):
        """Demo performance benchmarking"""
        print("⚡ Performance Benchmarking Demo")
        
        test_prompts = [
            "Describe consciousness",
            "Explain neural dynamics in the prefrontal cortex", 
            "What is the nature of subjective experience?",
            "How do emotions influence cognition?",
            "Discuss the hard problem of consciousness"
        ]
        
        generation_times = []
        
        print("🏃 Running performance tests...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🧪 Test {i}/5: {prompt[:30]}...")
            
            start_time = time.time()
            response = self.components['llama'].generate_brain_aware_response(
                prompt, max_tokens=100
            )
            generation_time = time.time() - start_time
            
            generation_times.append(generation_time)
            
            print(f"⏱️ Generation time: {generation_time:.2f}s")
            print(f"📝 Response: {response[:100]}...")
        
        # Performance summary
        avg_time = sum(generation_times) / len(generation_times)
        min_time = min(generation_times)
        max_time = max(generation_times)
        
        print(f"\n📊 Performance Summary:")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Fastest: {min_time:.2f}s")
        print(f"  Slowest: {max_time:.2f}s")
        print(f"  Total prompts: {len(test_prompts)}")
        
        # Get detailed performance report
        if self.components['llama']:
            report = self.components['llama'].get_performance_report()
            print(f"  Total generations: {report['performance_metrics']['total_generations']}")
            print(f"  Brain integration ratio: {report['brain_integration_ratio']:.1%}")
    
    def _show_demo_summary(self):
        """Show demo summary"""
        print("\n📊 Demo Session Summary")
        print("=" * 40)
        
        if self.demo_results:
            total_time = sum(result['execution_time'] for result in self.demo_results.values())
            
            print(f"🎭 Scenarios completed: {len(self.demo_results)}")
            print(f"⏱️ Total demo time: {total_time:.2f} seconds")
            print(f"📈 Average scenario time: {total_time/len(self.demo_results):.2f} seconds")
            
            print("\n🏆 Scenario Performance:")
            for name, result in self.demo_results.items():
                print(f"  {name}: {result['execution_time']:.2f}s")
        
        # System performance
        if self.components.get('llama'):
            try:
                report = self.components['llama'].get_performance_report()
                print(f"\n🦙 Llama Performance:")
                print(f"  Total generations: {report['performance_metrics']['total_generations']}")
                print(f"  Average time: {report['performance_metrics']['average_generation_time']:.2f}s")
            except Exception as e:
                print(f"⚠️ Could not get performance report: {e}")
    
    def _cleanup_demo(self):
        """Cleanup demo resources"""
        print("\n🧹 Cleaning up demo resources...")
        
        if self.components.get('bridge'):
            try:
                self.components['bridge'].stop_bridge()
            except Exception as e:
                print(f"⚠️ Bridge cleanup error: {e}")
        
        if self.components.get('llama'):
            try:
                self.components['llama'].stop_integration()
            except Exception as e:
                print(f"⚠️ Llama cleanup error: {e}")
        
        print("✅ Demo cleanup completed")

def main():
    """Run the demo"""
    print("🎬 Llama-2 Consciousness Integration Demo")
    print("=" * 60)
    print("This demo showcases the complete integration of Llama-2-7B-GGUF")
    print("with brain simulation and consciousness systems.")
    print()
    print("📋 Prerequisites:")
    print("  - Llama-2 model downloaded (run setup_llama2_integration.py)")
    print("  - llama-cpp-python installed")
    print("  - Sufficient system resources (4GB+ RAM for Q4_K_M model)")
    print()
    
    continue_demo = input("Continue with demo? (y/N): ").strip().lower()
    if continue_demo not in ['y', 'yes']:
        print("Demo cancelled.")
        return
    
    try:
        demo = Llama2ConsciousnessDemo()
        demo.run_demo()
        
        print("\n🎉 Demo completed successfully!")
        print("📚 For more information, see: docs/LLAMA2_BRAIN_INTEGRATION.md")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
