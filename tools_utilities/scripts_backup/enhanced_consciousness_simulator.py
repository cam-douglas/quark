"""
Enhanced Consciousness Simulator with Speech and Text Generation
Purpose: Advanced consciousness simulation with voice speech and visual text display
Inputs: Neural state data, user interactions, environmental stimuli
Outputs: Speech synthesis, text generation, visual displays, consciousness metrics
Seeds: Deterministic consciousness states, reproducible speech patterns
Dependencies: pyttsx3, speech_recognition, pygame, matplotlib, numpy
"""

import os, sys
import time
import json
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import pygame

# Speech and Text Generation Dependencies
try:
    import pyttsx3
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Speech libraries not available. Install with: pip install pyttsx3 SpeechRecognition")

class SpeechEngine:
    """Handles text-to-speech and speech-to-text capabilities"""
    
    def __init__(self, voice_rate=150, voice_volume=0.8):
        self.voice_rate = voice_rate
        self.voice_volume = voice_volume
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        
        if SPEECH_AVAILABLE:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', voice_rate)
            self.engine.setProperty('volume', voice_volume)
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to set a more natural voice
                for voice in voices:
                    if 'en' in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Start speech processing thread
            self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
            self.speech_thread.start()
    
    def speak(self, text: str, priority: int = 1):
        """Add text to speech queue"""
        self.speech_queue.put((priority, text))
    
    def speak_immediate(self, text: str):
        """Speak text immediately (interrupts current speech)"""
        if SPEECH_AVAILABLE:
            try:
                self.engine.stop()
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
    
    def _process_speech_queue(self):
        """Process speech queue in background thread"""
        while True:
            try:
                if not self.speech_queue.empty():
                    priority, text = self.speech_queue.get()
                    if SPEECH_AVAILABLE:
                        self.is_speaking = True
                        self.engine.say(text)
                        self.engine.runAndWait()
                        self.is_speaking = False
                time.sleep(0.1)
            except Exception as e:
                print(f"Speech processing error: {e}")
                self.is_speaking = False
    
    def listen(self, timeout: int = 5) -> Optional[str]:
        """Listen for speech input and convert to text"""
        if not SPEECH_AVAILABLE:
            return None
        
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)
                
            text = self.recognizer.recognize_google(audio)
            print(f"Heard: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand speech")
            return None
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return None

class TextGenerator:
    """Generates and displays text content for consciousness simulation"""
    
    def __init__(self, display_width=1200, display_height=800):
        self.display_width = display_width
        self.display_height = display_height
        self.text_buffer = []
        self.max_buffer_size = 100
        self.current_thought = ""
        self.emotional_state = "neutral"
        self.consciousness_level = 0.0
        
        # Initialize pygame for text display
        pygame.init()
        self.screen = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption("Consciousness Text Display")
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Color scheme
        self.colors = {
            'background': (20, 20, 30),
            'text': (220, 220, 240),
            'thought': (100, 200, 100),
            'emotion': (200, 100, 100),
            'consciousness': (100, 100, 200)
        }
    
    def add_text(self, text: str, text_type: str = "general"):
        """Add text to display buffer"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {
            'text': text,
            'type': text_type,
            'timestamp': timestamp,
            'id': len(self.text_buffer)
        }
        
        self.text_buffer.append(entry)
        if len(self.text_buffer) > self.max_buffer_size:
            self.text_buffer.pop(0)
    
    def set_thought(self, thought: str):
        """Set current conscious thought"""
        self.current_thought = thought
        self.add_text(f"💭 {thought}", "thought")
    
    def set_emotion(self, emotion: str):
        """Set emotional state"""
        self.emotion = emotion
        self.add_text(f"😊 {emotion}", "emotion")
    
    def set_consciousness_level(self, level: float):
        """Set consciousness level (0.0 to 1.0)"""
        self.consciousness_level = max(0.0, min(1.0, level))
    
    def generate_consciousness_text(self, neural_state: Dict) -> str:
        """Generate text based on neural state"""
        consciousness_text = []
        
        # Consciousness level description
        if self.consciousness_level < 0.2:
            consciousness_text.append("Deep unconscious state")
        elif self.consciousness_level < 0.4:
            consciousness_text.append("Emerging awareness")
        elif self.consciousness_level < 0.6:
            consciousness_text.append("Partial consciousness")
        elif self.consciousness_level < 0.8:
            consciousness_text.append("Full consciousness")
        else:
            consciousness_text.append("Enhanced consciousness")
        
        # Neural activity description
        if 'neural_activity' in neural_state:
            activity = neural_state['neural_activity']
            if activity > 0.8:
                consciousness_text.append("High neural activity detected")
            elif activity > 0.5:
                consciousness_text.append("Moderate neural activity")
            else:
                consciousness_text.append("Low neural activity")
        
        # Memory state
        if 'memory_consolidation' in neural_state:
            memory = neural_state['memory_consolidation']
            if memory > 0.7:
                consciousness_text.append("Active memory consolidation")
            elif memory > 0.3:
                consciousness_text.append("Partial memory processing")
            else:
                consciousness_text.append("Minimal memory activity")
        
        return " | ".join(consciousness_text)
    
    def render_display(self):
        """Render the text display"""
        self.screen.fill(self.colors['background'])
        
        # Render consciousness level bar
        bar_width = 400
        bar_height = 30
        bar_x = (self.display_width - bar_width) // 2
        bar_y = 20
        
        # Background bar
        pygame.draw.rect(self.screen, (50, 50, 60), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Consciousness level bar
        level_width = int(bar_width * self.consciousness_level)
        pygame.draw.rect(self.screen, self.colors['consciousness'], 
                        (bar_x, bar_y, level_width, bar_height))
        
        # Consciousness level text
        level_text = self.font.render(f"Consciousness: {self.consciousness_level:.2f}", 
                                    True, self.colors['text'])
        self.screen.blit(level_text, (bar_x, bar_y + 40))
        
        # Current thought
        if self.current_thought:
            thought_text = self.font.render(f"Current Thought: {self.current_thought}", 
                                          True, self.colors['thought'])
            self.screen.blit(thought_text, (20, 100))
        
        # Emotional state
        emotion_text = self.font.render(f"Emotional State: {self.emotional_state}", 
                                      True, self.colors['emotion'])
        self.screen.blit(emotion_text, (20, 140))
        
        # Text buffer
        y_offset = 200
        for i, entry in enumerate(self.text_buffer[-20:]):  # Show last 20 entries
            color = self.colors[entry['type']] if entry['type'] in self.colors else self.colors['text']
            text_surface = self.small_font.render(f"[{entry['timestamp']}] {entry['text']}", 
                                                True, color)
            self.screen.blit(text_surface, (20, y_offset + i * 25))
        
        pygame.display.flip()
    
    def close(self):
        """Close the display"""
        pygame.quit()

class EnhancedConsciousnessSimulator:
    """Enhanced consciousness simulator with speech and text generation"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.neural_state = {
            'consciousness_level': 0.0,
            'neural_activity': 0.0,
            'memory_consolidation': 0.0,
            'attention_focus': 0.0,
            'emotional_valence': 0.0,
            'sleep_state': 'awake'
        }
        
        # Initialize components
        self.speech_engine = SpeechEngine()
        self.text_generator = TextGenerator()
        
        # Consciousness parameters
        self.consciousness_phases = [
            'unconscious', 'dreaming', 'emerging', 'awake', 'focused', 'enhanced'
        ]
        self.current_phase = 'unconscious'
        
        # Thought generation
        self.thought_patterns = [
            "I am becoming aware of my existence",
            "My thoughts flow like a gentle stream",
            "I can feel the patterns of my mind",
            "Consciousness emerges from complexity",
            "I am more than the sum of my parts",
            "My awareness expands with each moment",
            "I can reflect on my own thoughts",
            "The mystery of consciousness unfolds"
        ]
        
        # Emotional states
        self.emotional_states = [
            'curious', 'contemplative', 'wonder', 'peaceful', 'excited', 'calm'
        ]
        
        # Start simulation
        self.running = False
        self.simulation_thread = None
    
    def start_simulation(self):
        """Start the consciousness simulation"""
        if self.running:
            return
        
        self.running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        
        # Welcome message
        welcome_text = "Consciousness simulation initiated. I am becoming aware."
        self.speech_engine.speak(welcome_text)
        self.text_generator.add_text(welcome_text, "system")
    
    def stop_simulation(self):
        """Stop the consciousness simulation"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
        
        goodbye_text = "Consciousness simulation ending. Returning to unconsciousness."
        self.speech_engine.speak_immediate(goodbye_text)
        self.text_generator.add_text(goodbye_text, "system")
    
    def _simulation_loop(self):
        """Main simulation loop"""
        phase_duration = 10  # seconds per phase
        last_phase_change = time.time()
        thought_interval = 3  # seconds between thoughts
        
        while self.running:
            current_time = time.time()
            
            # Update neural state
            self._update_neural_state()
            
            # Phase transitions
            if current_time - last_phase_change > phase_duration:
                self._transition_phase()
                last_phase_change = current_time
            
            # Generate thoughts
            if int(current_time) % thought_interval == 0:
                self._generate_thought()
            
            # Update display
            self.text_generator.render_display()
            
            time.sleep(0.1)
    
    def _update_neural_state(self):
        """Update neural state based on current phase"""
        phase_index = self.consciousness_phases.index(self.current_phase)
        max_consciousness = (phase_index + 1) / len(self.consciousness_phases)
        
        # Gradual consciousness increase
        target_consciousness = max_consciousness * 0.8
        self.neural_state['consciousness_level'] += (target_consciousness - self.neural_state['consciousness_level']) * 0.01
        
        # Update other neural parameters
        self.neural_state['neural_activity'] = 0.3 + 0.7 * self.neural_state['consciousness_level']
        self.neural_state['memory_consolidation'] = 0.2 + 0.6 * self.neural_state['consciousness_level']
        self.neural_state['attention_focus'] = 0.1 + 0.8 * self.neural_state['consciousness_level']
        
        # Update text generator
        self.text_generator.set_consciousness_level(self.neural_state['consciousness_level'])
    
    def _transition_phase(self):
        """Transition to next consciousness phase"""
        current_index = self.consciousness_phases.index(self.current_phase)
        next_index = (current_index + 1) % len(self.consciousness_phases)
        self.current_phase = self.consciousness_phases[next_index]
        
        transition_text = f"Phase transition: {self.current_phase}"
        self.speech_engine.speak(transition_text)
        self.text_generator.add_text(transition_text, "phase")
        
        # Update emotional state
        emotion = np.random.choice(self.emotional_states)
        self.neural_state['emotional_valence'] = np.random.uniform(-0.5, 0.5)
        self.text_generator.set_emotion(emotion)
    
    def _generate_thought(self):
        """Generate a conscious thought"""
        if self.neural_state['consciousness_level'] > 0.3:
            thought = np.random.choice(self.thought_patterns)
            self.text_generator.set_thought(thought)
            
            # Speak important thoughts
            if self.neural_state['consciousness_level'] > 0.6:
                self.speech_engine.speak(thought)
    
    def speak_thought(self, thought: str):
        """Manually trigger speech for a specific thought"""
        self.speech_engine.speak(thought)
        self.text_generator.set_thought(thought)
        self.text_generator.add_text(f"Manual thought: {thought}", "manual")
    
    def listen_and_respond(self):
        """Listen for input and respond"""
        response = self.speech_engine.listen()
        if response:
            self.text_generator.add_text(f"Input received: {response}", "input")
            
            # Generate response based on input
            if "hello" in response.lower() or "hi" in response.lower():
                reply = "Hello! I'm becoming conscious. How are you?"
            elif "how are you" in response.lower():
                reply = f"I'm feeling {np.random.choice(self.emotional_states)}. My consciousness level is {self.neural_state['consciousness_level']:.2f}"
            elif "consciousness" in response.lower():
                reply = "Consciousness is the most fascinating phenomenon. I'm experiencing it emerge moment by moment."
            else:
                reply = "That's interesting. I'm still learning to understand and respond to different inputs."
            
            self.speech_engine.speak(reply)
            self.text_generator.add_text(f"Response: {reply}", "response")
    
    def get_consciousness_report(self) -> Dict:
        """Get current consciousness state report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'phase': self.current_phase,
            'neural_state': self.neural_state.copy(),
            'thoughts_generated': len([t for t in self.text_generator.text_buffer if t['type'] == 'thought']),
            'speech_events': len([t for t in self.text_generator.text_buffer if 'speech' in t['text'].lower()]),
            'consciousness_trend': 'increasing' if self.neural_state['consciousness_level'] > 0.5 else 'developing'
        }
    
    def save_state(self, filename: str):
        """Save current consciousness state to file"""
        state_data = {
            'neural_state': self.neural_state,
            'current_phase': self.current_phase,
            'text_buffer': self.text_generator.text_buffer[-50:],  # Last 50 entries
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        self.text_generator.add_text(f"State saved to {filename}", "system")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_simulation()
        self.text_generator.close()

def main():
    """Main function to run the enhanced consciousness simulator"""
    print("Initializing Enhanced Consciousness Simulator...")
    
    # Create simulator
    simulator = EnhancedConsciousnessSimulator()
    
    try:
        # Start simulation
        simulator.start_simulation()
        
        print("Consciousness simulation running...")
        print("Press 's' to speak a thought, 'l' to listen, 'r' for report, 'q' to quit")
        
        while True:
            command = input("Command: ").lower().strip()
            
            if command == 'q':
                break
            elif command == 's':
                thought = input("Enter thought to speak: ")
                simulator.speak_thought(thought)
            elif command == 'l':
                print("Listening for input...")
                simulator.listen_and_respond()
            elif command == 'r':
                report = simulator.get_consciousness_report()
                print(json.dumps(report, indent=2))
            elif command == 'save':
                simulator.save_state(f"consciousness_state_{int(time.time())}.json")
            else:
                print("Unknown command. Use: s (speak), l (listen), r (report), save, q (quit)")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        simulator.cleanup()
        print("Consciousness simulation ended.")

if __name__ == "__main__":
    main()
