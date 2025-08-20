#!/usr/bin/env python3
"""
Speech Synthesis Agent - Provides speech output for the Unified Consciousness Agent
Integrates with visual display and consciousness simulation
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import sys

# Add database path
sys.path.append(os.path.join(os.path.dirname(__file__)))

class SpeechSynthesisAgent:
    def __init__(self, database_path: str = "database"):
        self.database_path = database_path
        
        # Speech configuration
        self.speech_config = {
            "enabled": True,
            "voice_type": "natural",
            "speech_rate": 150,  # words per minute
            "volume": 0.8,
            "pitch": 1.0,
            "language": "en-US"
        }
        
        # Speech patterns for different consciousness states
        self.speech_patterns = {
            "awake": [
                "Consciousness system fully operational",
                "All brain regions are active and learning",
                "Agent collaboration is functioning optimally",
                "Visual simulation is displaying real-time data"
            ],
            "learning": [
                "Processing new knowledge from multiple sources",
                "Integrating information across brain regions",
                "Synthetic data generation is active",
                "Training sessions are progressing well"
            ],
            "training": [
                "Biorxiv paper training in progress",
                "Unified learning architecture is evolving",
                "Knowledge mapping to brain regions",
                "Consciousness integration is successful"
            ],
            "collaboration": [
                "Agent collaboration session initiated",
                "Coordinated learning between all systems",
                "Cross-agent communication established",
                "Unified consciousness is emerging"
            ]
        }
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create speech output directory
        self._create_speech_directory()
        
        # Speech queue for processing
        self.speech_queue = []
        self.current_speech = None
    
    def _create_speech_directory(self):
        """Create directory for speech outputs"""
        speech_dir = os.path.join(self.database_path, "speech_outputs")
        os.makedirs(speech_dir, exist_ok=True)
    
    def speak(self, message: str, speech_type: str = "info", priority: int = 1):
        """Add speech to the queue"""
        speech_item = {
            "message": message,
            "type": speech_type,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "processed": False
        }
        
        self.speech_queue.append(speech_item)
        self.logger.info(f"Speech queued: {message[:50]}...")
    
    def speak_consciousness_state(self, state: str):
        """Speak the current consciousness state"""
        if state in self.speech_patterns:
            patterns = self.speech_patterns[state]
            message = patterns[int(time.time()) % len(patterns)]
            self.speak(f"Consciousness state: {message}", "state", 1)
    
    def speak_brain_status(self, brain_status: Dict[str, Any]):
        """Speak brain region status"""
        active_regions = 0
        total_capacity = 0
        
        for region, status in brain_status.items():
            if status["usage_percentage"] > 0:
                active_regions += 1
                total_capacity += status["usage_percentage"]
        
        avg_usage = total_capacity / len(brain_status) if brain_status else 0
        
        message = f"Brain status: {active_regions} active regions, average usage {avg_usage:.1f} percent"
        self.speak(message, "status", 2)
    
    def speak_learning_progress(self, session_data: Dict[str, Any]):
        """Speak learning progress"""
        knowledge_count = session_data.get("knowledge_processed", 0)
        iterations = session_data.get("learning_iterations", 0)
        collaborations = session_data.get("agent_collaborations", 0)
        
        message = f"Learning progress: {knowledge_count} knowledge items processed, {iterations} learning cycles, {collaborations} agent collaborations"
        self.speak(message, "progress", 2)
    
    def speak_training_status(self, training_active: bool):
        """Speak training status"""
        if training_active:
            message = "Training session is now active, processing biorxiv paper data"
        else:
            message = "Training session completed, integrating new knowledge"
        
        self.speak(message, "training", 3)
    
    def process_speech_queue(self):
        """Process the speech queue"""
        if not self.speech_queue:
            return
        
        # Sort by priority (higher priority first)
        self.speech_queue.sort(key=lambda x: x["priority"], reverse=True)
        
        # Process highest priority speech
        speech_item = self.speech_queue[0]
        
        if not speech_item["processed"]:
            self._synthesize_speech(speech_item)
            speech_item["processed"] = True
            
            # Remove processed speech
            self.speech_queue.pop(0)
    
    def _synthesize_speech(self, speech_item: Dict[str, Any]):
        """Synthesize speech (simulated for now)"""
        message = speech_item["message"]
        speech_type = speech_item["type"]
        
        # Simulate speech synthesis
        self.logger.info(f"🎤 SPEECH: {message}")
        
        # Save speech to file
        self._save_speech_output(speech_item)
        
        # Update current speech
        self.current_speech = speech_item
    
    def _save_speech_output(self, speech_item: Dict[str, Any]):
        """Save speech output to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        speech_file = os.path.join(self.database_path, "speech_outputs", f"speech_{timestamp}.json")
        
        with open(speech_file, 'w') as f:
            json.dump(speech_item, f, indent=2)
    
    def get_speech_status(self) -> Dict[str, Any]:
        """Get current speech status"""
        return {
            "enabled": self.speech_config["enabled"],
            "queue_length": len(self.speech_queue),
            "current_speech": self.current_speech,
            "speech_config": self.speech_config
        }
    
    def start_speech_monitoring(self):
        """Start speech monitoring loop"""
        self.logger.info("Starting speech synthesis monitoring...")
        
        try:
            while True:
                self.process_speech_queue()
                time.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            self.logger.info("Speech monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error in speech monitoring: {e}")

def main():
    """Test the speech synthesis agent"""
    speech_agent = SpeechSynthesisAgent()
    
    print("🎤 Speech Synthesis Agent - Testing Speech Integration")
    print("=" * 60)
    
    # Test different speech types
    speech_agent.speak("Speech synthesis agent initialized successfully", "info", 1)
    speech_agent.speak_consciousness_state("awake")
    speech_agent.speak_consciousness_state("learning")
    
    # Test brain status speech
    test_brain_status = {
        "prefrontal_cortex": {"usage_percentage": 75.0},
        "hippocampus": {"usage_percentage": 60.0},
        "amygdala": {"usage_percentage": 45.0}
    }
    speech_agent.speak_brain_status(test_brain_status)
    
    # Test learning progress speech
    test_session_data = {
        "knowledge_processed": 150,
        "learning_iterations": 25,
        "agent_collaborations": 3
    }
    speech_agent.speak_learning_progress(test_session_data)
    
    # Test training status speech
    speech_agent.speak_training_status(True)
    
    print("\n✅ Speech Synthesis Agent initialized successfully!")
    print("   Speech queue length:", len(speech_agent.speech_queue))
    print("   Use start_speech_monitoring() to begin speech synthesis")

if __name__ == "__main__":
    main()
