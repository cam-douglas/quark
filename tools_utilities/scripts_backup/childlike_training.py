"""
Childlike Training for Llama Lexi
Trains the model using childlike learning patterns instead of standard training
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from child_learner import ChildLearner

class ChildlikeTrainer:
    """
    Trains Llama Lexi using childlike learning patterns:
    - Natural curiosity and discovery
    - Context exploration and play
    - Emotional connection to words
    - Repetition through natural usage
    """
    
    def __init__(self, model_path: str = "Orenguteng/Llama-3-8B-Lexi-Uncensored"):
        self.model_path = model_path
        self.child_learner = ChildLearner()
        self.training_data = []
        self.learning_scenarios = self._create_learning_scenarios()
        
        print("🧒 Childlike Training System Initialized")
        print("   Learning through curiosity, play, and discovery")
    
    def _create_learning_scenarios(self) -> List[Dict[str, Any]]:
        """Create realistic learning scenarios for a 5-year-old"""
        return [
            # Nature and Discovery
            {
                "theme": "nature_discovery",
                "scenarios": [
                    {"context": "garden exploration", "emotion": "excited", "words": ["butterfly", "flower", "bee", "soil"]},
                    {"context": "rainy day", "emotion": "curious", "words": ["rain", "puddle", "umbrella", "cloud"]},
                    {"context": "sunny day", "emotion": "happy", "words": ["sun", "shadow", "warm", "bright"]}
                ]
            },
            # Family and Home
            {
                "theme": "family_home",
                "scenarios": [
                    {"context": "cooking together", "emotion": "excited", "words": ["mix", "stir", "taste", "hot"]},
                    {"context": "bedtime story", "emotion": "calm", "words": ["dream", "sleep", "story", "night"]},
                    {"context": "morning routine", "emotion": "energetic", "words": ["wake", "brush", "eat", "ready"]}
                ]
            },
            # Play and Imagination
            {
                "theme": "play_imagination",
                "scenarios": [
                    {"context": "building blocks", "emotion": "creative", "words": ["build", "tower", "fall", "stack"]},
                    {"context": "pretend play", "emotion": "imaginative", "words": ["pretend", "magic", "transform", "adventure"]},
                    {"context": "drawing", "emotion": "artistic", "words": ["draw", "color", "line", "shape"]}
                ]
            },
            # Emotions and Feelings
            {
                "theme": "emotions_feelings",
                "scenarios": [
                    {"context": "feeling happy", "emotion": "joyful", "words": ["happy", "smile", "laugh", "fun"]},
                    {"context": "feeling sad", "emotion": "concerned", "words": ["sad", "cry", "comfort", "hug"]},
                    {"context": "feeling scared", "emotion": "nervous", "words": ["scared", "safe", "protect", "brave"]}
                ]
            },
            # Learning and School
            {
                "theme": "learning_school",
                "scenarios": [
                    {"context": "counting numbers", "emotion": "proud", "words": ["one", "two", "three", "many"]},
                    {"context": "learning letters", "emotion": "excited", "words": ["letter", "sound", "word", "read"]},
                    {"context": "sharing with friends", "emotion": "friendly", "words": ["share", "friend", "together", "help"]}
                ]
            }
        ]
    
    def start_learning_session(self, duration_minutes: int = 20):
        """Start a childlike learning session"""
        print(f"\n🎯 Starting Learning Session ({duration_minutes} minutes)")
        print("   Let's discover new words through play and curiosity!")
        
        session_words = []
        session_scenarios = random.sample(self.learning_scenarios, 3)
        
        for scenario_group in session_scenarios:
            print(f"\n🌍 Theme: {scenario_group['theme'].replace('_', ' ').title()}")
            
            for scenario in scenario_group['scenarios']:
                print(f"   📖 Context: {scenario['context']}")
                print(f"   😊 Emotion: {scenario['emotion']}")
                
                # Discover words in this scenario
                for word in scenario['words']:
                    self.child_learner.discover_word(word, scenario['context'], scenario['emotion'])
                    session_words.append(word)
                    
                    # Ask childlike questions
                    question = self.child_learner.ask_question(word)
                    print(f"      🤔 {question}")
                    
                    # Explore contexts
                    new_contexts = self._generate_context_variations(word, scenario['context'])
                    for new_context in new_contexts[:2]:  # Limit to 2 new contexts
                        self.child_learner.explore_context(word, new_context)
                
                print()  # Spacing
        
        # Find connections between words
        print("🔗 Discovering Connections Between Words...")
        for word in random.sample(session_words, min(3, len(session_words))):
            connections = self.child_learner.find_connections(word)
            if connections:
                print(f"   '{word}' connections:")
                for connection in connections:
                    print(f"      {connection}")
        
        # Save session progress
        self.child_learner.save_progress()
        
        # Get learning status
        status = self.child_learner.get_learning_status()
        print(f"\n📊 Session Complete!")
        print(f"   Words discovered: {len(session_words)}")
        print(f"   Total vocabulary: {status['total_words']}")
        print(f"   Learning progress: {status['learning_progress']:.1%}")
        
        return session_words
    
    def _generate_context_variations(self, word: str, base_context: str) -> List[str]:
        """Generate new contexts for a word (children apply words broadly)"""
        variations = []
        
        # Time variations
        time_variations = ["in the morning", "at night", "during the day", "on weekends"]
        for time_var in time_variations:
            variations.append(f"{word} {time_var}")
        
        # Place variations
        place_variations = ["at home", "at school", "in the park", "at the store"]
        for place_var in place_variations:
            variations.append(f"{word} {place_var}")
        
        # Activity variations
        activity_variations = ["while playing", "while eating", "while sleeping", "while reading"]
        for activity_var in activity_variations:
            variations.append(f"{word} {activity_var}")
        
        # Emotional variations
        emotional_variations = ["when happy", "when sad", "when excited", "when scared"]
        for emotion_var in emotional_variations:
            variations.append(f"{word} {emotion_var}")
        
        # Randomly select variations
        return random.sample(variations, min(4, len(variations)))
    
    def interactive_learning(self):
        """Interactive learning mode - let user guide the learning"""
        print("\n🎮 Interactive Learning Mode")
        print("   You guide the learning process like a parent or teacher")
        print("   Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                user_input = input("\n🧒 What would you like to explore? ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'stop']:
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'status':
                    self._show_status()
                elif user_input.lower() == 'scenarios':
                    self._show_scenarios()
                elif user_input.lower() == 'words':
                    self._show_words()
                elif user_input.lower().startswith('discover '):
                    word = user_input[9:].strip()
                    context = input(f"   In what context did you discover '{word}'? ").strip()
                    emotion = input(f"   How did it make you feel? ").strip()
                    self.child_learner.discover_word(word, context, emotion)
                elif user_input.lower().startswith('ask '):
                    word = user_input[4:].strip()
                    question = self.child_learner.ask_question(word)
                    print(f"   🤔 {question}")
                elif user_input.lower().startswith('explore '):
                    word = user_input[8:].strip()
                    new_context = input(f"   What new context would you like to explore for '{word}'? ").strip()
                    self.child_learner.explore_context(word, new_context)
                elif user_input.lower().startswith('connect '):
                    word = user_input[8:].strip()
                    connections = self.child_learner.find_connections(word)
                    if connections:
                        print(f"   🔗 Connections for '{word}':")
                        for connection in connections:
                            print(f"      {connection}")
                    else:
                        print(f"   No connections found for '{word}' yet")
                else:
                    print("   I don't understand. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Learning session interrupted. Saving progress...")
                break
            except Exception as e:
                print(f"   Error: {e}")
        
        # Save progress before exiting
        self.child_learner.save_progress()
        print("💾 Progress saved. Goodbye!")
    
    def _show_help(self):
        """Show available commands"""
        print("\n📚 Available Commands:")
        print("   help - Show this help")
        print("   status - Show learning progress")
        print("   scenarios - Show available learning scenarios")
        print("   words - Show discovered words")
        print("   discover [word] - Discover a new word")
        print("   ask [word] - Ask a question about a word")
        print("   explore [word] - Explore a word in new contexts")
        print("   connect [word] - Find connections for a word")
        print("   quit - Exit learning session")
    
    def _show_status(self):
        """Show current learning status"""
        status = self.child_learner.get_learning_status()
        print(f"\n📊 Learning Status:")
        print(f"   Total words: {status['total_words']}")
        print(f"   Mastered words: {status['mastered_words']}")
        print(f"   Learning progress: {status['learning_progress']:.1%}")
        print(f"   Most curious words:")
        for word, curiosity in status['most_curious_words']:
            print(f"      '{word}': {curiosity:.2f}")
    
    def _show_scenarios(self):
        """Show available learning scenarios"""
        print("\n🌍 Available Learning Scenarios:")
        for i, scenario_group in enumerate(self.learning_scenarios, 1):
            print(f"   {i}. {scenario_group['theme'].replace('_', ' ').title()}")
            for scenario in scenario_group['scenarios']:
                print(f"      • {scenario['context']} ({scenario['emotion']})")
    
    def _show_words(self):
        """Show discovered words"""
        if not self.child_learner.known_words:
            print("\n📝 No words discovered yet!")
            return
        
        print(f"\n📝 Discovered Words ({len(self.child_learner.known_words)}):")
        for word, data in self.child_learner.known_words.items():
            contexts = ", ".join(data['contexts'][:2])  # Show first 2 contexts
            if len(data['contexts']) > 2:
                contexts += f" (+{len(data['contexts'])-2} more)"
            print(f"   '{word}' - {contexts}")
            print(f"      Curiosity: {data['curiosity']:.2f}, Times heard: {data['times_heard']}")


def main():
    """Main function to run childlike training"""
    print("🧒 Welcome to Childlike Training for Llama Lexi!")
    print("   Learning language the way a 5-year-old does")
    print("   Through curiosity, play, and natural discovery")
    
    trainer = ChildlikeTrainer()
    
    while True:
        print("\n🎯 Choose Learning Mode:")
        print("   1. Guided Learning Session (20 minutes)")
        print("   2. Interactive Learning (you guide)")
        print("   3. View Learning Status")
        print("   4. Exit")
        
        choice = input("\n   Your choice (1-4): ").strip()
        
        if choice == "1":
            trainer.start_learning_session()
        elif choice == "2":
            trainer.interactive_learning()
        elif choice == "3":
            trainer._show_status()
        elif choice == "4":
            print("👋 Thank you for learning with us!")
            break
        else:
            print("   Please choose 1, 2, 3, or 4")


if __name__ == "__main__":
    main()
