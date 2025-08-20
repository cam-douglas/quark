"""
Childlike Learning System for Llama Lexi
Mimics how a 5-year-old learns language through curiosity and discovery
"""

import json
import random
from datetime import datetime
from typing import Dict, List, Any

class ChildLearner:
    """Simulates how a 5-year-old learns new words and language"""
    
    def __init__(self):
        self.known_words = {}
        self.curiosity_levels = {}
        self.contexts = {}
        self.emotional_memories = {}
        self.learning_sessions = []
        
        # Childlike learning parameters
        self.attention_span = 15  # minutes
        self.curiosity_boost = 0.2
        self.repetition_threshold = 3
        
    def discover_word(self, word: str, context: str, emotion: str = "curious"):
        """Discover a new word like a child would"""
        if word not in self.known_words:
            # New word discovery
            self.known_words[word] = {
                "first_heard": datetime.now().isoformat(),
                "times_heard": 1,
                "contexts": [context],
                "emotions": [emotion],
                "curiosity": 0.8,  # Children are naturally curious
                "questions_asked": [],
                "related_words": []
            }
            print(f"🎉 New word discovered: '{word}' in context: '{context}'")
        else:
            # Word already known - increase familiarity
            self.known_words[word]["times_heard"] += 1
            if context not in self.known_words[word]["contexts"]:
                self.known_words[word]["contexts"].append(context)
            if emotion not in self.known_words[word]["emotions"]:
                self.known_words[word]["emotions"].append(emotion)
            
            # Boost curiosity through repetition
            self.known_words[word]["curiosity"] = min(1.0, 
                self.known_words[word]["curiosity"] + 0.1)
    
    def ask_question(self, word: str) -> str:
        """Generate a childlike question about a word"""
        if word not in self.known_words:
            return f"What is '{word}'?"
        
        word_data = self.known_words[word]
        questions = [
            f"Why does '{word}' work like that?",
            f"What if '{word}' was different?",
            f"Can I use '{word}' in a new way?",
            f"Is '{word}' like something else I know?",
            f"What happens when I change '{word}'?"
        ]
        
        # Add context-specific questions
        for context in word_data["contexts"]:
            questions.append(f"How does '{word}' work in '{context}'?")
        
        # Choose question based on curiosity level
        if word_data["curiosity"] > 0.7:
            question = random.choice(questions[:3])  # More imaginative questions
        else:
            question = random.choice(questions[3:])  # More practical questions
        
        word_data["questions_asked"].append(question)
        return question
    
    def explore_context(self, word: str, new_context: str):
        """Explore a word in a new context"""
        if word not in self.known_words:
            self.discover_word(word, new_context)
            return
        
        word_data = self.known_words[word]
        if new_context not in word_data["contexts"]:
            word_data["contexts"].append(new_context)
            word_data["curiosity"] = min(1.0, word_data["curiosity"] + 0.15)
            print(f"🔍 Explored '{word}' in new context: '{new_context}'")
    
    def find_connections(self, word: str) -> List[str]:
        """Find connections between words (like children do)"""
        connections = []
        for other_word, other_data in self.known_words.items():
            if other_word == word:
                continue
            
            # Context similarity
            for context in self.known_words[word]["contexts"]:
                if any(context.lower() in other_context.lower() 
                       for other_context in other_data["contexts"]):
                    connections.append(f"'{word}' and '{other_word}' are used in similar situations")
                    break
            
            # Emotional similarity
            for emotion in self.known_words[word]["emotions"]:
                if emotion in other_data["emotions"]:
                    connections.append(f"'{word}' and '{other_word}' make me feel {emotion}")
                    break
        
        return connections[:3]  # Limit connections (child attention span)
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status"""
        total_words = len(self.known_words)
        mastered_words = sum(1 for word_data in self.known_words.values() 
                           if word_data["times_heard"] >= self.repetition_threshold)
        
        return {
            "total_words": total_words,
            "mastered_words": mastered_words,
            "learning_progress": mastered_words / total_words if total_words > 0 else 0,
            "most_curious_words": sorted(
                [(word, data["curiosity"]) for word, data in self.known_words.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }
    
    def save_progress(self, filename: str = "child_learning_progress.json"):
        """Save learning progress"""
        data = {
            "known_words": self.known_words,
            "learning_sessions": self.learning_sessions,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Learning progress saved to {filename}")


# Example usage
if __name__ == "__main__":
    learner = ChildLearner()
    
    # Discover words like a child
    learner.discover_word("butterfly", "in the garden", "excited")
    learner.discover_word("thunder", "during storm", "scared")
    learner.discover_word("rainbow", "after rain", "happy")
    
    # Ask questions
    print(learner.ask_question("butterfly"))
    print(learner.ask_question("thunder"))
    
    # Explore contexts
    learner.explore_context("butterfly", "flying in the sky")
    learner.explore_context("rainbow", "in a storybook")
    
    # Find connections
    connections = learner.find_connections("butterfly")
    for connection in connections:
        print(f"🔗 {connection}")
    
    # Get status
    status = learner.get_learning_status()
    print(f"📊 Learning Status: {status}")
    
    # Save progress
    learner.save_progress()
