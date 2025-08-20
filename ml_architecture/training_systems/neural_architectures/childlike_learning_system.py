"""
Childlike Learning System for Llama Lexi
Mimics how a 5-year-old discovers new words and learns language
through natural curiosity, context exploration, and playful discovery
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CuriosityNode:
    """
    Represents a child's natural curiosity about a word or concept
    Mimics the "why?" questions and exploration patterns of young children
    """
    
    def __init__(self, word: str, context: str, curiosity_level: float = 1.0):
        self.word = word
        self.context = context
        self.curiosity_level = curiosity_level  # 0.0 to 1.0
        self.discovery_time = datetime.now()
        self.exploration_count = 0
        self.related_words = []
        self.questions_generated = []
        self.emotional_response = "excited"  # excited, confused, surprised, happy
        
        # Childlike learning patterns
        self.repetition_count = 0
        self.context_variations = []
        self.playful_experiments = []
        
    def explore(self) -> List[str]:
        """Generate childlike exploration questions"""
        questions = []
        
        # "What if..." questions (imagination)
        if "what if" not in [q.lower() for q in self.questions_generated]:
            questions.append(f"What if {self.word} was different?")
        
        # "Why does..." questions (curiosity)
        if "why" not in [q.lower() for q in self.questions_generated]:
            questions.append(f"Why does {self.word} work like that?")
        
        # "Can I..." questions (experimentation)
        if "can i" not in [q.lower() for q in self.questions_generated]:
            questions.append(f"Can I use {self.word} in a new way?")
        
        # "What happens when..." questions (cause and effect)
        if "what happens" not in [q.lower() for q in self.questions_generated]:
            questions.append(f"What happens when I change {self.word}?")
        
        # "Is it like..." questions (pattern recognition)
        if "is it like" not in [q.lower() for q in self.questions_generated]:
            questions.append(f"Is {self.word} like something else I know?")
        
        self.questions_generated.extend(questions)
        self.exploration_count += 1
        
        return questions
    
    def add_context_variation(self, new_context: str):
        """Add a new context where the word is used"""
        if new_context not in self.context_variations:
            self.context_variations.append(new_context)
            self.curiosity_level = min(1.0, self.curiosity_level + 0.1)
    
    def add_related_word(self, related_word: str, relationship: str):
        """Connect to other words (building vocabulary network)"""
        if related_word not in [w[0] for w in self.curiosity_nodes]:
            self.related_words.append((related_word, relationship))
            self.curiosity_level = min(1.0, self.curiosity_level + 0.05)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "word": self.word,
            "context": self.context,
            "curiosity_level": self.curiosity_level,
            "discovery_time": self.discovery_time.isoformat(),
            "exploration_count": self.exploration_count,
            "related_words": self.related_words,
            "questions_generated": self.questions_generated,
            "emotional_response": self.emotional_response,
            "repetition_count": self.repetition_count,
            "context_variations": self.context_variations,
            "playful_experiments": self.playful_experiments
        }


class ChildlikeLearningSystem:
    """
    Main system that mimics how a 5-year-old learns language
    Focuses on curiosity-driven discovery rather than formal training
    """
    
    def __init__(self, learning_data_path: str = "childlike_learning_data"):
        self.learning_data_path = Path(learning_data_path)
        self.learning_data_path.mkdir(exist_ok=True)
        
        # Learning state
        self.curiosity_nodes = {}  # word -> CuriosityNode
        self.learning_sessions = []
        self.vocabulary_network = {}
        self.emotional_memory = {}
        
        # Childlike learning parameters
        self.attention_span = 15  # minutes (typical for 5-year-old)
        self.curiosity_decay = 0.1  # how quickly curiosity fades
        self.exploration_boost = 0.2  # how much exploration increases curiosity
        self.context_sensitivity = 0.8  # how much context affects learning
        
        # Learning patterns
        self.repetition_threshold = 3  # how many times before "mastered"
        self.context_variation_threshold = 2  # how many contexts before "understood"
        self.curiosity_threshold = 0.3  # minimum curiosity to continue exploring
        
        # Load existing learning data
        self._load_learning_data()
        
        logger.info("Childlike Learning System initialized")
    
    def _load_learning_data(self):
        """Load existing learning data from files"""
        try:
            # Load curiosity nodes
            nodes_file = self.learning_data_path / "curiosity_nodes.json"
            if nodes_file.exists():
                with open(nodes_file, 'r') as f:
                    data = json.load(f)
                    for word, node_data in data.items():
                        node = CuriosityNode(
                            node_data["word"], 
                            node_data["context"], 
                            node_data["curiosity_level"]
                        )
                        # Restore node state
                        node.exploration_count = node_data["exploration_count"]
                        node.related_words = node_data["related_words"]
                        node.questions_generated = node_data["questions_generated"]
                        node.emotional_response = node_data["emotional_response"]
                        node.repetition_count = node_data["repetition_count"]
                        node.context_variations = node_data["context_variations"]
                        node.playful_experiments = node_data["playful_experiments"]
                        
                        self.curiosity_nodes[word] = node
            
            # Load vocabulary network
            network_file = self.learning_data_path / "vocabulary_network.json"
            if network_file.exists():
                with open(network_file, 'r') as f:
                    self.vocabulary_network = json.load(f)
            
            # Load emotional memory
            emotion_file = self.learning_data_path / "emotional_memory.json"
            if emotion_file.exists():
                with open(emotion_file, 'r') as f:
                    self.emotional_memory = json.load(f)
                    
        except Exception as e:
            logger.warning(f"Could not load existing learning data: {e}")
    
    def _save_learning_data(self):
        """Save current learning data to files"""
        try:
            # Save curiosity nodes
            nodes_data = {word: node.to_dict() for word, node in self.curiosity_nodes.items()}
            with open(self.learning_data_path / "curiosity_nodes.json", 'w') as f:
                json.dump(nodes_data, f, indent=2)
            
            # Save vocabulary network
            with open(self.learning_data_path / "vocabulary_network.json", 'w') as f:
                json.dump(self.vocabulary_network, f, indent=2)
            
            # Save emotional memory
            with open(self.learning_data_path / "emotional_memory.json", 'w') as f:
                json.dump(self.emotional_memory, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def discover_new_word(self, word: str, context: str, emotional_context: str = "neutral") -> CuriosityNode:
        """
        Discover a new word (like a child hearing it for the first time)
        
        Args:
            word: The new word discovered
            context: How/where the word was encountered
            emotional_context: Emotional context of discovery
            
        Returns:
            CuriosityNode for the new word
        """
        if word.lower() in self.curiosity_nodes:
            # Word already known - increase curiosity about it
            existing_node = self.curiosity_nodes[word.lower()]
            existing_node.curiosity_level = min(1.0, existing_node.curiosity_level + 0.2)
            existing_node.repetition_count += 1
            existing_node.add_context_variation(context)
            return existing_node
        
        # Create new curiosity node
        curiosity_level = self._calculate_initial_curiosity(word, context, emotional_context)
        node = CuriosityNode(word, context, curiosity_level)
        
        # Set emotional response based on context
        emotional_responses = {
            "happy": ["excited", "happy", "surprised"],
            "sad": ["confused", "curious", "concerned"],
            "scary": ["nervous", "curious", "brave"],
            "funny": ["excited", "happy", "surprised"],
            "neutral": ["curious", "interested", "neutral"]
        }
        node.emotional_response = random.choice(emotional_responses.get(emotional_context, ["curious"]))
        
        # Add to vocabulary network
        self.vocabulary_network[word.lower()] = {
            "discovery_time": datetime.now().isoformat(),
            "contexts": [context],
            "emotional_contexts": [emotional_context],
            "related_words": [],
            "learning_progress": 0.0
        }
        
        # Store the node
        self.curiosity_nodes[word.lower()] = node
        
        # Generate initial exploration questions
        initial_questions = node.explore()
        
        logger.info(f"🎉 Discovered new word: '{word}' in context: '{context}'")
        logger.info(f"   Emotional response: {node.emotional_response}")
        logger.info(f"   Initial curiosity: {curiosity_level:.2f}")
        logger.info(f"   Exploration questions: {len(initial_questions)}")
        
        return node
    
    def _calculate_initial_curiosity(self, word: str, context: str, emotional_context: str) -> float:
        """Calculate initial curiosity level based on various factors"""
        base_curiosity = 0.5
        
        # Word complexity (shorter words are more interesting to children)
        length_factor = max(0.1, 1.0 - (len(word) - 3) * 0.1)
        
        # Context novelty (new contexts are more interesting)
        context_novelty = 1.0
        for existing_context in self.vocabulary_network.values():
            if context in existing_context.get("contexts", []):
                context_novelty = 0.5
                break
        
        # Emotional intensity
        emotional_factors = {
            "happy": 1.2, "sad": 0.8, "scary": 1.3, "funny": 1.4, "neutral": 1.0
        }
        emotional_boost = emotional_factors.get(emotional_context, 1.0)
        
        # Random variation (children are unpredictable)
        random_factor = random.uniform(0.8, 1.2)
        
        curiosity = base_curiosity * length_factor * context_novelty * emotional_boost * random_factor
        return min(1.0, max(0.1, curiosity))
    
    def explore_word(self, word: str, new_context: str = None) -> List[str]:
        """
        Explore a word through childlike curiosity
        Generates questions and connections like a 5-year-old would
        """
        word_lower = word.lower()
        if word_lower not in self.curiosity_nodes:
            # Discover the word if not known
            node = self.discover_new_word(word, new_context or "exploration")
        else:
            node = self.curiosity_nodes[word_lower]
        
        # Add new context if provided
        if new_context:
            node.add_context_variation(new_context)
        
        # Generate exploration questions
        questions = node.explore()
        
        # Update curiosity level
        node.curiosity_level = min(1.0, node.curiosity_level + self.exploration_boost)
        
        # Find related words through context
        related_words = self._find_contextual_relationships(word_lower, new_context)
        for related_word, relationship in related_words:
            node.add_related_word(related_word, relationship)
        
        # Update learning progress
        self._update_learning_progress(word_lower)
        
        return questions
    
    def _find_contextual_relationships(self, word: str, context: str) -> List[Tuple[str, str]]:
        """Find words that are contextually related (like a child making connections)"""
        relationships = []
        
        # Look for words in similar contexts
        for other_word, word_data in self.vocabulary_network.items():
            if other_word == word:
                continue
            
            other_contexts = word_data.get("contexts", [])
            for other_context in other_contexts:
                # Simple context similarity (children make simple connections)
                if any(word in other_context.lower() or other_word in context.lower()):
                    relationship = "context_similarity"
                    relationships.append((other_word, relationship))
                    break
        
        # Look for emotional connections
        word_emotions = self.vocabulary_network.get(word, {}).get("emotional_contexts", [])
        for other_word, word_data in self.vocabulary_network.items():
            if other_word == word:
                continue
            
            other_emotions = word_data.get("emotional_contexts", [])
            if any(emotion in other_emotions for emotion in word_emotions):
                relationship = "emotional_connection"
                relationships.append((other_word, relationship))
        
        return relationships[:3]  # Limit to 3 connections (like a child's attention span)
    
    def _update_learning_progress(self, word: str):
        """Update learning progress based on exploration and context variety"""
        if word not in self.vocabulary_network:
            return
        
        node = self.curiosity_nodes.get(word)
        if not node:
            return
        
        # Calculate progress based on multiple factors
        context_progress = min(1.0, len(node.context_variations) / self.context_variation_threshold)
        repetition_progress = min(1.0, node.repetition_count / self.repetition_threshold)
        curiosity_progress = node.curiosity_level
        exploration_progress = min(1.0, node.exploration_count / 5)
        
        # Weighted average (children learn through repetition and context)
        total_progress = (
            context_progress * 0.3 +
            repetition_progress * 0.3 +
            curiosity_progress * 0.2 +
            exploration_progress * 0.2
        )
        
        self.vocabulary_network[word]["learning_progress"] = total_progress
        
        # Update emotional memory
        if word not in self.emotional_memory:
            self.emotional_memory[word] = []
        
        self.emotional_memory[word].append({
            "timestamp": datetime.now().isoformat(),
            "progress": total_progress,
            "emotional_response": node.emotional_response,
            "contexts_explored": len(node.context_variations)
        })
    
    def playful_experiment(self, word: str, experiment_type: str = "random") -> Dict[str, Any]:
        """
        Conduct a playful experiment with a word (like a child playing with language)
        
        Args:
            word: Word to experiment with
            experiment_type: Type of experiment
            
        Returns:
            Experiment results
        """
        word_lower = word.lower()
        if word_lower not in self.curiosity_nodes:
            node = self.discover_new_word(word, "playful_experiment")
        else:
            node = self.curiosity_nodes[word_lower]
        
        # Choose experiment type
        if experiment_type == "random":
            experiment_types = ["rhyme", "syllable_play", "context_mixing", "emotional_exploration"]
            experiment_type = random.choice(experiment_types)
        
        experiment_result = {
            "word": word,
            "experiment_type": experiment_type,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "discoveries": []
        }
        
        if experiment_type == "rhyme":
            # Find rhyming words (children love rhymes)
            rhymes = self._find_rhyming_words(word)
            experiment_result["discoveries"] = rhymes
            
        elif experiment_type == "syllable_play":
            # Play with syllables (children experiment with word structure)
            syllables = self._analyze_syllables(word)
            experiment_result["discoveries"] = syllables
            
        elif experiment_type == "context_mixing":
            # Mix contexts (children apply words in new situations)
            new_contexts = self._generate_context_variations(word)
            experiment_result["discoveries"] = new_contexts
            
        elif experiment_type == "emotional_exploration":
            # Explore emotional variations (children understand emotional context)
            emotions = self._explore_emotional_variations(word)
            experiment_result["discoveries"] = emotions
        
        # Record the experiment
        node.playful_experiments.append(experiment_result)
        
        # Boost curiosity through play
        node.curiosity_level = min(1.0, node.curiosity_level + 0.15)
        
        logger.info(f"🎮 Playful experiment with '{word}': {experiment_type}")
        logger.info(f"   Discoveries: {len(experiment_result['discoveries'])}")
        
        return experiment_result
    
    def _find_rhyming_words(self, word: str) -> List[str]:
        """Find words that rhyme (simple rhyming for children)"""
        # Simple rhyming logic (children make simple connections)
        rhyming_patterns = {
            "at": ["cat", "bat", "hat", "rat", "sat"],
            "ing": ["ring", "sing", "wing", "thing", "bring"],
            "ight": ["light", "bright", "night", "right", "fight"],
            "ake": ["make", "take", "cake", "lake", "snake"]
        }
        
        rhymes = []
        for pattern, rhyming_words in rhyming_patterns.items():
            if word.lower().endswith(pattern):
                rhymes.extend(rhyming_words)
                break
        
        return rhymes[:3]  # Limit to 3 rhymes
    
    def _analyze_syllables(self, word: str) -> Dict[str, Any]:
        """Analyze word syllables (children notice word structure)"""
        # Simple syllable counting (approximate)
        vowels = "aeiouy"
        syllable_count = 0
        prev_vowel = False
        
        for char in word.lower():
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                syllable_count += 1
            prev_vowel = is_vowel
        
        return {
            "syllable_count": syllable_count,
            "word_length": len(word),
            "vowel_count": sum(1 for c in word.lower() if c in vowels),
            "consonant_count": len(word) - sum(1 for c in word.lower() if c in vowels)
        }
    
    def _generate_context_variations(self, word: str) -> List[str]:
        """Generate new contexts for a word (children apply words broadly)"""
        base_contexts = [
            f"Using {word} in a story",
            f"Using {word} in a game",
            f"Using {word} when talking to friends",
            f"Using {word} when describing feelings",
            f"Using {word} in a song"
        ]
        
        # Add some randomness (children are creative)
        random_contexts = [
            f"Using {word} in space",
            f"Using {word} underwater",
            f"Using {word} in a magical forest",
            f"Using {word} when dreaming"
        ]
        
        all_contexts = base_contexts + random_contexts
        return random.sample(all_contexts, min(3, len(all_contexts)))
    
    def _explore_emotional_variations(self, word: str) -> List[Dict[str, str]]:
        """Explore how a word can be used in different emotional contexts"""
        emotions = ["happy", "sad", "excited", "scared", "surprised", "angry", "loving"]
        
        variations = []
        for emotion in random.sample(emotions, 3):
            variations.append({
                "emotion": emotion,
                "usage": f"Using '{word}' when feeling {emotion}",
                "example": f"I feel {emotion} about {word}"
            })
        
        return variations
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get a comprehensive report of the childlike learning progress"""
        total_words = len(self.curiosity_nodes)
        mastered_words = sum(1 for word_data in self.vocabulary_network.values() 
                           if word_data.get("learning_progress", 0) >= 0.8)
        
        # Calculate average curiosity
        avg_curiosity = np.mean([node.curiosity_level for node in self.curiosity_nodes.values()]) if self.curiosity_nodes else 0
        
        # Find most curious words
        most_curious = sorted(
            [(word, node.curiosity_level) for word, node in self.curiosity_nodes.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        # Find most explored words
        most_explored = sorted(
            [(word, node.exploration_count) for word, node in self.curiosity_nodes.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        # Learning session summary
        session_summary = {
            "total_sessions": len(self.learning_sessions),
            "last_session": self.learning_sessions[-1] if self.learning_sessions else None,
            "average_session_length": np.mean([s.get("duration", 0) for s in self.learning_sessions]) if self.learning_sessions else 0
        }
        
        return {
            "learning_summary": {
                "total_words_discovered": total_words,
                "mastered_words": mastered_words,
                "learning_progress": mastered_words / total_words if total_words > 0 else 0,
                "average_curiosity": avg_curiosity
            },
            "top_curious_words": most_curious,
            "top_explored_words": most_explored,
            "session_summary": session_summary,
            "vocabulary_network_size": len(self.vocabulary_network),
            "emotional_memory_size": len(self.emotional_memory)
        }
    
    def start_learning_session(self, duration_minutes: int = None) -> str:
        """Start a new learning session (like a child's play session)"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        duration = duration_minutes or self.attention_span
        
        session = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "duration_minutes": duration,
            "words_explored": [],
            "experiments_conducted": [],
            "curiosity_boosts": 0
        }
        
        self.learning_sessions.append(session)
        logger.info(f"🎯 Started learning session: {session_id} (duration: {duration} minutes)")
        
        return session_id
    
    def end_learning_session(self, session_id: str):
        """End a learning session and save progress"""
        session = None
        for s in self.learning_sessions:
            if s["session_id"] == session_id:
                session = s
                break
        
        if session:
            session["end_time"] = datetime.now().isoformat()
            session["actual_duration"] = (
                datetime.fromisoformat(session["end_time"]) - 
                datetime.fromisoformat(session["start_time"])
            ).total_seconds() / 60
            
            logger.info(f"✅ Ended learning session: {session_id}")
            logger.info(f"   Words explored: {len(session['words_explored'])}")
            logger.info(f"   Experiments: {len(session['experiments_conducted'])}")
            logger.info(f"   Duration: {session['actual_duration']:.1f} minutes")
        
        # Save learning data
        self._save_learning_data()
    
    def cleanup_old_memories(self, days_to_keep: int = 30):
        """Clean up old emotional memories (like a child forgetting old details)"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for word in list(self.emotional_memory.keys()):
            memories = self.emotional_memory[word]
            # Keep only recent memories
            self.emotional_memory[word] = [
                memory for memory in memories
                if datetime.fromisoformat(memory["timestamp"]) > cutoff_date
            ]
            
            # Remove word if no memories left
            if not self.emotional_memory[word]:
                del self.emotional_memory[word]
        
        logger.info(f"🧹 Cleaned up memories older than {days_to_keep} days")


# Convenience functions
def create_childlike_learner(data_path: str = "childlike_learning_data") -> ChildlikeLearningSystem:
    """Create a new childlike learning system"""
    return ChildlikeLearningSystem(data_path)


def get_global_childlike_learner() -> ChildlikeLearningSystem:
    """Get the global childlike learning system instance"""
    if not hasattr(get_global_childlike_learner, '_instance'):
        get_global_childlike_learner._instance = ChildlikeLearningSystem()
    return get_global_childlike_learner._instance


if __name__ == "__main__":
    # Example usage
    learner = create_childlike_learner()
    
    # Start a learning session
    session_id = learner.start_learning_session(20)
    
    # Discover new words
    learner.discover_new_word("butterfly", "in the garden", "happy")
    learner.discover_new_word("thunder", "during a storm", "scary")
    learner.discover_new_word("rainbow", "after the rain", "excited")
    
    # Explore words
    learner.explore_word("butterfly", "flying in the sky")
    learner.explore_word("thunder", "making loud noises")
    
    # Conduct playful experiments
    learner.playful_experiment("butterfly", "rhyme")
    learner.playful_experiment("rainbow", "emotional_exploration")
    
    # End session
    learner.end_learning_session(session_id)
    
    # Get learning report
    report = learner.get_learning_report()
    print("Learning Report:", json.dumps(report, indent=2))
