"""
Curiosity Engine - Core curiosity-driven learning system
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class CuriosityState:
    """Represents the current state of curiosity and interest"""
    current_interests: List[str]
    question_queue: List[str]
    uncertainty_levels: Dict[str, float]
    exploration_history: List[Dict[str, Any]]
    surprise_threshold: float = 0.7
    interest_decay: float = 0.95

class CuriosityEngine:
    """
    Core engine for curiosity-driven learning.
    
    Implements intrinsic motivation through:
    - Novelty detection
    - Uncertainty quantification  
    - Question generation
    - Interest prioritization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.state = CuriosityState(
            current_interests=[],
            question_queue=[],
            uncertainty_levels={},
            exploration_history=[]
        )
        self.logger = logging.getLogger(__name__)
        
        # Curiosity parameters
        self.novelty_weight = self.config.get('novelty_weight', 0.4)
        self.uncertainty_weight = self.config.get('uncertainty_weight', 0.3)
        self.potential_weight = self.config.get('potential_weight', 0.3)
        
    def assess_curiosity(self, observation: Any, context: Dict = None) -> float:
        """
        Assess curiosity level for a given observation.
        
        Returns a curiosity score [0, 1] indicating how interesting
        this observation is for learning.
        """
        context = context or {}
        
        novelty_score = self._calculate_novelty(observation, context)
        uncertainty_score = self._calculate_uncertainty(observation, context)
        potential_score = self._calculate_learning_potential(observation, context)
        
        curiosity_score = (
            self.novelty_weight * novelty_score +
            self.uncertainty_weight * uncertainty_score +
            self.potential_weight * potential_score
        )
        
        self.logger.debug(f"Curiosity assessment: {curiosity_score:.3f} "
                         f"(novelty: {novelty_score:.3f}, "
                         f"uncertainty: {uncertainty_score:.3f}, "
                         f"potential: {potential_score:.3f})")
        
        return curiosity_score
        
    def generate_questions(self, observation: Any, num_questions: int = 3) -> List[str]:
        """Generate questions about an observation to drive exploration."""
        questions = []
        
        # Template-based question generation
        question_templates = [
            "What would happen if {}?",
            "Why does {} occur?", 
            "How is {} related to {}?",
            "What are the consequences of {}?",
            "What patterns exist in {}?",
            "How can {} be improved or optimized?",
            "What causes {} to change?",
            "What are alternative explanations for {}?"
        ]
        
        # Simple question generation (can be enhanced with LLM)
        observation_str = str(observation)[:100]  # Truncate for template
        
        for template in question_templates[:num_questions]:
            if '{}' in template:
                if template.count('{}') == 1:
                    question = template.format(observation_str)
                else:
                    # For templates with multiple placeholders
                    question = template.format(observation_str, "related concepts")
                questions.append(question)
        
        self.state.question_queue.extend(questions)
        return questions
        
    def update_interest(self, topic: str, feedback: float):
        """Update interest level for a topic based on learning feedback."""
        current_interest = self.state.uncertainty_levels.get(topic, 0.5)
        
        # Update with exponential moving average
        alpha = 0.3  # Learning rate
        new_interest = alpha * feedback + (1 - alpha) * current_interest
        
        self.state.uncertainty_levels[topic] = new_interest
        
        # Add to current interests if above threshold
        if new_interest > 0.6 and topic not in self.state.current_interests:
            self.state.current_interests.append(topic)
        elif new_interest <= 0.4 and topic in self.state.current_interests:
            self.state.current_interests.remove(topic)
            
    def get_next_exploration_target(self) -> Optional[str]:
        """Get the next most interesting target for exploration."""
        if not self.state.current_interests:
            return None
            
        # Sort by uncertainty/interest level
        sorted_interests = sorted(
            self.state.current_interests,
            key=lambda topic: self.state.uncertainty_levels.get(topic, 0),
            reverse=True
        )
        
        return sorted_interests[0] if sorted_interests else None
        
    def record_exploration(self, target: str, outcome: Dict[str, Any]):
        """Record the outcome of an exploration for learning."""
        exploration_record = {
            'target': target,
            'outcome': outcome,
            'timestamp': len(self.state.exploration_history),
            'surprise_level': self._calculate_surprise(outcome)
        }
        
        self.state.exploration_history.append(exploration_record)
        
        # Update interest based on outcome
        feedback = outcome.get('learning_value', 0.5)
        self.update_interest(target, feedback)
        
    def _calculate_novelty(self, observation: Any, context: Dict) -> float:
        """Calculate novelty score for an observation."""
        # Simple novelty: how different from recent observations
        if not self.state.exploration_history:
            return 1.0  # Everything is novel initially
            
        recent_observations = [
            record['outcome'] for record in self.state.exploration_history[-10:]
        ]
        
        # Simplified novelty calculation
        # In practice, this would use embeddings or other similarity metrics
        observation_str = str(observation)
        similarities = [
            self._simple_similarity(observation_str, str(obs))
            for obs in recent_observations
        ]
        
        max_similarity = max(similarities) if similarities else 0
        novelty = 1.0 - max_similarity
        
        return max(0.0, min(1.0, novelty))
        
    def _calculate_uncertainty(self, observation: Any, context: Dict) -> float:
        """Calculate uncertainty about an observation."""
        # Simplified uncertainty calculation
        # In practice, this would use model confidence, entropy, etc.
        
        # For now, use a simple heuristic based on observation complexity
        observation_str = str(observation)
        complexity = len(set(observation_str.split())) / max(len(observation_str.split()), 1)
        
        # Higher complexity -> higher uncertainty
        uncertainty = min(1.0, complexity * 2)
        
        return uncertainty
        
    def _calculate_learning_potential(self, observation: Any, context: Dict) -> float:
        """Calculate potential learning value of an observation."""
        # Simplified potential calculation
        # Could be based on information gain, connection potential, etc.
        
        # For now, use a combination of novelty and uncertainty
        novelty = self._calculate_novelty(observation, context)
        uncertainty = self._calculate_uncertainty(observation, context)
        
        # High novelty + moderate uncertainty = high learning potential
        potential = novelty * (1 - abs(uncertainty - 0.5) * 2)
        
        return max(0.0, min(1.0, potential))
        
    def _calculate_surprise(self, outcome: Dict[str, Any]) -> float:
        """Calculate surprise level of an exploration outcome."""
        # Simplified surprise calculation
        expected = outcome.get('expected_outcome', 0.5)
        actual = outcome.get('actual_outcome', 0.5)
        
        surprise = abs(expected - actual)
        return min(1.0, surprise)
        
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def get_curiosity_report(self) -> Dict[str, Any]:
        """Generate a report on current curiosity state."""
        return {
            'current_interests': self.state.current_interests,
            'pending_questions': len(self.state.question_queue),
            'exploration_count': len(self.state.exploration_history),
            'top_uncertainties': dict(sorted(
                self.state.uncertainty_levels.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'avg_surprise': np.mean([
                record['surprise_level'] 
                for record in self.state.exploration_history
            ]) if self.state.exploration_history else 0.0
        }
