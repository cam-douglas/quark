"""
Interest Scorer - Scores and prioritizes interests for curiosity-driven learning
"""

from typing import Dict, List, Any, Optional, Tuple
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
import math

@dataclass
class InterestMetrics:
    """Metrics for scoring interest in a topic or observation"""
    novelty: float = 0.0
    uncertainty: float = 0.0
    complexity: float = 0.0
    relevance: float = 0.0
    learning_potential: float = 0.0
    surprise: float = 0.0
    
class InterestScorer:
    """
    Scores and prioritizes interests for curiosity-driven learning.
    
    Uses multiple factors to determine how interesting something is:
    - Novelty (how different from known)
    - Uncertainty (how much we don't know)
    - Complexity (information richness)
    - Relevance (connection to current goals)
    - Learning potential (expected knowledge gain)
    - Surprise (deviation from expectations)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Scoring weights
        self.weights = {
            'novelty': self.config.get('novelty_weight', 0.25),
            'uncertainty': self.config.get('uncertainty_weight', 0.20),
            'complexity': self.config.get('complexity_weight', 0.15),
            'relevance': self.config.get('relevance_weight', 0.15),
            'learning_potential': self.config.get('learning_potential_weight', 0.15),
            'surprise': self.config.get('surprise_weight', 0.10)
        }
        
        # Historical data for comparison
        self.observation_history: List[Any] = []
        self.interest_history: List[float] = []
        self.current_goals: List[str] = []
        
    def score_interest(self, 
                      observation: Any,
                      context: Optional[Dict] = None) -> Tuple[float, InterestMetrics]:
        """
        Score the interest level of an observation.
        
        Returns:
            Tuple of (total_score, detailed_metrics)
        """
        context = context or {}
        
        # Calculate individual metrics
        metrics = InterestMetrics()
        
        metrics.novelty = self._calculate_novelty(observation, context)
        metrics.uncertainty = self._calculate_uncertainty(observation, context)
        metrics.complexity = self._calculate_complexity(observation, context)
        metrics.relevance = self._calculate_relevance(observation, context)
        metrics.learning_potential = self._calculate_learning_potential(observation, context)
        metrics.surprise = self._calculate_surprise(observation, context)
        
        # Calculate weighted total score
        total_score = (
            self.weights['novelty'] * metrics.novelty +
            self.weights['uncertainty'] * metrics.uncertainty +
            self.weights['complexity'] * metrics.complexity +
            self.weights['relevance'] * metrics.relevance +
            self.weights['learning_potential'] * metrics.learning_potential +
            self.weights['surprise'] * metrics.surprise
        )
        
        # Record this observation and score
        self.observation_history.append(observation)
        self.interest_history.append(total_score)
        
        # Keep history bounded
        max_history = self.config.get('max_history', 1000)
        if len(self.observation_history) > max_history:
            self.observation_history = self.observation_history[-max_history//2:]
            self.interest_history = self.interest_history[-max_history//2:]
            
        return total_score, metrics
        
    def rank_interests(self, 
                      observations: List[Any],
                      contexts: Optional[List[Dict]] = None) -> List[Tuple[Any, float, InterestMetrics]]:
        """
        Rank a list of observations by interest level.
        
        Returns:
            List of (observation, score, metrics) tuples, sorted by score
        """
        if contexts is None:
            contexts = [{}] * len(observations)
            
        scored_observations = []
        
        for obs, ctx in zip(observations, contexts):
            score, metrics = self.score_interest(obs, ctx)
            scored_observations.append((obs, score, metrics))
            
        # Sort by score (descending)
        return sorted(scored_observations, key=lambda x: x[1], reverse=True)
        
    def update_goals(self, goals: List[str]):
        """Update current learning goals for relevance scoring."""
        self.current_goals = goals
        
    def get_interest_trends(self) -> Dict[str, Any]:
        """Analyze trends in interest over time."""
        if len(self.interest_history) < 5:
            return {'trend': 'insufficient_data'}
            
        recent_scores = self.interest_history[-10:]
        older_scores = self.interest_history[-20:-10] if len(self.interest_history) >= 20 else []
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores) if older_scores else recent_avg
        
        trend = 'increasing' if recent_avg > older_avg else 'decreasing'
        
        return {
            'trend': trend,
            'recent_average': recent_avg,
            'older_average': older_avg,
            'volatility': self._calculate_volatility(recent_scores),
            'peak_score': max(self.interest_history),
            'total_observations': len(self.interest_history)
        }
        
    def _calculate_novelty(self, observation: Any, context: Dict) -> float:
        """Calculate how novel an observation is compared to history."""
        if not self.observation_history:
            return 1.0  # Everything is novel when starting
            
        # Compare with recent observations
        similarities = []
        recent_observations = self.observation_history[-20:]  # Compare with last 20
        
        for hist_obs in recent_observations:
            similarity = self._calculate_similarity(observation, hist_obs)
            similarities.append(similarity)
            
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_similarity
        
        return max(0.0, min(1.0, novelty))
        
    def _calculate_uncertainty(self, observation: Any, context: Dict) -> float:
        """Calculate uncertainty about an observation."""
        # Factors that increase uncertainty:
        # 1. Complexity of the observation
        # 2. Lack of similar examples in history
        # 3. Contradictory information
        
        obs_str = str(observation)
        
        # Complexity-based uncertainty
        complexity_uncertainty = min(len(set(obs_str.split())) / 20.0, 1.0)
        
        # Historical familiarity
        if self.observation_history:
            familiarities = [
                self._calculate_similarity(observation, hist_obs)
                for hist_obs in self.observation_history[-10:]
            ]
            avg_familiarity = sum(familiarities) / len(familiarities)
            familiarity_uncertainty = 1.0 - avg_familiarity
        else:
            familiarity_uncertainty = 1.0
            
        # Combine uncertainties
        uncertainty = (complexity_uncertainty + familiarity_uncertainty) / 2.0
        
        return max(0.0, min(1.0, uncertainty))
        
    def _calculate_complexity(self, observation: Any, context: Dict) -> float:
        """Calculate complexity/information richness of observation."""
        obs_str = str(observation)
        
        # Length-based complexity
        length_complexity = min(len(obs_str) / 500.0, 1.0)
        
        # Vocabulary diversity
        words = obs_str.split()
        unique_words = len(set(words))
        vocabulary_complexity = min(unique_words / max(len(words), 1), 1.0) if words else 0.0
        
        # Structural complexity (presence of special characters, numbers)
        special_chars = sum(1 for c in obs_str if not c.isalnum() and not c.isspace())
        structural_complexity = min(special_chars / max(len(obs_str), 1), 0.5)
        
        # Average complexities
        complexity = (length_complexity + vocabulary_complexity + structural_complexity) / 3.0
        
        return max(0.0, min(1.0, complexity))
        
    def _calculate_relevance(self, observation: Any, context: Dict) -> float:
        """Calculate relevance to current goals and interests."""
        if not self.current_goals:
            return 0.5  # Neutral relevance when no goals set
            
        obs_str = str(observation).lower()
        
        # Check overlap with current goals
        relevance_scores = []
        
        for goal in self.current_goals:
            goal_words = set(goal.lower().split())
            obs_words = set(obs_str.split())
            
            if goal_words and obs_words:
                overlap = len(goal_words.intersection(obs_words))
                relevance = overlap / len(goal_words)
                relevance_scores.append(relevance)
            else:
                relevance_scores.append(0.0)
                
        # Use maximum relevance to any goal
        max_relevance = max(relevance_scores) if relevance_scores else 0.0
        
        return max(0.0, min(1.0, max_relevance))
        
    def _calculate_learning_potential(self, observation: Any, context: Dict) -> float:
        """Calculate potential for learning from this observation."""
        # Learning potential is high when:
        # 1. Observation is moderately complex (not too simple, not too complex)
        # 2. We have some but not complete knowledge about it
        # 3. It connects to existing knowledge
        
        novelty = self._calculate_novelty(observation, context)
        uncertainty = self._calculate_uncertainty(observation, context)
        complexity = self._calculate_complexity(observation, context)
        
        # Optimal learning occurs at moderate levels
        # Too familiar = low learning, too unfamiliar = hard to learn
        optimal_novelty = 1.0 - abs(novelty - 0.6) * 2  # Peak at 0.6 novelty
        optimal_uncertainty = 1.0 - abs(uncertainty - 0.5) * 2  # Peak at 0.5 uncertainty
        optimal_complexity = 1.0 - abs(complexity - 0.4) * 2.5  # Peak at 0.4 complexity
        
        learning_potential = (optimal_novelty + optimal_uncertainty + optimal_complexity) / 3.0
        
        return max(0.0, min(1.0, learning_potential))
        
    def _calculate_surprise(self, observation: Any, context: Dict) -> float:
        """Calculate surprise level based on expectations."""
        # Surprise occurs when observation differs from predictions
        # For now, use a simple heuristic based on historical patterns
        
        if len(self.observation_history) < 5:
            return 0.5  # Moderate surprise when insufficient history
            
        # Compare with recent trend patterns
        recent_similarities = []
        for hist_obs in self.observation_history[-5:]:
            similarity = self._calculate_similarity(observation, hist_obs)
            recent_similarities.append(similarity)
            
        # If current observation is very different from recent pattern, it's surprising
        avg_recent_similarity = sum(recent_similarities) / len(recent_similarities)
        surprise = 1.0 - avg_recent_similarity
        
        # Check for pattern breaks
        if len(self.interest_history) >= 3:
            recent_interest_trend = self.interest_history[-3:]
            if all(x <= y for x, y in zip(recent_interest_trend, recent_interest_trend[1:])):
                # Interest was increasing, sudden low interest is surprising
                current_score, _ = self.score_interest(observation, context)
                if current_score < recent_interest_trend[-1] * 0.5:
                    surprise = max(surprise, 0.8)
                    
        return max(0.0, min(1.0, surprise))
        
    def _calculate_similarity(self, obs1: Any, obs2: Any) -> float:
        """Calculate similarity between two observations."""
        str1 = str(obs1).lower()
        str2 = str(obs2).lower()
        
        if not str1 or not str2:
            return 0.0
            
        # Word-based similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Character-level similarity for additional precision
        char_similarity = self._character_similarity(str1, str2)
        
        # Combine similarities
        similarity = (jaccard_similarity * 0.7 + char_similarity * 0.3)
        
        return max(0.0, min(1.0, similarity))
        
    def _character_similarity(self, str1: str, str2: str) -> float:
        """Calculate character-level similarity using simple edit distance."""
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        # Simple Levenshtein-like distance
        max_len = max(len(str1), len(str2))
        
        # Count character differences
        min_len = min(len(str1), len(str2))
        differences = sum(c1 != c2 for c1, c2 in zip(str1[:min_len], str2[:min_len]))
        differences += abs(len(str1) - len(str2))  # Add length difference
        
        similarity = 1.0 - (differences / max_len)
        
        return max(0.0, similarity)
        
    def _calculate_volatility(self, scores: List[float]) -> float:
        """Calculate volatility/variance in interest scores."""
        if len(scores) < 2:
            return 0.0
            
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        volatility = math.sqrt(variance)
        
        return volatility
