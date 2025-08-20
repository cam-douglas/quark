"""
Exploration Module - Core exploration system for active learning
"""

from typing import List, Dict, Any, Optional, Tuple, Protocol
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
from enum import Enum
import random
import logging

class ExplorationMode(Enum):
    RANDOM = "random"
    GUIDED = "guided"
    CURIOSITY_DRIVEN = "curiosity_driven"
    SAFE = "safe"
    AGGRESSIVE = "aggressive"

@dataclass
class ExplorationResult:
    """Result of an exploration action"""
    action: str
    observation: Any
    reward: float
    info: Dict[str, Any]
    success: bool
    learning_value: float

class EnvironmentInterface(Protocol):
    """Protocol for environment interaction"""
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take a step in the environment"""
        ...
        
    def reset(self) -> Any:
        """Reset the environment"""
        ...
        
    def get_state(self) -> Any:
        """Get current environment state"""
        ...

class ExplorationModule:
    """
    Core exploration module for active learning and knowledge discovery.
    
    Implements various exploration strategies to maximize learning
    while maintaining safety and efficiency.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Exploration parameters
        self.exploration_rate = self.config.get('exploration_rate', 0.3)
        self.safety_threshold = self.config.get('safety_threshold', 0.8)
        self.learning_threshold = self.config.get('learning_threshold', 0.1)
        
        # Exploration history
        self.exploration_history: List[ExplorationResult] = []
        self.environment_model: Dict[str, Any] = {}
        self.safety_model: Dict[str, float] = {}
        
        # Current mode
        self.mode = ExplorationMode.GUIDED
        
    def explore(self, 
                environment: EnvironmentInterface,
                target: Optional[str] = None,
                mode: Optional[ExplorationMode] = None) -> ExplorationResult:
        """
        Execute an exploration action in the environment.
        
        Args:
            environment: Environment to explore
            target: Specific target to explore (if any)
            mode: Exploration mode to use
            
        Returns:
            ExplorationResult with outcome details
        """
        mode = mode or self.mode
        
        # Choose exploration action based on mode
        action = self._choose_action(environment, target, mode)
        
        # Assess safety before taking action
        if not self._is_safe_action(action, environment):
            self.logger.warning(f"Skipping unsafe action: {action}")
            return self._create_safe_fallback_result(action)
            
        # Execute action
        try:
            observation, reward, done, info = environment.step(action)
            
            # Calculate learning value
            learning_value = self._calculate_learning_value(
                action, observation, reward, info
            )
            
            result = ExplorationResult(
                action=str(action),
                observation=observation,
                reward=reward,
                info=info,
                success=True,
                learning_value=learning_value
            )
            
            # Update models
            self._update_environment_model(action, observation, reward)
            self._update_safety_model(action, observation, reward)
            
            # Record exploration
            self.exploration_history.append(result)
            
            self.logger.info(f"Exploration successful: {action} -> "
                           f"reward={reward:.3f}, learning={learning_value:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Exploration failed: {action} -> {e}")
            return ExplorationResult(
                action=str(action),
                observation=None,
                reward=-1.0,
                info={'error': str(e)},
                success=False,
                learning_value=0.0
            )
            
    def plan_exploration_sequence(self, 
                                  targets: List[str],
                                  max_steps: int = 10) -> List[str]:
        """
        Plan a sequence of exploration actions to investigate targets.
        
        Returns optimized sequence of actions.
        """
        if not targets:
            return []
            
        # Simple greedy planning - can be enhanced with more sophisticated algorithms
        planned_actions = []
        remaining_targets = targets.copy()
        
        for step in range(max_steps):
            if not remaining_targets:
                break
                
            # Choose next target based on expected learning value
            next_target = self._select_next_target(remaining_targets)
            action = self._target_to_action(next_target)
            
            planned_actions.append(action)
            remaining_targets.remove(next_target)
            
        return planned_actions
        
    def get_exploration_recommendations(self, 
                                       num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get recommendations for promising exploration directions."""
        recommendations = []
        
        # Analyze exploration history for patterns
        if len(self.exploration_history) < 3:
            # Not enough history - recommend basic exploration
            return self._get_basic_recommendations(num_recommendations)
            
        # Find high-value areas
        high_value_actions = [
            result for result in self.exploration_history[-20:]
            if result.learning_value > self.learning_threshold
        ]
        
        # Recommend similar actions or unexplored variations
        for result in high_value_actions[:num_recommendations]:
            variations = self._generate_action_variations(result.action)
            for variation in variations:
                recommendations.append({
                    'action': variation,
                    'expected_value': result.learning_value * 0.8,  # Discounted
                    'confidence': self._estimate_action_confidence(variation),
                    'reason': f"Similar to successful action: {result.action}"
                })
                
        # Fill remaining slots with novelty-based recommendations
        while len(recommendations) < num_recommendations:
            novel_action = self._generate_novel_action()
            recommendations.append({
                'action': novel_action,
                'expected_value': 0.5,  # Unknown
                'confidence': 0.3,  # Low confidence
                'reason': "Novel exploration direction"
            })
            
        return recommendations[:num_recommendations]
        
    def _choose_action(self, 
                       environment: EnvironmentInterface,
                       target: Optional[str],
                       mode: ExplorationMode) -> str:
        """Choose an exploration action based on the current mode."""
        
        if mode == ExplorationMode.RANDOM:
            return self._random_action(environment)
        elif mode == ExplorationMode.GUIDED and target:
            return self._guided_action(environment, target)
        elif mode == ExplorationMode.CURIOSITY_DRIVEN:
            return self._curiosity_driven_action(environment)
        elif mode == ExplorationMode.SAFE:
            return self._safe_action(environment)
        elif mode == ExplorationMode.AGGRESSIVE:
            return self._aggressive_action(environment)
        else:
            # Default to guided or random
            if target:
                return self._guided_action(environment, target)
            else:
                return self._random_action(environment)
                
    def _random_action(self, environment: EnvironmentInterface) -> str:
        """Generate a random exploration action."""
        # Simple random action generation
        action_types = ['investigate', 'analyze', 'experiment', 'observe', 'test']
        action_type = random.choice(action_types)
        
        # Add some randomness to the action
        action_id = random.randint(1, 100)
        return f"{action_type}_{action_id}"
        
    def _guided_action(self, environment: EnvironmentInterface, target: str) -> str:
        """Generate a guided action towards a specific target."""
        return f"investigate_{target}"
        
    def _curiosity_driven_action(self, environment: EnvironmentInterface) -> str:
        """Generate an action based on curiosity/uncertainty."""
        # Find areas with high uncertainty in our model
        uncertain_areas = [
            action for action, confidence in self.safety_model.items()
            if confidence < 0.5
        ]
        
        if uncertain_areas:
            target_area = random.choice(uncertain_areas)
            return f"explore_{target_area}"
        else:
            return self._random_action(environment)
            
    def _safe_action(self, environment: EnvironmentInterface) -> str:
        """Generate a safe exploration action."""
        # Choose from known safe actions
        safe_actions = [
            result.action for result in self.exploration_history
            if result.success and result.reward >= 0
        ]
        
        if safe_actions:
            return random.choice(safe_actions)
        else:
            return "observe_environment"  # Default safe action
            
    def _aggressive_action(self, environment: EnvironmentInterface) -> str:
        """Generate an aggressive exploration action."""
        # Choose actions that might have high learning value but unknown safety
        return f"experiment_novel_{random.randint(1, 1000)}"
        
    def _is_safe_action(self, action: str, environment: EnvironmentInterface) -> bool:
        """Assess if an action is safe to execute."""
        # Check safety model
        safety_score = self.safety_model.get(action, 0.5)  # Unknown = moderate safety
        
        # Apply safety threshold
        return safety_score >= self.safety_threshold
        
    def _create_safe_fallback_result(self, unsafe_action: str) -> ExplorationResult:
        """Create a safe fallback result when an action is deemed unsafe."""
        return ExplorationResult(
            action=f"safe_fallback_for_{unsafe_action}",
            observation="Action skipped for safety",
            reward=0.0,
            info={'reason': 'safety_fallback', 'original_action': unsafe_action},
            success=True,
            learning_value=0.1  # Small learning value for safety awareness
        )
        
    def _calculate_learning_value(self, 
                                  action: str,
                                  observation: Any,
                                  reward: float,
                                  info: Dict) -> float:
        """Calculate the learning value of an exploration result."""
        # Simple learning value calculation
        # Can be enhanced with more sophisticated metrics
        
        value = 0.0
        
        # Reward contributes to learning value
        value += max(0, reward) * 0.3
        
        # Novelty contributes (how different from previous observations)
        novelty = self._calculate_novelty(observation)
        value += novelty * 0.4
        
        # Information gain contributes
        info_gain = len(str(info)) / 1000.0  # Simple proxy
        value += min(info_gain, 0.3)
        
        return min(1.0, value)
        
    def _calculate_novelty(self, observation: Any) -> float:
        """Calculate novelty of an observation."""
        if not self.exploration_history:
            return 1.0
            
        # Simple novelty calculation
        observation_str = str(observation)
        
        # Compare with recent observations
        recent_observations = [
            str(result.observation) 
            for result in self.exploration_history[-10:]
        ]
        
        similarities = [
            self._simple_similarity(observation_str, obs)
            for obs in recent_observations
        ]
        
        max_similarity = max(similarities) if similarities else 0
        return 1.0 - max_similarity
        
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        if not text1 or not text2:
            return 0.0
            
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def _update_environment_model(self, action: str, observation: Any, reward: float):
        """Update internal model of the environment."""
        if action not in self.environment_model:
            self.environment_model[action] = {
                'observations': [],
                'rewards': [],
                'count': 0
            }
            
        model = self.environment_model[action]
        model['observations'].append(observation)
        model['rewards'].append(reward)
        model['count'] += 1
        
        # Keep only recent history
        if len(model['observations']) > 100:
            model['observations'] = model['observations'][-50:]
            model['rewards'] = model['rewards'][-50:]
            
    def _update_safety_model(self, action: str, observation: Any, reward: float):
        """Update safety model based on exploration results."""
        # Simple safety scoring
        if reward >= 0:
            safety_score = 0.8  # Positive reward = generally safe
        else:
            safety_score = 0.2  # Negative reward = potentially unsafe
            
        # Update with exponential moving average
        current_safety = self.safety_model.get(action, 0.5)
        alpha = 0.3
        new_safety = alpha * safety_score + (1 - alpha) * current_safety
        
        self.safety_model[action] = new_safety
        
    def _get_basic_recommendations(self, num: int) -> List[Dict[str, Any]]:
        """Get basic exploration recommendations when history is limited."""
        basic_actions = [
            'observe_environment',
            'analyze_patterns', 
            'test_hypothesis',
            'explore_boundaries',
            'investigate_anomalies'
        ]
        
        recommendations = []
        for i, action in enumerate(basic_actions[:num]):
            recommendations.append({
                'action': action,
                'expected_value': 0.6,
                'confidence': 0.7,
                'reason': 'Basic exploration strategy'
            })
            
        return recommendations
        
    def _select_next_target(self, targets: List[str]) -> str:
        """Select the next target for exploration planning."""
        # Simple selection - choose the first target
        # Can be enhanced with value-based selection
        return targets[0]
        
    def _target_to_action(self, target: str) -> str:
        """Convert a target to an exploration action."""
        return f"investigate_{target}"
        
    def _generate_action_variations(self, action: str) -> List[str]:
        """Generate variations of a successful action."""
        base = action.split('_')[0] if '_' in action else action
        variations = [
            f"{base}_variant_1",
            f"{base}_variant_2", 
            f"enhanced_{action}",
            f"focused_{action}"
        ]
        return variations
        
    def _estimate_action_confidence(self, action: str) -> float:
        """Estimate confidence in an action's outcome."""
        # Simple confidence based on safety and history
        safety = self.safety_model.get(action, 0.5)
        history_count = self.environment_model.get(action, {}).get('count', 0)
        
        confidence = safety * 0.7 + min(history_count / 10.0, 0.3)
        return min(1.0, confidence)
        
    def _generate_novel_action(self) -> str:
        """Generate a novel action for exploration."""
        action_types = ['discover', 'analyze', 'synthesize', 'experiment', 'probe']
        action_type = random.choice(action_types)
        novel_id = len(self.exploration_history) + random.randint(1, 100)
        
        return f"{action_type}_novel_{novel_id}"
        
    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get statistics about exploration performance."""
        if not self.exploration_history:
            return {'total_explorations': 0}
            
        total = len(self.exploration_history)
        successful = sum(1 for r in self.exploration_history if r.success)
        avg_reward = sum(r.reward for r in self.exploration_history) / total
        avg_learning = sum(r.learning_value for r in self.exploration_history) / total
        
        return {
            'total_explorations': total,
            'success_rate': successful / total,
            'average_reward': avg_reward,
            'average_learning_value': avg_learning,
            'environment_model_size': len(self.environment_model),
            'safety_model_size': len(self.safety_model),
            'current_mode': self.mode.value
        }
