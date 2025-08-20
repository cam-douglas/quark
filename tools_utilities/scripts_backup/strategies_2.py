"""
Exploration Strategies - Different approaches to exploration
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import random

class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""
    
    @abstractmethod
    def choose_action(self, environment: Any, context: Dict) -> str:
        """Choose an exploration action."""
        pass
        
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        pass

class RandomStrategy(ExplorationStrategy):
    """Random exploration strategy."""
    
    def __init__(self, action_space: Optional[List[str]] = None):
        self.action_space = action_space or [
            'investigate', 'analyze', 'experiment', 'observe', 'test'
        ]
        
    def choose_action(self, environment: Any, context: Dict) -> str:
        """Choose a random action."""
        base_action = random.choice(self.action_space)
        action_id = random.randint(1, 1000)
        return f"{base_action}_{action_id}"
        
    def get_strategy_name(self) -> str:
        return "random"

class GuidedStrategy(ExplorationStrategy):
    """Guided exploration strategy based on targets."""
    
    def __init__(self, targets: Optional[List[str]] = None):
        self.targets = targets or []
        self.current_target_index = 0
        
    def choose_action(self, environment: Any, context: Dict) -> str:
        """Choose action based on current target."""
        if self.targets:
            target = self.targets[self.current_target_index % len(self.targets)]
            self.current_target_index += 1
            return f"investigate_{target}"
        else:
            return "explore_general"
            
    def get_strategy_name(self) -> str:
        return "guided"
