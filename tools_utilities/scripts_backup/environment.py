"""
Environment Interface - Defines exploration environments
"""

from typing import Any, Tuple, Dict
from abc import ABC, abstractmethod

class EnvironmentInterface(ABC):
    """Interface for exploration environments."""
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take a step in the environment."""
        pass
        
    @abstractmethod
    def reset(self) -> Any:
        """Reset the environment."""
        pass
        
    @abstractmethod
    def get_state(self) -> Any:
        """Get current environment state."""
        pass

class FileSystemEnvironment(EnvironmentInterface):
    """File system exploration environment."""
    
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.current_state = "idle"
        
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Explore the file system."""
        observation = f"Explored {action} in {self.root_path}"
        reward = 0.5  # Neutral reward
        done = False
        info = {'action': str(action), 'path': self.root_path}
        
        return observation, reward, done, info
        
    def reset(self) -> Any:
        """Reset to root directory."""
        self.current_state = "idle"
        return f"Reset to {self.root_path}"
        
    def get_state(self) -> Any:
        """Get current state."""
        return self.current_state
