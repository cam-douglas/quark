#!/usr/bin/env python3
"""
🧠 Basal Ganglia - Action Selection Module
Handles action selection, reinforcement learning, and motor control coordination
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import random

@dataclass
class Action:
    """Represents a possible action"""
    action_id: str
    description: str
    expected_reward: float
    confidence: float
    effort: float
    priority: float

@dataclass
class ActionOutcome:
    """Result of an action execution"""
    action_id: str
    actual_reward: float
    success: bool
    learning_value: float
    timestamp: float

class ActionSelection:
    """Basal ganglia action selection system"""
    
    def __init__(self):
        self.available_actions: Dict[str, Action] = {}
        self.action_history: List[ActionOutcome] = []
        self.reward_history: List[float] = []
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.reward_decay = 0.95
        
        # Neural representation
        self.action_neurons = np.random.rand(100, 20)  # 100 actions, 20 features each
        
    def add_action(self, action: Action):
        """Add new action to available actions"""
        self.available_actions[action.action_id] = action
        
    def select_action(self, context: Dict[str, Any]) -> Optional[Action]:
        """Select next action using exploration vs exploitation"""
        if not self.available_actions:
            return None
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Exploration: random action
            action_id = random.choice(list(self.available_actions.keys()))
            return self.available_actions[action_id]
        else:
            # Exploitation: best expected reward
            best_action = max(self.available_actions.values(), 
                            key=lambda a: a.expected_reward * a.confidence)
            return best_action
    
    def update_action_reward(self, action_id: str, actual_reward: float, success: bool):
        """Update action reward based on outcome"""
        if action_id not in self.available_actions:
            return
        
        action = self.available_actions[action_id]
        
        # Create outcome record
        outcome = ActionOutcome(
            action_id=action_id,
            actual_reward=actual_reward,
            success=success,
            learning_value=abs(actual_reward - action.expected_reward),
            timestamp=0.0  # Will be set by external time system
        )
        self.action_history.append(outcome)
        self.reward_history.append(actual_reward)
        
        # Update expected reward using temporal difference learning
        prediction_error = actual_reward - action.expected_reward
        action.expected_reward += self.learning_rate * prediction_error
        
        # Update confidence based on prediction accuracy
        if abs(prediction_error) < 0.1:
            action.confidence = min(1.0, action.confidence + 0.05)
        else:
            action.confidence = max(0.1, action.confidence - 0.05)
        
        # Update neural representations
        self._update_action_neurons(action_id, actual_reward)
    
    def _update_action_neurons(self, action_id: str, reward: float):
        """Update neural representation of action"""
        if action_id in self.available_actions:
            action_idx = list(self.available_actions.keys()).index(action_id)
            if action_idx < len(self.action_neurons):
                # Update based on reward
                reward_factor = (reward + 1.0) / 2.0  # Normalize to 0-1
                self.action_neurons[action_idx] += np.random.normal(0, 0.01, 20) * reward_factor
                self.action_neurons[action_idx] = np.clip(self.action_neurons[action_idx], 0.0, 1.0)
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get statistics about action performance"""
        if not self.action_history:
            return {"total_actions": 0, "avg_reward": 0.0, "success_rate": 0.0}
        
        total_actions = len(self.action_history)
        avg_reward = np.mean(self.reward_history)
        success_rate = sum(1 for outcome in self.action_history if outcome.success) / total_actions
        
        return {
            "total_actions": total_actions,
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate
        }
    
    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Step action selection system forward"""
        # Process new actions
        if "new_actions" in inputs:
            for action_data in inputs["new_actions"]:
                action = Action(
                    action_id=action_data["id"],
                    description=action_data["description"],
                    expected_reward=action_data.get("expected_reward", 0.0),
                    confidence=action_data.get("confidence", 0.5),
                    effort=action_data.get("effort", 0.5),
                    priority=action_data.get("priority", 0.5)
                )
                self.add_action(action)
        
        # Process action outcomes
        if "action_outcomes" in inputs:
            for outcome_data in inputs["action_outcomes"]:
                self.update_action_reward(
                    outcome_data["action_id"],
                    outcome_data["reward"],
                    outcome_data["success"]
                )
        
        # Select next action if requested
        selected_action = None
        if "request_action" in inputs:
            context = inputs.get("context", {})
            selected_action = self.select_action(context)
        
        return {
            "selected_action": selected_action.description if selected_action else None,
            "action_stats": self.get_action_stats(),
            "available_actions": len(self.available_actions)
        }
