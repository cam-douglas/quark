"""Long-Term Memory Module for Lifelong Learning
This module stores experiences across all episodes to enable curiosity and novelty-seeking.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
from collections import defaultdict
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class LongTermMemory:
    """
    A persistent memory system that records the frequency of state-action pairs
    encountered by the agent throughout its entire existence.
    """
    def __init__(self):
        """
        Initializes the long-term memory.
        The memory is a dictionary where keys are state tuples and values are
        dictionaries of action counts.
        """
        # {(state): {action: count}}
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        self.total_experiences = 0
        print("🧠 Long-Term Memory initialized. Ready to remember everything.")

    def record_experience(self, state: Tuple, action: int):
        """
        Records that a specific action was taken in a specific state.
        This helps the agent understand what it has already tried.
        """
        self.state_action_counts[state][action] += 1
        self.total_experiences += 1

    def get_visit_count(self, state: Tuple, action: int) -> int:
        """
        Retrieves how many times a specific action has been taken in a state.
        A high count means the action is "boring" and well-explored.
        """
        return self.state_action_counts[state].get(action, 0)

    def get_total_visits_for_state(self, state: Tuple) -> int:
        """
        Calculates the total number of times any action has been taken from a state.
        """
        return sum(self.state_action_counts[state].values())

    def get_novelty_score(self, state: Tuple) -> float:
        """
        Calculates a novelty score based on how infrequently the given state
        has been visited. This is a key component of intrinsic curiosity.
        The score is inversely proportional to the visit count.
        A higher score means the state is more novel.
        """
        visit_count = self.get_total_visits_for_state(state)

        # The novelty is the inverse of the square root of the visit count.
        # We add 1 to the denominator to avoid division by zero for new states.
        # This ensures that brand new states have a high novelty score.
        return 1.0 / np.sqrt(1 + visit_count)

    def get_status(self):
        """Returns statistics about the memory's contents."""
        return {
            "unique_states_visited": len(self.state_action_counts),
            "total_experiences_recorded": self.total_experiences,
        }

def create_long_term_memory() -> LongTermMemory:
    """Factory function for creating a LongTermMemory instance."""
    return LongTermMemory()
