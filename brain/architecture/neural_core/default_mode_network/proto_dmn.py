"""Proto Default Mode Network (DMN) - Phase 2 Prototype
Handles internal simulation and replay of learned patterns.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

import numpy as np
from typing import List, Dict, Any

# A dummy class to represent MemoryEpisode if the real one isn't available
# In the integrated system, this will come from the EpisodicMemory module
class MemoryEpisode:
    def __init__(self, content, importance):
        self.content = content
        self.importance = importance

class ProtoDMN:
    """
    An early-stage Default Mode Network for internal simulation and replay.
    """
    def __init__(self, replay_threshold: float = 0.8):
        """
        Initializes the Proto-DMN.
        Args:
            replay_threshold: The importance threshold a memory must meet to be replayed.
        """
        self.replay_threshold = replay_threshold
        self.is_active = False
        self.last_replay_activity = None

    def step(self, is_resting: bool, recent_memories: List[MemoryEpisode]) -> Dict[str, Any]:
        """
        Processes one time step of the DMN.
        Args:
            is_resting: A boolean indicating if the brain is in a resting state.
            recent_memories: A list of recent MemoryEpisode objects from the hippocampus.
        Returns:
            A dictionary containing the replay activity and the DMN's status.
        """
        self.is_active = is_resting

        if not self.is_active:
            self.last_replay_activity = None
            return {"active": False, "replay_activity": None, "replayed_memories": 0}

        # Select important memories to replay
        memories_to_replay = [
            mem for mem in recent_memories if mem.importance >= self.replay_threshold
        ]

        if not memories_to_replay:
            self.last_replay_activity = None
            return {"active": True, "replay_activity": None, "replayed_memories": 0}

        # Generate a simulated activity pattern based on the replayed memories
        # This is a simplified representation of replay
        replay_activity = self._generate_replay_activity(memories_to_replay)
        self.last_replay_activity = replay_activity

        return {
            "active": True,
            "replay_activity": replay_activity,
            "replayed_memories": len(memories_to_replay)
        }

    def _generate_replay_activity(self, memories: List[MemoryEpisode]) -> np.ndarray:
        """
        Generates a simulated neural activity pattern from a list of memories.
        This is a placeholder for a more complex process.
        """
        # Simple hash-based activity pattern
        combined_content = "".join([str(mem.content) for mem in memories])

        # Use a fixed-size vector
        activity_vector = np.zeros(100)

        for i, char in enumerate(combined_content):
            if i >= 100:
                break
            activity_vector[i] = (ord(char) % 100) / 100.0

        return activity_vector

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the DMN."""
        return {
            "is_active": self.is_active,
            "last_replay_activity_summary": np.mean(self.last_replay_activity) if self.last_replay_activity is not None else 0
        }
