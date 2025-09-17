"""
Minimal QuantumDecisionEngine implementation to satisfy brain_main.py imports.
"""

import logging
import random

logger = logging.getLogger(__name__)


class QuantumDecisionEngine:
    """A minimal quantum decision engine implementation."""

    def __init__(self):
        self.initialized = True
        self.use_quantum = False  # Default to classical in minimal mode
        logger.info("QuantumDecisionEngine initialized (minimal mode)")

    def should_use_quantum(self, problem_type: str = None, complexity: float = 0.0) -> bool:
        """Decide whether to use quantum computing - always returns False in minimal mode."""
        return False

    def quantum_sample(self, probabilities: list) -> int:
        """Sample from probability distribution - uses classical random in minimal mode."""
        if not probabilities:
            return 0
        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            return 0
        normalized = [p / total for p in probabilities]
        # Classical sampling
        return random.choices(range(len(normalized)), weights=normalized, k=1)[0]

    def get_quantum_advantage(self, problem_size: int) -> float:
        """Estimate quantum advantage - returns 1.0 (no advantage) in minimal mode."""
        return 1.0
