"""
Minimal AutonomousAgent implementation to satisfy brain_main.py imports.
"""

import logging

logger = logging.getLogger(__name__)


class AutonomousAgent:
    """A minimal autonomous agent implementation."""

    def __init__(self, workspace_root: str = None):
        self.workspace_root = workspace_root or "."
        self.initialized = True
        self.roadmap = []  # Empty roadmap in minimal mode

        # Add compliance attribute for QuarkDriver compatibility
        from .prompt_guardian import PromptGuardian
        self.compliance = PromptGuardian(workspace_root=self.workspace_root)

        logger.info("AutonomousAgent initialized (minimal mode)")

    def execute_next_goal(self) -> bool:
        """Execute next goal - minimal implementation returns False."""
        # In minimal mode, we don't actually execute goals
        # This prevents the background thread from doing work
        return False

    def get_next_goal(self):
        """Get next goal - returns None in minimal mode."""
        return None

    def get_next_actionable_goal(self):
        """Get next actionable goal - returns None in minimal mode."""
        return None

    def is_ready(self) -> bool:
        """Check if agent is ready."""
        return self.initialized
