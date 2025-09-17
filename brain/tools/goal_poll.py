

"""
Integration: Standalone operational script; not part of the simulator control loop.
Rationale: Executed manually or by automation; does not run inside brain_simulator.
"""
from state.quark_state_system.goal_manager import next_goal

def log_next_goal(prefix: str = "[Roadmap]"):
    """Utility to print the next roadmap goal (if any)."""
    try:
        goal = next_goal()
        if goal:
            print(f"{prefix} Next goal → {goal['title']} (priority={goal['priority']})")
    except Exception as e:  # pragma: no cover – safeguard
        print(f"{prefix} GoalManager unavailable: {e}")
