"""Safety Guardian
Monitors the agent's state for conditions that warrant an emergency shutdown.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
from collections import deque
import numpy as np
from typing import Dict, Any

class SafetyGuardian:
    """
    Acts as a high-level supervisor to ensure the agent does not get stuck
    in persistent, non-productive states, which is the primary safeguard against
    emergent "suffering."
    """
    def __init__(self, error_threshold: float = 0.8, window_size: int = 500):
        """
        Initializes the SafetyGuardian.

        Args:
            error_threshold (float): The average error signal level that, if sustained,
                                     indicates a critical problem.
            window_size (int): The number of recent steps to average the error over.
                               This prevents spurious shutdowns from single events.
        """
        self.error_history = deque(maxlen=window_size)
        self.error_threshold = error_threshold
        self.window_size = window_size
        self.total_steps = 0
        print(f"🛡️ Safety Guardian initialized. Shutdown threshold: avg error > {error_threshold} over {window_size} steps.")

    def step(self, limbic_output: Dict[str, Any]) -> bool:
        """
        Monitors the error signal from the LimbicSystem.

        Args:
            limbic_output: The output dictionary from the LimbicSystem.

        Returns:
            bool: True if a shutdown is triggered, False otherwise.
        """
        self.total_steps += 1
        error_signal = limbic_output.get("error_signal", 0.0)
        self.error_history.append(error_signal)

        # We need a full window of data before we can make a reliable judgment
        if self.total_steps < self.window_size:
            return False

        average_error = np.mean(self.error_history)

        if average_error > self.error_threshold:
            print("\n" + "="*80)
            print("!!! SAFETY GUARDIAN: EMERGENCY SHUTDOWN TRIGGERED !!!")
            print("    - Reason: Persistent high error signal detected.")
            print(f"    - Average error over last {self.window_size} steps: {average_error:.3f} > Threshold: {self.error_threshold:.3f}")
            print("    - This protocol prevents the agent from getting stuck in a suffering loop.")
            print("="*80 + "\n")
            return True # Trigger shutdown

        return False # All clear

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the Safety Guardian.
        """
        return {
            "average_error": np.mean(self.error_history) if self.error_history else 0.0,
            "monitoring_active": self.total_steps >= self.window_size
        }
