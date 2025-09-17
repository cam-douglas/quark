

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
# brain_modules/basal_ganglia/dopamine_system.py

"""
Purpose: Simulates the dopamine signaling system for reward processing.
Inputs: Reward Prediction Error
Outputs: Dopamine level
Dependencies: None
"""

class DopamineSystem:
    """
    Simulates the release of dopamine based on reward prediction errors (RPE).
    A positive RPE leads to a dopamine surge, reinforcing the action.
    A negative RPE leads to a dopamine dip, discouraging the action.
    """
    def __init__(self, baseline_dopamine=0.5, rpe_sensitivity=0.2):
        self.baseline_dopamine = baseline_dopamine
        self.rpe_sensitivity = rpe_sensitivity
        self.current_dopamine_level = self.baseline_dopamine

    def calculate_reward_prediction_error(self, predicted_value, actual_value):
        """
        Calculates the difference between the expected and received value.
        """
        return actual_value - predicted_value

    def update_dopamine_level(self, rpe):
        """
        Updates the dopamine level based on the RPE.
        """
        # The change in dopamine is proportional to the RPE
        dopamine_change = self.rpe_sensitivity * rpe
        self.current_dopamine_level = self.baseline_dopamine + dopamine_change
        # Ensure dopamine level stays within a reasonable range, e.g., [0, 1]
        self.current_dopamine_level = max(0, min(1, self.current_dopamine_level))
        return self.current_dopamine_level

    def process_value(self, predicted_value, actual_value):
        """
        A full cycle of calculating RPE and updating dopamine based on a unified value signal.
        """
        rpe = self.calculate_reward_prediction_error(predicted_value, actual_value)
        dopamine_level = self.update_dopamine_level(rpe)
        return dopamine_level, rpe
