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

    def calculate_reward_prediction_error(self, predicted_reward, actual_reward):
        """
        Calculates the difference between the expected and received reward.
        """
        return actual_reward - predicted_reward

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

    def process_reward(self, predicted_reward, actual_reward):
        """
        A full cycle of calculating RPE and updating dopamine.
        """
        rpe = self.calculate_reward_prediction_error(predicted_reward, actual_reward)
        dopamine_level = self.update_dopamine_level(rpe)
        return dopamine_level, rpe

if __name__ == '__main__':
    # Example usage
    dopamine_system = DopamineSystem()

    # --- Scenario 1: Positive surprise (unexpected reward) ---
    predicted_reward = 0.2
    actual_reward = 1.0
    dopamine_level, rpe = dopamine_system.process_reward(predicted_reward, actual_reward)
    print(f"--- Positive Surprise ---")
    print(f"Predicted Reward: {predicted_reward}, Actual Reward: {actual_reward}")
    print(f"Reward Prediction Error (RPE): {rpe:.2f}")
    print(f"Resulting Dopamine Level: {dopamine_level:.2f} (Surge)")
    print("-" * 25)

    # --- Scenario 2: Negative surprise (reward less than expected) ---
    predicted_reward = 0.8
    actual_reward = 0.1
    dopamine_level, rpe = dopamine_system.process_reward(predicted_reward, actual_reward)
    print(f"--- Negative Surprise ---")
    print(f"Predicted Reward: {predicted_reward}, Actual Reward: {actual_reward}")
    print(f"Reward Prediction Error (RPE): {rpe:.2f}")
    print(f"Resulting Dopamine Level: {dopamine_level:.2f} (Dip)")
    print("-" * 25)

    # --- Scenario 3: No surprise (reward as expected) ---
    predicted_reward = 0.5
    actual_reward = 0.5
    dopamine_level, rpe = dopamine_system.process_reward(predicted_reward, actual_reward)
    print(f"--- No Surprise ---")
    print(f"Predicted Reward: {predicted_reward}, Actual Reward: {actual_reward}")
    print(f"Reward Prediction Error (RPE): {rpe:.2f}")
    print(f"Resulting Dopamine Level: {dopamine_level:.2f} (Baseline)")
    print("-" * 25)
