
def _print_learning_diagnostics(self, brain_output: Dict[str, Any], reward: float, step_count: int):
        """Prints learning progress diagnostics for the crawling task."""
        action = brain_output.get("ppo_goal") # Use the high-level goal
        
        # Ensure the action is an integer before using it as an index.
        if action is not None:
            try:
                action_idx = int(action)
                primitive = self.motor_primitives[action_idx] if action_idx < len(self.motor_primitives) else "unknown"
            except (ValueError, TypeError):
                primitive = "invalid_action"
        else:
            primitive = "None"
            action_idx = "N/A"

        # Crawl diagnostics
        x = float(self.runner.data.qpos[0])
        step_speed = getattr(self, "_last_step_speed", 0.0)
        total_dist = getattr(self, "_cumulative_distance", 0.0)

        print(f"\n🧎 QUARK CRAWLING (Step {step_count}):")
        print(f"   - 🎯 Goal: crawl_forward")
        print(f"   - 🎬 Action: {action_idx} ({primitive})")
        print(f"   - 💰 Reward: {reward:.4f}")
        print(f"   - 📏 X Position: {x:.3f}")
        print(f"   - 🚀 Speed (step): {step_speed:.3f} m/s")
        print(f"   - 🧮 Distance (cumulative): {total_dist:.3f} m")
        print(f"   - 🤖 PPO Rollout Size: {getattr(self.brain.ppo_agent, 'rollout_len', 'n/a') if hasattr(self.brain, 'ppo_agent') and self.brain.ppo_agent else 'n/a'}")


    def _is_fallen(self) -> bool:
        """
        Checks if the humanoid has fallen over.
        Returns True if the torso is below a certain height threshold.
        """
        torso_z_position = self.runner.data.qpos[2]
        return torso_z_position < 0.3 # Lower fall threshold for crawling


    def _quat_to_euler(self, q):
        """Converts a quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = q
        # roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        # pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        # yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return roll, pitch, yaw


    def _balance_overlay_control(self):
        """Small stabilizing corrections at hips to counter torso pitch/roll."""
        # Root orientation quaternion is at qpos[3:7] = (w, x, y, z)
        q = self.runner.data.qpos[3:7]
        roll, pitch, _ = self._quat_to_euler(q)
        # Proportional gains (small)
        k_pitch = 0.6
        k_roll = 0.6
        # Map: hip_x ~ forward/back (pitch), hip_y ~ left/right (roll)
        correction = np.zeros(self.runner.model.nu) # Use runner.model.nu for total actuators
        if 'right_hip_x' in self.actuator_ids and 'left_hip_x' in self.actuator_ids:
            correction[self.actuator_ids['right_hip_x']] += -k_pitch * pitch
            correction[self.actuator_ids['left_hip_x']]  += -k_pitch * pitch
        if 'right_hip_y' in self.actuator_ids and 'left_hip_y' in self.actuator_ids:
            correction[self.actuator_ids['right_hip_y']] += -k_roll * roll
            correction[self.actuator_ids['left_hip_y']]  += -k_roll * roll
        return correction

    def handle_user_interaction(self):
        """Handles the user interaction sequence: prompt, get response, and auto-resume."""
        prompt_color = "\033[96m"
        reset_color = "\033[0m"
        
        try:
            message = self._safe_input(f"\n{prompt_color}USER PROMPT > {reset_color}")
            
            if message:
                self.last_user_input = message
                
                brain_inputs = {
                    "user_prompt": message,
                    "sensory_inputs": {}, "reward": 0.0, "qpos": self.runner.data.qpos.copy(), "is_fallen": False,
               }
