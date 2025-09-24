
    def _reset_to_crawl_pose(self):
        """
        Sets the humanoid to a stable, physically plausible crawling position.
        """
        self.runner._reset_simulation()

        # Get the default standing pose to modify from
        qpos = self.runner.data.qpos.copy()
        
        # Place torso at safe height just above ground to avoid penetration
        qpos[2] = max(0.6, qpos[2])

        # --- Set a stable quadrupedal stance ---
        # Set hips to a 90-degree angle
        qpos[7] = 0.0   # Right Hip X (forward/back)
        qpos[8] = -1.57 # Right Hip Y (abduction) -> leg out to the side
        qpos[13] = 0.0  # Left Hip X
        qpos[14] = 1.57 # Left Hip Y

        # Bend knees to 90 degrees
        qpos[9] = 1.57  # Right Knee
        qpos[15] = 1.57 # Left Knee
        
        # --- Position the arms for support ---
        # Shoulders forward
        qpos[5] = 1.0 # Right Shoulder X
        qpos[11] = 1.0 # Left Shoulder X
        
        self.runner.data.qpos[:] = qpos
        # Zero velocities to avoid initial impulses
        try:
            self.runner.data.qvel[:] = 0.0
        except Exception:
            self.runner.data.qvel[:] = 0.0  # already zeroed or protected

        mujoco.mj_forward(self.runner.model, self.runner.data)
        
        # Reset internal state trackers
        self._prev_x = None
        self._cumulative_distance = 0.0
        self._last_step_speed = 0.0

    def _calculate_pose_error(self, current_qpos: np.ndarray, target_qpos: np.ndarray) -> Optional[float]:
        """Calculates the difference between the current pose and the target pose."""
        if target_qpos is None or current_qpos is None:
            return None
        
        num_joints_to_match = min(len(current_qpos), len(target_qpos))
        error = np.linalg.norm(
            current_qpos[:num_joints_to_match] - target_qpos[:num_joints_to_match]
        )
        return float(error)


    def _calculate_developmental_reward(self, current_qpos: np.ndarray, is_fallen: bool) -> float:
        """
        Calculate reward based on developmental curriculum progress like a human baby.
        """
        try:
            # Get current developmental target
            target_pose = self.developmental_curriculum.get_current_target_pose()
            
            # Calculate pose error (difference between current and target pose)
            current_pose = current_qpos[7:]  # Skip the free joint (first 7 elements)
            target_joint_pose = target_pose[:len(current_pose)]  # Match dimensions
            
            pose_error = np.linalg.norm(current_pose - target_joint_pose)
            
            # Prepare state information for developmental assessment
            current_state = {
                "pose_error": pose_error,
                "is_fallen": is_fallen,
                "is_stable": not is_fallen and pose_error < 0.5,
                "time_stable": getattr(self, '_stability_timer', 0.0),
                "distance_moved": getattr(self, '_distance_moved', 0.0),
                "movement_variance": np.std(current_pose) if len(current_pose) > 1 else 0.0
            }
            
            # Get developmental reward (includes intrinsic motivation, curiosity, social feedback)
            developmental_reward = self.developmental_curriculum.calculate_developmental_reward(current_state)
            
            # Update progress and check for milestone completion
            milestone_achieved = self.developmental_curriculum.update_progress(current_state)
            if milestone_achieved:
                developmental_reward += 5.0  # Big bonus for milestone achievement!
            
            return float(developmental_reward)
            
        except Exception as e:
            logger.warning(f"Error calculating developmental reward: {e}")
            return -0.1

    def _update_curriculum_stage(self):
        """Checks performance and advances the curriculum stage if successful."""
        if len(self.reward_buffer) < self.reward_buffer.maxlen:
            return # Not enough data yet

        avg_reward = np.mean(self.reward_buffer)
        if avg_reward > self.stage_success_threshold:
            self.curriculum_stage += 1
            self.reward_buffer.clear() # Reset buffer for the new stage
            logger.info(f"🎉 SUCCESS! Advancing to Curriculum Stage {self.curriculum_stage}")
            self.brain.rl_agent.epsilon = 0.9 # Encourage exploration in the new stage
