
    def _calculate_pose_error(self, current_qpos, target_pose_full) -> Optional[float]:
        """Calculates the difference between the current pose and the target pose."""
        if target_pose_full is None or current_qpos is None:
            return None
        
        num_joints_to_match = min(len(current_qpos), len(target_pose_full))
        error = np.linalg.norm(
            current_qpos[:num_joints_to_match] - target_pose_full[:num_joints_to_match]
        )
        return float(error)
