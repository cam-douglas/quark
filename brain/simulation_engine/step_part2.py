        blended_reward = (reward + positive_valence) - error_signal
        outputs['blended_reward'] = blended_reward
        
        # Store transition (training decoupled)
        if (self.prev_obs is not None and self.last_action is not None 
            and hasattr(self, "last_logprob") and hasattr(self, "last_value")):
            self.ppo_agent.store_transition(self.prev_obs.astype(np.float32), int(self.last_action), float(self.last_logprob), float(self.last_value), float(blended_reward), is_done)
        
        if is_done:
            self.ppo_agent.train_if_ready()

        # Cache
        # Cache current step state for next transition storage
        self.prev_obs = obs
        self.last_qpos = current_qpos
        self.last_action = ppo_goal
        self.last_logprob = logprob
        self.last_value = value
        t_learning = time.time()

        self._update_profiling_data(t_start, t_vision, t_sensory, t_proto, t_ppo_select, t_motor, t_cognitive, t_learning)

        # --- Periodic persistence save -------------------------------------
        try:
            if (self.total_steps - self._last_persist_step) >= self._persist_every_steps:
                self.persistence.save_all()
                self._last_persist_step = self.total_steps
        except Exception:
            pass

        # Optional HRM status
        if self.hrm_planner is not None:
            outputs['hrm'] = {
                "active": True
            }

        return outputs
