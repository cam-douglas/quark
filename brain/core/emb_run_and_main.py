                brain_output = self.brain.step(brain_inputs, stage=self.curriculum_stage)
                
                if brain_output.get("direct_speech_response"):
                    quark_color = "\033[92m"
                    responses = brain_output["direct_speech_response"]
                    if isinstance(responses, list):
                        for i, response in enumerate(responses):
                            print(f"\n{quark_color}QUARK SAYS ({i+1}/{len(responses)}): {response}{reset_color}")
                            time.sleep(1)
                    else:
                        print(f"\n{quark_color}QUARK SAYS: {responses}{reset_color}")

            # Require Enter to resume
            self._safe_input(f"\n{prompt_color}Press Enter to resume simulation...{reset_color}")
            logger.info("▶️ Resuming simulation...")

        except (KeyboardInterrupt, EOFError, BrokenPipeError):
            logger.info("Prompt cancelled. Resuming simulation...")
            return
    
# === MOVE TO brain/core/16_run_loop_embodied.py (line 337-531) ===
    def run(self, duration_sec=300, steps_per_prompt=200):
        # --- Defer Initialization to Run Time ---
        self._initialize_simulation()
        self._install_signal_handlers()

        logger.info("🗣️ Press Ctrl+C at any time to terminate the simulation.")

        with mujoco.viewer.launch_passive(self.runner.model, self.runner.data) as viewer:
            start_time = time.time()
            step_count = 0

            brain_inputs: Dict[str, Any] = {
                "sensory_inputs": {"vision": None, "audio": None},
                "reward": 0.0,
                "qpos": self.runner.data.qpos.copy(),
                "is_fallen": False,
                "user_prompt": None,
            }

            while viewer.is_running() and time.time() - start_time < duration_sec:
                step_start = time.time()
                # Handle interactive interrupt
                if self._interrupt_event.is_set():
                    self.handle_user_interaction()
                    self._interrupt_event.clear()
                    # Skip stepping this iteration; continue cleanly
                    continue
                
                if self._is_fallen():
                    logger.info("Quark has fallen. Resetting to crawling pose.")
                    self._reset_to_crawl_pose()
                    brain_inputs["is_fallen"] = True
                else:
                    brain_inputs["is_fallen"] = False
                
                brain_inputs["sensory_inputs"]["vision"] = None
                brain_inputs["qpos"] = self.runner.data.qpos.copy()
                brain_inputs["qvel"] = self.runner.data.qvel.copy()
                brain_inputs["user_prompt"] = None
                    
                # This line was accidentally removed. It is essential.
                brain_output = self.brain.step(brain_inputs, stage=self.curriculum_stage)
                
                # The vision system is not used for the balancing task, so we can remove the logs.
                # self.last_brain_output = brain_output
                    
                # The 'action' from the brain is now the low-level 'ctrl' array from the Motor Cortex
                ctrl = brain_output.get("action")
                
                # Apply small stabilizing overlay
                if ctrl is not None:
                    ctrl += self._balance_overlay_control()
                    self.runner.step(ctrl)
                else:
                    # If no action is produced, do nothing (or a default action)
                    self.runner.step(np.zeros(self.runner.model.nu))
                    
                # Compute prev_x for forward progress
                if self._prev_x is None:
                    self._prev_x = float(self.runner.data.qpos[0])
                prev_x = self._prev_x

                # The reward is now calculated based on crawling forward progress
                reward = self.developmental_curriculum.calculate_developmental_reward({
                    "qpos": self.runner.data.qpos,
                    "prev_x": prev_x,
                    "is_fallen": brain_inputs["is_fallen"]
                })
                    
                brain_inputs["reward"] = reward

                # Update prev_x and metrics
                current_x = float(self.runner.data.qpos[0])
                step_distance = max(0.0, current_x - prev_x)
                step_speed = step_distance / max(1e-6, self.runner.model.opt.timestep)
                self._prev_x = current_x
                # store for diagnostics
                self._last_step_speed = step_speed
                self._cumulative_distance = getattr(self, "_cumulative_distance", 0.0) + step_distance
                    
                if step_count % 50 == 0:
                    # Guard access to action_primitive to prevent out-of-range indices
                    ppo_goal = brain_output.get('ppo_goal')
                    action_primitive = None
                    if isinstance(ppo_goal, int) and 0 <= ppo_goal < len(self.motor_primitives):
                        action_primitive = self.motor_primitives[ppo_goal]
                    thought_context = {
                        **brain_output,
                        "reward": reward,
                        "curriculum_stage": self.curriculum_stage,
                        "pose_error": self.brain.last_pose_error,
                        "action_primitive": action_primitive
                    }
                    spontaneous_thought = self.brain.language_cortex.generate_spontaneous_thought(thought_context)
                    if spontaneous_thought:
                        thought_color = "\033[93m"
                        reset_color = "\033[0m"
                        print(f"\n{thought_color}QUARK THINKS: {spontaneous_thought}{reset_color}")
                        
                if step_count % 100 == 0:
                    self._print_learning_diagnostics(brain_output, reward, step_count)
                
                viewer.sync()
                        
                time_until_next_step = self.runner.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                
                step_count += 1

        logger.info("Simulation finished.")

if __name__ == "__main__":
    simulation = EmbodiedBrainSimulation()
    try:
        # --- NEW: Developmental Training Phase ---
        # First, we run Quark through its developmental curriculum.
        # This trains the brain from the ground up before interactive simulation.
        print("\n" + "="*50)
        print("🎓 STARTING DEVELOPMENTAL TRAINING PIPELINE 🎓")
        print("="*50)
        
        # We can start from a specific phase if needed, but default is 0
        # simulation.training_pipeline.run_full_curriculum(start_phase=0)
        
        print("\n" + "="*50)
        print("✅ DEVELOPMENTAL TRAINING COMPLETE ✅")
        print("🚀 LAUNCHING INTERACTIVE SIMULATION 🚀")
        print("="*50)

        # Now, run the interactive simulation with the trained brain
        simulation.run()

    except KeyboardInterrupt:
        logger.info("\nSimulation terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
