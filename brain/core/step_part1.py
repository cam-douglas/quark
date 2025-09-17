from typing import Dict, Any
import time
import numpy as np
from brain.architecture.neural_core.learning.ppo_agent import PPOAgent
from brain.architecture.neural_core.proto_cortex.layer_sheet import LayerSheet

# AlphaGenome biological compliance
try:
    from brain.modules.alphagenome_integration.compliance_engine import ComplianceEngine
    from brain.modules.alphagenome_integration.dna_controller import DNAController
    ALPHAGENOME_COMPLIANCE_AVAILABLE = True
except ImportError:
    ALPHAGENOME_COMPLIANCE_AVAILABLE = False

if True:
    def step(self, inputs: Dict[str, Any], stage: int = 0) -> Dict[str, Any]:
        # --- Lazy Load Datasets on First Step ---
        self._load_training_datasets_if_needed()

        outputs = {}
        t_start = time.time()

        # --- Handle interactive user prompt (if provided) ---
        user_prompt = inputs.get("user_prompt")
        if isinstance(user_prompt, str) and len(user_prompt.strip()) > 0:
            try:
                response_text = self.language_cortex.process_input(user_prompt)
                outputs["direct_speech_response"] = response_text
            except Exception:
                outputs["direct_speech_response"] = "I had trouble responding just now, but I'm still learning."

        # 1. Process sensory inputs
        sensory_inputs = inputs.get("sensory_inputs", {})

        # --- Vision Processing (Decoupled) ---
        self.vision_step_counter += 1
        if self.visual_cortex and self.vision_step_counter >= self.vision_update_frequency:
            self.vision_step_counter = 0
            visual_cortex_output = self.visual_cortex.step({"vision": sensory_inputs})
            self.last_visual_output = visual_cortex_output # Cache the latest output
        else:
            visual_cortex_output = self.last_visual_output # Use cached output

        outputs['visual_cortex'] = visual_cortex_output
        t_vision = time.time()

        # --- SLAM / Perception Adapters (optional) ---
        slam_outputs = {}
        if self.use_rtabmap and self.rtabmap and isinstance(visual_cortex_output, dict):
            try:
                rgb = visual_cortex_output.get("processed_features")
                depth = None
                odom = {"qpos": current_qpos.tolist() if current_qpos is not None else []}
                slam_outputs['rtabmap'] = self.rtabmap.update(rgb, depth, odom)
            except Exception:
                slam_outputs['rtabmap'] = {"pose": [0,0,0], "nodes": 0}
        if self.use_gtsam and self.gtsam:
            try:
                measurements = {"visual": visual_cortex_output}
                slam_outputs['gtsam'] = self.gtsam.update(measurements)
            except Exception:
                slam_outputs['gtsam'] = {"pose": [0,0,0]}
        outputs['slam'] = slam_outputs

        if hasattr(self, 'auditory_cortex') and self.auditory_cortex is not None:
            auditory_cortex_output = self.auditory_cortex.step(sensory_inputs.get("audio"))
            outputs['auditory_cortex'] = auditory_cortex_output
        else:
            auditory_cortex_output = None  # Initialize for later use
            outputs['auditory_cortex'] = None

        current_qpos = inputs.get("qpos")
        current_qvel = inputs.get("qvel")
        if hasattr(self, 'somatosensory_cortex') and self.somatosensory_cortex is not None:
            somatosensory_output = self.somatosensory_cortex.step(current_qpos, current_qvel)
            outputs['somatosensory_cortex'] = somatosensory_output
        else:
            somatosensory_output = {"body_schema": np.zeros(32)}  # Default output
            outputs['somatosensory_cortex'] = None
        t_sensory = time.time()

        # 2. Get current physical state from the environment
        reward = inputs.get("reward", 0.0)
        is_done = bool(inputs.get("is_fallen", False))

        # Build observation vector using the processed body schema
        qpos = current_qpos
        quat = qpos[3:7]
        pos = qpos[0:3]
        body_schema = somatosensory_output.get("body_schema", np.zeros(32)) # Default to zero vector

        # --- NEW: Integrate Visual Cortex Output ---
        # Flatten and normalize the visual data to be included in the observation
        visual_features = visual_cortex_output.get("processed_features", np.array([]))
        if visual_features.size > 0:
            visual_features_flat = visual_features.flatten() / 255.0 # Normalize pixel values
        else:
            # Create a zero vector of a fixed size if no visual features are available
            # This size should be consistent. Let's assume a fixed feature size, e.g., 10x10.
            visual_features_flat = np.zeros(100)

        obs = np.concatenate([
            quat,
            pos,
            body_schema.astype(np.float32), # Use the processed body schema
            visual_features_flat.astype(np.float32)
        ], axis=0).astype(np.float32)

        # Lazy init PPO and Proto-Cortex with proper obs dim
        if self.ppo_agent is None:
            obs_dim = obs.shape[0]
            # Align PPO action space with embodiment action dimension when available
            num_actions = self.act_dim if self.act_dim is not None else 16
            self.ppo_agent = PPOAgent(obs_dim=obs_dim, num_actions=num_actions)
            self.proto_cortex = LayerSheet(input_dim=obs_dim)

        # --- Proto-Cortex Step (Decoupled) ---
        self.proto_cortex_step_counter += 1
        if self.proto_cortex and self.proto_cortex_step_counter >= self.proto_cortex_update_frequency:
            self.proto_cortex_step_counter = 0
            self.total_steps += 1 # Only increment when we actually train
            proto_cortex_output = self.proto_cortex.step(obs, self.total_steps, max_iterations=20000)
        else:
            proto_cortex_output = outputs.get('proto_cortex', {}) # Use last known output

        outputs['proto_cortex'] = proto_cortex_output
        t_proto = time.time()

        # --- Oculomotor Control Step ---
        if hasattr(self, 'oculomotor_cortex') and self.oculomotor_cortex is not None:
            oculomotor_output = self.oculomotor_cortex.step()
            outputs['oculomotor_cortex'] = oculomotor_output
        else:
            outputs['oculomotor_cortex'] = None

        # --- Advanced Planning / Servoing (feature-flagged) ---
        planned_traj = None
        if self.use_ompl and self.ompl:
            try:
                start = qpos[7:7+16] if qpos is not None and len(qpos) >= 23 else np.zeros(16, dtype=np.float32)
                goal = start.copy()
                planned_traj = self.ompl.plan(start, goal, num_points=20)
            except Exception:
                planned_traj = None
        gait_seed = None
        if self.use_towr and self.towr:
            try:
                gait_seed = self.towr.generate_gait_seed(num_steps=10)
            except Exception:
                gait_seed = None
        visp_delta = None
        if self.use_visp and self.visp:
            try:
                loc = visual_cortex_output.get("object_location")
                if isinstance(loc, dict):
                    err = np.array([loc.get("x", 0.0), loc.get("y", 0.0)], dtype=np.float32)
                    visp_delta = self.visp.servo_step(err, gain=0.3)
            except Exception:
                visp_delta = None

        # --- PPO high-level goal ---
        ppo_goal, logprob, value = self.ppo_agent.select_action(obs)
        outputs['ppo_goal'] = ppo_goal
        t_ppo_select = time.time()

        # --- Motor Cortex ---
        if hasattr(self, 'motor_cortex') and self.motor_cortex is not None:
            motor_cortex_output = self.motor_cortex.step(
                ppo_goal=ppo_goal,
                hrm_subgoal=(self.hrm_planner.plan(obs) if self.hrm_planner is not None else None),
                current_qpos=current_qpos
            )
            # Raw command
            raw_motor_command = motor_cortex_output.get('ctrl')
        else:
            motor_cortex_output = {"ctrl": np.zeros(16)}  # Default motor output
            raw_motor_command = motor_cortex_output.get('ctrl')

        # Optional MPC correction (OCS2)
        if self.use_ocs2 and self.ocs2 and raw_motor_command is not None:
            try:
                desired = raw_motor_command.astype(np.float32)
                state = desired.copy()
                raw_motor_command = self.ocs2.mpc_step(state, desired)
            except Exception:
                pass

        # Optional Ruckig retiming toward target joint positions (approximate)
        if self.use_ruckig and self.ruckig and raw_motor_command is not None and current_qpos is not None:
            try:
                current = current_qpos[7:7+len(raw_motor_command)] if len(current_qpos) >= 7+len(raw_motor_command) else np.zeros_like(raw_motor_command)
                raw_motor_command = self.ruckig.retime_step(current, None, None, raw_motor_command)
            except Exception:
                pass

        # --- Cerebellum refinement ---
        if hasattr(self, 'cerebellum') and self.cerebellum is not None:
            refined_motor_command = self.cerebellum.refine_motor_command(raw_motor_command)
        else:
            refined_motor_command = raw_motor_command  # Pass through without refinement

        # Apply small visual servoing delta if available (task-space shim)
        if visp_delta is not None and refined_motor_command is not None and refined_motor_command.shape[0] >= 2:
            refined_motor_command[:2] += visp_delta

        action = refined_motor_command
        outputs['action'] = action
        outputs['motor_cortex'] = motor_cortex_output
        outputs['cerebellum_output'] = refined_motor_command
        t_motor = time.time()

        # --- Limbic System Update ---
        limbic_output = self.limbic_system.step(
            reward=reward,
            is_fallen=is_done,
            goal_achieved=False
        )
        outputs['limbic_system'] = limbic_output

        # --- Brain Stem ---
        sensory_summary = {
            "visual": visual_cortex_output,
            "somatosensory": somatosensory_output,
            "auditory": auditory_cortex_output,
            "limbic": limbic_output
        }
        motor_summary = {"action": action}
        brain_stem_output = self.brain_stem.step(sensory_summary, motor_summary)
        outputs['brain_stem'] = brain_stem_output

        # --- Memory Synchronisation -----------------------------------
        try:
            sync_stats = self.memory_synchronizer.sync()
            outputs['memory_sync'] = sync_stats
        except Exception:
            pass  # keep simulation running even if sync fails
        t_cognitive = time.time()

        # --- Safety ---
        if self.safety_guardian.step(limbic_output):
            outputs['emergency_shutdown'] = True
            return outputs

        # --- Blended Reward ---
        positive_valence = limbic_output.get("positive_valence", 0.0)
        error_signal = limbic_output.get("error_signal", 0.0)
        blended_reward = positive_valence - error_signal
        outputs['reward'] = blended_reward

        # --- AlphaGenome Biological Rule Enforcement ---
        if ALPHAGENOME_COMPLIANCE_AVAILABLE:
            try:
                # Initialize compliance engine if not exists
                if not hasattr(self, 'compliance_engine'):
                    self.compliance_engine = ComplianceEngine()
                    print("🧬 AlphaGenome Compliance Engine activated for biological rule enforcement")

                # Get current brain state for validation
                current_state = {
                    'neuron_activity': visual_cortex_output.get('neural_activity', np.array([0.0])),
                    'synaptic_weights': motor_cortex_output.get('synaptic_weights', np.array([1.0])),
                    'cell_population': len(getattr(self, 'cells', [])),  # If cell tracking exists
                    'simulation_time': getattr(self, 'total_simulation_time', 0.0)
                }

                # Validate biological constraints
                validation_results = {
                    'biological_rules_followed': True,
                    'warnings': [],
                    'violations': []
                }

                # Check simulation boundaries
                cell_count = current_state.get('cell_population', 0)
                sim_time_hours = current_state.get('simulation_time', 0.0) / 3600.0  # Convert to hours

                if not self.compliance_engine.check_simulation_boundaries(cell_count, int(sim_time_hours)):
                    validation_results['violations'].append('Simulation boundaries exceeded')
                    validation_results['biological_rules_followed'] = False

                # Add biological validation to outputs
                outputs['alphagenome_validation'] = validation_results

                # If DNA controller is available, use it for regulatory analysis
                if hasattr(self, 'dna_controller'):
                    try:
                        # Periodic DNA analysis (every 100 steps)
                        if getattr(self, 'total_steps', 0) % 100 == 0:
                            regulatory_analysis = self.dna_controller.analyze_regulatory_network(
                                chromosome="chr1",
                                start=1000000,
                                end=1010000
                            )
                            outputs['regulatory_analysis'] = regulatory_analysis
                    except Exception as e:
                        outputs['dna_analysis_error'] = str(e)

            except Exception as e:
                # Don't crash simulation if biological validation fails
                outputs['alphagenome_error'] = f"Biological validation error: {e}"

        return outputs
