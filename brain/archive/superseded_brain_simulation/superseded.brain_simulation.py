# === MOVE TO brain/core/01_header_imports_(line_1-120).py ===

class BrainSimulator:
    """The master controller for orchestrating all brain modules."""

    def __init__(self, use_hrm: bool = False, obs_dim: int = None, act_dim: int = None, embodiment: Any = None):
        """
        Initializes the BrainSimulator.
        """

        # Caching mechanism for the brain's biological specification
        spec_cache_path = "brain_specification.json"
        disable_alphagenome = os.environ.get("QUARK_DISABLE_ALPHAGENOME", "1") == "1"
        force_alphagenome = os.environ.get("QUARK_FORCE_ALPHAGENOME", "0") == "1"

        # Decide path: use cache unless explicitly forcing AlphaGenome or cache missing
        use_cache = (os.path.exists(spec_cache_path) and not force_alphagenome)

        if use_cache:
            print("🧠 Found cached brain specification. Loading from file...")
            with open(spec_cache_path, 'r') as f:
                self.brain_bio_spec = json.load(f)
            print("✅ Brain specification loaded from cache.")
        else:
            if disable_alphagenome and not force_alphagenome:
                print("🧬 AlphaGenome disabled. Using minimal default biological spec.")
                self.brain_bio_spec = {
                    "final_state": {"cell_type_distribution": {"neuron": 10}, "total_tissues": 0, "tissue_types": []}
                }
            else:
                print("🧬 Running AlphaGenome biological simulation...")
                try:
                    from brain.modules.alphagenome_integration.biological_simulator import BiologicalSimulator
                    self.bio_simulator = BiologicalSimulator()
                    # Run a 3-day (72-hour) simulation to develop the brain structure
                    self.brain_bio_spec = self.bio_simulator.run_simulation(duration=72.0)
                    print("✅ Biological simulation complete. Brain specification generated.")
                except Exception as e:
                    print(f"⚠️ AlphaGenome unavailable, using fallback spec: {e}")
                    self.brain_bio_spec = {
                        "final_state": {"cell_type_distribution": {"neuron": 10}, "total_tissues": 0, "tissue_types": []}
                    }

            # Cache the result for future runs
            # A simple function to convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            print(f"💾 Caching brain specification to {spec_cache_path}...")
            with open(spec_cache_path, 'w') as f:
                json.dump(self.brain_bio_spec, f, indent=2, default=convert_numpy)
            print("✅ Sucessfully cached brain specification.")

        self.use_hrm = use_hrm
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.embodiment = embodiment

        # Dynamically construct the brain from the biological simulation specification
        self._construct_brain_from_bio_spec(self.brain_bio_spec)

        # ----- Roadmap / Chat Task integration -----
        try:
            TASK_BRIDGE.sync()
            if hasattr(self, "goal_manager"):
                for _task in TASK_BRIDGE.get_pending_tasks():
                    self.goal_manager.enqueue(_task)
        except Exception as _exc:
            print("⚠️ Task bridge sync failed:", _exc)

        # Feature flags
        self.use_ruckig = os.environ.get("QUARK_USE_RUCKIG", "0") == "1"
        self.use_toppra = os.environ.get("QUARK_USE_TOPPRA", "0") == "1"
        self.use_ompl = os.environ.get("QUARK_USE_OMPL", "0") == "1"
        self.use_towr = os.environ.get("QUARK_USE_TOWR", "0") == "1"
        self.use_ocs2 = os.environ.get("QUARK_USE_OCS2", "0") == "1"
        self.use_visp = os.environ.get("QUARK_USE_VISP", "0") == "1"
        self.use_gtsam = os.environ.get("QUARK_USE_GTSAM", "0") == "1"
        self.use_rtabmap = os.environ.get("QUARK_USE_RTABMAP", "0") == "1"
        self.use_pcl = os.environ.get("QUARK_USE_PCL", "0") == "1"
        self.use_pin = os.environ.get("QUARK_USE_PINOCCHIO", "0") == "1"
        self.use_dart = os.environ.get("QUARK_USE_DART", "0") == "1"
        self.use_drake = os.environ.get("QUARK_USE_DRAKE", "0") == "1"
        self.use_sophus = os.environ.get("QUARK_USE_SOPHUS", "0") == "1"
        self.use_manif = os.environ.get("QUARK_USE_MANIF", "0") == "1"
        self.use_spatialmath = os.environ.get("QUARK_USE_SPATIALMATH", "0") == "1"
        # Optimizers / backends
        self.use_casadi = os.environ.get("QUARK_USE_CASADI", "0") == "1"
        self.use_osqp = os.environ.get("QUARK_USE_OSQP", "0") == "1"
        self.use_ceres = os.environ.get("QUARK_USE_CERES", "0") == "1"
        self.use_qpsolvers = os.environ.get("QUARK_USE_QPSOLVERS", "0") == "1"
        self.use_nlopt = os.environ.get("QUARK_USE_NLOPT", "0") == "1"
        self.use_ipopt = os.environ.get("QUARK_USE_IPOPT", "0") == "1"
        self.use_proxsuite = os.environ.get("QUARK_USE_PROXSUITE", "0") == "1"
        self.use_pagmo = os.environ.get("QUARK_USE_PAGMO", "0") == "1"
        self.use_pymoo = os.environ.get("QUARK_USE_PYMOO", "0") == "1"
        self.use_trajopt = os.environ.get("QUARK_USE_TRAJOPT", "0") == "1"
        self.use_topico = os.environ.get("QUARK_USE_TOPICO", "0") == "1"
        self.use_fuse = os.environ.get("QUARK_USE_FUSE", "0") == "1"

        # --- High-level Planning ---
        if self.use_hrm:
            # Provide num_actions; fall back to 16 if unknown
            num_actions = self.act_dim if self.act_dim is not None else 16
            self.hrm_planner = HRMPlanner(num_actions=num_actions)
            print("🧠 HRM Planner is ACTIVE.")
        else:
            self.hrm_planner = None
            print("🧠 HRM Planner is INACTIVE.")

        # --- Reinforcement Learning Agent ---
        # The PPO agent is initialized lazily in the first `step` call
        # because its observation dimension depends on the simulation state.
        self.ppo_agent: Optional[PPOAgent] = None

        # --- LLM-Powered Advanced Modules (Lazy Loading) ---
        # These are initialized on-demand to prevent slow startup times.
        self._llm_ik = None
        self._llm_manipulation_planner = None

        # Dataset Integration - Access to training data from external repositories
        self.dataset_integration = dataset_integration

        # Datasets are now loaded lazily.
        self._ik_training_data = None
        self._manipulation_training_data = None
        self._unified_training_data = None
        self._training_recommendations = None

        # Initialize attributes for learning loop
        self.last_state = None
        self.last_action = None
        self.last_qpos = None

        # --- Cognitive Modules (Now dynamically initialized) ---
        # The following lines are replaced by _construct_brain_from_bio_spec
        # self.meta_controller = MetaController(extrinsic_weight=0.7, intrinsic_weight=0.3)
        # self.visual_cortex = None
        # self.somatosensory_cortex = SomatosensoryCortex(num_joints=16, num_velocities=16)
        # self.sleep_engine = SleepConsolidationEngine()
        # self.salience_network = BasicSalienceNetwork(num_sources=2)
        # self.dmn = ProtoDMN()
        # self.world_model = SimpleWorldModel(num_states=10, num_actions=self.act_dim if self.act_dim is not None else 16)
        # self.limbic_system = LimbicSystem()
        # self.working_memory = WorkingMemory(capacity=10)
        # self.global_workspace = GlobalWorkspace()
        # self.cerebellum = Cerebellum(num_actuators=16)
        # self.auditory_cortex = AuditoryCortex()
        # self.oculomotor_cortex = OculomotorCortex()
        # self.motor_cortex = MotorCortex(amass_data_path="/Users/camdouglas/quark/datasets/amass/AMASS/")
        # self.language_cortex = LanguageCortex()

        # --- Fundamental Components ---
        self.brain_stem = BrainStem()

        # --- Safety ---
        self.safety_guardian = SafetyGuardian()

        # --- Proto-Cortex (SOM) ---
        # The input dimension is determined by the observation vector size.
        # This will be initialized lazily in the first step.
        self.proto_cortex = None
        self.total_steps = 0 # To be used for SOM training iterations

        # Initialize the new Motor Cortex with the correct path to the local AMASS data
        # self.motor_cortex = MotorCortex(amass_data_path="/Users/camdouglas/quark/datasets/amass/AMASS/")

        # The language cortex is a major cognitive module
        # self.language_cortex = LanguageCortex()

        # --- NEW: State for tracking learning steps ---
        self.last_action: Optional[int] = None
        self.last_pose_error: Optional[float] = None

        # This will be set by the embodiment simulation
        # self.num_possible_actions = 10

        # This will hold the "conscious" broadcast from the previous step
        self.last_broadcast = {}
        # This will hold the state of the ongoing conversation
        self.chat_history_ids = None
        self.chat_attention_mask = None

        # --- Performance Optimization ---
        self.vision_step_counter = 0
        self.vision_update_frequency = 15  # Render a new frame every 15 steps (~33Hz)
        self.last_visual_output = {"object_location": None, "processed_features": np.array([])}
        self.proto_cortex_step_counter = 0
        self.proto_cortex_update_frequency = 20 # Update the SOM every 20 steps (~25Hz)
        # Track previous observation for PPO storage to ensure consistent dimensions
        self.prev_obs: Optional[np.ndarray] = None

        # --- Profiling ---
        self.profiling_data = {}
        self.profiling_counter = 0

        # --- Adapters (instantiate based on flags) ---
        self.ruckig = RuckigAdapter(dof=16) if self.use_ruckig else None
        self.toppra = ToppraAdapter() if self.use_toppra else None
        self.ompl = OMPLAdapter(dof=16) if self.use_ompl else None
        self.towr = TOWRAdapter() if self.use_towr else None
        self.ocs2 = OCS2Adapter(dof=16) if self.use_ocs2 else None
        self.visp = ViSPAdapter() if self.use_visp else None
        # Perception / SLAM / Dynamics / Math
        self.gtsam = GTSAMAdapter() if self.use_gtsam else None
        self.rtabmap = RTABMapAdapter() if self.use_rtabmap else None
        self.pcl = PCLAdapter() if self.use_pcl else None
        # URDF path optional; keep empty default for now
        self.pin = PinocchioAdapter(urdf_path=os.environ.get("QUARK_URDF_PATH", "")) if self.use_pin else None
        self.dart = DARTAdapter() if self.use_dart else None
        self.drake = DrakeAdapter() if self.use_drake else None
        self.sophus = SophusAdapter() if self.use_sophus else None
        self.manif = ManifAdapter() if self.use_manif else None
        self.spatialmath = SpatialMathAdapter() if self.use_spatialmath else None
        # Optimizers / backends
        self.casadi = CasadiAdapter() if (self.use_casadi and CasadiAdapter) else None
        self.osqp_backend = OSQPAdapter() if (self.use_osqp and OSQPAdapter) else None
        self.ceres = CeresAdapter() if (self.use_ceres and CeresAdapter) else None
        self.qps = QPSolversAdapter() if (self.use_qpsolvers and QPSolversAdapter) else None
        self.nlopt = NLOptAdapter() if (self.use_nlopt and NLOptAdapter) else None
        self.ipopt = IpoptAdapter() if (self.use_ipopt and IpoptAdapter) else None
        self.proxsuite = ProxSuiteAdapter() if (self.use_proxsuite and ProxSuiteAdapter) else None
        self.pagmo = PagmoAdapter() if (self.use_pagmo and PagmoAdapter) else None
        self.pymoo = PyMooAdapter() if (self.use_pymoo and PyMooAdapter) else None
        self.trajopt = TrajOptAdapter() if (self.use_trajopt and TrajOptAdapter) else None
        self.topico = TopiCoAdapter() if (self.use_topico and TopiCoAdapter) else None
        self.fuse = FuseAdapter() if (self.use_fuse and FuseAdapter) else None

        # Initialize Resource Manager (lazy)
        self.resource_manager = ResourceManager(auto_scan=False)

        # --- Advanced Planner (LLM) Integration ---  NEW
        try:
            from state.quark_state_system import advanced_planner as _adv_planner
            self._planner_fn = _adv_planner.plan
            print("📝 Advanced Planner available – tasks can be generated on demand.")
        except Exception as e:
            self._planner_fn = None
            print(f"⚠️ Advanced Planner unavailable: {e}")

        # ---- Callback listener ----
        def _on_resource(event, data):
            print(f"[BrainSimulator] Event {event}: {data}")
        hub.register(_on_resource)

        # Set the initial developmental stage in relevant modules
        print("✅ Brain Simulator initialized successfully with biologically compliant architecture.")
# === MOVE TO brain/core/02_brain_simulator_init_(line_121-373).py ===

    # ------------------------------------------------------------------
    # External QA Facade
    # ------------------------------------------------------------------
    def ask(self, question: str, *, top_k: int = 3) -> str:
        """Answer a natural-language *question* using memory and LLM fallback."""
        if not question or not isinstance(question, str):
            return "I need a question to answer."

        try:
            result = self.knowledge_hub.retrieve(
                question, episodic_memory=self.hippocampus, top_k=top_k
            )
            episodes = result.get("episodes", [])
            if episodes:
                # Return the content of the top episode (simple heuristic)
                content = episodes[0].content
                if isinstance(content, dict) and "text" in content:
                    return content["text"]
                return str(content)

            llm_ans = result.get("llm_answer")
            if llm_ans:
                return llm_ans
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(f"BrainSimulator.ask failed: {e}")
        return "I'm not sure about that yet."
# === MOVE TO brain/core/03_ask_method_(line_374-401).py ===

    def _construct_brain_from_bio_spec(self, bio_spec: Dict[str, Any]):
        """
        Dynamically constructs the brain modules based on the output of the
        BiologicalSimulator. This ensures the architecture is compliant with
        AlphaGenome's developmental rules.
        """
        print("🧠 Constructing brain modules from biological specification...")

        # A mapping from biological cell types to our simulator's module classes
        # This allows for dynamic instantiation based on simulation results.
        MODULE_MAP = {
            "cortex": [
                ("neuron", VisualCortex, {"embodiment": self.embodiment}),
                ("neuron", SomatosensoryCortex, {"num_joints": 16, "num_velocities": 16}),
                ("neuron", AuditoryCortex, {}),
                ("neuron", MotorCortex, {"amass_data_path": "/Users/camdouglas/quark/datasets/amass/AMASS/"}),
                ("neuron", LanguageCortex, {}),
                ("neuron", OculomotorCortex, {}),
            ],
            "hippocampus": [("neuron", EpisodicMemory, {})],
            "basal_ganglia": [("neuron", QLearningAgent, {})], # Simplified for now
            "thalamus": [("neuron", ThalamicRelay, {})],
            "cerebellum": [("neuron", Cerebellum, {"num_actuators": 16})],
            "general": [
                (None, MetaController, {"extrinsic_weight": 0.7, "intrinsic_weight": 0.3}),
                (None, SleepConsolidationEngine, {}),
                (None, BasicSalienceNetwork, {"num_sources": 2}),
                (None, ProtoDMN, {}),
                (None, SimpleWorldModel, {"num_states": 10, "num_actions": self.act_dim if self.act_dim is not None else 16}),
                (None, LimbicSystem, {}),
                (None, WorkingMemory, {"capacity": 10}),
                (None, GlobalWorkspace, {}),
            ]
        }

        # Initialize all potential modules to None
        self.visual_cortex = None
        self.somatosensory_cortex = None
        self.auditory_cortex = None
        self.motor_cortex = None
        self.language_cortex = None
        self.oculomotor_cortex = None
        self.hippocampus = None
        self.rl_agent = None
        self.thalamus = None
        self.cerebellum = None

        final_cell_dist = bio_spec.get("final_state", {}).get("cell_type_distribution", {})

        # Instantiate modules based on the presence of cell types in the final simulated state
        if final_cell_dist.get(CellType.NEURON.value, 0) > 0:
            print("   - Neurons are present. Initializing core cognitive modules.")
            self.hippocampus = EpisodicMemory()
            # Long-term memory and synchroniser setup -------------------
            self.long_term_memory = LongTermMemory()
            self.memory_synchronizer = MemorySynchronizer(
                EpisodicMemoryStore(self.hippocampus),
                LongTermMemoryStore(self.long_term_memory),
            )
            # Persistence manager ------------------------------------------------
            self.persistence = MemoryPersistenceManager({
                "episodic": EpisodicMemoryStore(self.hippocampus),
                "ltm": LongTermMemoryStore(self.long_term_memory),
            })
            # Load previous snapshot if exists
            self.persistence.load_all()
            # Auto-save settings ---------------------------------------------
            import os
            self._persist_every_steps = int(os.environ.get("QUARK_PERSIST_EVERY", "1000"))
            self._last_persist_step = 0
            # Knowledge hub for ingestion & retrieval
            self.knowledge_hub = KnowledgeHub()
            # Initialize a simple tabular RL agent with sensible defaults
            # Use a modest discrete state space size; actions derived from embodiment action dimension
            num_actions = self.act_dim if self.act_dim is not None else 16
            self.rl_agent = QLearningAgent(num_states=100, num_actions=num_actions)
            self.thalamus = ThalamicRelay()
            self.cerebellum = Cerebellum(num_actuators=16)
            self.motor_cortex = MotorCortex(amass_data_path="/Users/camdouglas/quark/datasets/amass/AMASS/")
            self.language_cortex = LanguageCortex()
            self.somatosensory_cortex = SomatosensoryCortex(num_joints=16, num_velocities=16)
            self.auditory_cortex = AuditoryCortex()
            self.oculomotor_cortex = OculomotorCortex()


        # Instantiate general cognitive systems that are not tied to a specific cell type count
        self.meta_controller = MetaController(extrinsic_weight=0.7, intrinsic_weight=0.3)
        self.sleep_engine = SleepConsolidationEngine()
        self.salience_network = BasicSalienceNetwork(num_sources=2)
        self.dmn = ProtoDMN()
        self.world_model = SimpleWorldModel(num_states=10, num_actions=self.act_dim if self.act_dim is not None else 16)
        self.limbic_system = LimbicSystem()
        self.working_memory = WorkingMemory(capacity=10)
        self.global_workspace = GlobalWorkspace()

        print("✅ Brain modules constructed.")
# === MOVE TO brain/core/04_construct_brain_(line_402-498).py ===

    @property
    def llm_ik(self):
        """Lazy loader for LLMInverseKinematics."""
        if self._llm_ik is None:
            print("🤖 Lazily loading LLMInverseKinematics model...")
            self._llm_ik = LLMInverseKinematics()
            print("✅ LLMInverseKinematics loaded.")
        return self._llm_ik
# === MOVE TO brain/core/05_llm_ik_property_(line_499-506).py ===

    @property
    def llm_manipulation_planner(self):
        """Lazy loader for LLMManipulationPlanner."""
        if self._llm_manipulation_planner is None:
            print("🤖 Lazily loading LLMManipulationPlanner model...")
            self._llm_manipulation_planner = LLMManipulationPlanner()
            print("✅ LLMManipulationPlanner loaded.")
        return self._llm_manipulation_planner
# === MOVE TO brain/core/06_llm_manipulation_property_(line_508-515).py ===

    def _load_training_datasets_if_needed(self):
        """
        Load training datasets on-demand to avoid slow startup.
        """
        if self._ik_training_data is not None:
            return # Already loaded

        try:
            print("📚 Lazily loading training datasets...")

            # Load IK training data
            self._ik_training_data = self.dataset_integration.load_ik_training_data()
            print(f"   ✅ IK Solutions: {len(self._ik_training_data['solutions'])}")

            # Load manipulation training data
            self._manipulation_training_data = self.dataset_integration.load_manipulation_training_data()
            print(f"   ✅ Manipulation Demos: {len(self._manipulation_training_data['demonstrations'])}")

            # Create unified dataset
            self._unified_training_data = self.dataset_integration.create_unified_training_dataset()
            print(f"   ✅ Unified Dataset: {self._unified_training_data['dataset_statistics']['total_training_samples']} samples")

            # Get training recommendations
            self._training_recommendations = self.dataset_integration.get_training_recommendations()
            print(f"   💡 Training Recommendations: {len(self._training_recommendations['curriculum_order'])} phases")

        except Exception as e:
            print(f"⚠️ Warning: Could not load training datasets: {e}")
            self._ik_training_data = {"solutions": []}
            self._manipulation_training_data = {"demonstrations": []}
            self._unified_training_data = {"dataset_statistics": {"total_training_samples": 0}}
            self._training_recommendations = {"curriculum_order": []}
# === MOVE TO brain/core/07_load_training_datasets_(line_517-549).py ===

    def set_viewer(self, viewer: Any):
        """
        Sets the viewer for the VisualCortex after it has been initialized.
        """
        if viewer and self.embodiment:
            # Pass the entire embodiment, which has model, data, and viewer
            if self.visual_cortex:
                self.visual_cortex.set_embodiment(self.embodiment)
            else:
                 # If bio-sim did not produce a visual cortex, we can create one here
                 # or leave it disabled. For now, we'll create it if a viewer is provided.
                self.visual_cortex = VisualCortex(embodiment=self.embodiment)
                print("   - Visual Cortex dynamically instantiated after viewer was set.")
        else:
            print("⚠️ Warning: No viewer/embodiment provided. VisualCortex is disabled.")
# === MOVE TO brain/core/08_set_viewer_(line_550-565).py ===

    def update_stage(self, new_stage: str):
        """
        Updates the brain's developmental stage.
        """
        self.current_stage = new_stage
        print(f"🧠 Brain simulator stage updated to: {new_stage}")
# === MOVE TO brain/core/09_update_stage_(line_566-572).py ===

    def _calculate_pose_error(self, current_qpos, target_pose_full) -> Optional[float]:
        """Calculates the difference between the current pose and the target pose."""
        if target_pose_full is None or current_qpos is None:
            return None

        num_joints_to_match = min(len(current_qpos), len(target_pose_full))
        error = np.linalg.norm(
            current_qpos[:num_joints_to_match] - target_pose_full[:num_joints_to_match]
        )
        return float(error)
# === MOVE TO brain/core/10_calculate_pose_error_(line_573-583).py ===

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

        auditory_cortex_output = self.auditory_cortex.step(sensory_inputs.get("audio"))
        outputs['auditory_cortex'] = auditory_cortex_output

        current_qpos = inputs.get("qpos")
        current_qvel = inputs.get("qvel")
        somatosensory_output = self.somatosensory_cortex.step(current_qpos, current_qvel)
        outputs['somatosensory_cortex'] = somatosensory_output
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
        oculomotor_output = self.oculomotor_cortex.step()
        outputs['oculomotor_cortex'] = oculomotor_output

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
        motor_cortex_output = self.motor_cortex.step(
            ppo_goal=ppo_goal,
            hrm_subgoal=(self.hrm_planner.plan(obs) if self.hrm_planner is not None else None),
            current_qpos=current_qpos
        )

        # Raw command
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
        refined_motor_command = self.cerebellum.refine_motor_command(raw_motor_command)

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
# === MOVE TO brain/core/11_step_part1_(line_584-683).py ===
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
# === MOVE TO brain/core/12_step_part2_(line_684-757).py ===

    # ------------------------------------------------------------------
    # Task Planning API (LLM-powered)
    # ------------------------------------------------------------------
    def generate_subtasks(self, bullet: str, priority: str = "medium") -> None:
        """Generate subtasks for *bullet* and broadcast them.

        The tasks are sent to:
        • GlobalWorkspace.broadcast({"tasks": tasks})
        • MetaController.ingest_tasks(tasks)
        """
        if not self._planner_fn:
            print("⚠️ Advanced Planner unavailable – cannot generate subtasks.")
            return
        tasks = self._planner_fn(bullet)
        if not tasks:
            print("⚠️ Planner returned no tasks.")
            return
        # Stamp priority if provided
        for t in tasks:
            t.setdefault("priority", priority)
        # Broadcast to global workspace
        if hasattr(self, "global_workspace"):
            self.global_workspace.broadcast({"tasks": tasks, "source": "advanced_planner"})
        # Feed meta-controller
        if hasattr(self, "meta_controller") and hasattr(self.meta_controller, "ingest_tasks"):
            self.meta_controller.ingest_tasks(tasks)
        print(f"✅ Generated {len(tasks)} tasks for bullet: '{bullet[:40]}...' and broadcasted.")
# === MOVE TO brain/core/13_generate_subtasks_(line_836-863).py ===

    def _update_profiling_data(self, t_start, t_vision, t_sensory, t_proto, t_ppo_select, t_motor, t_cognitive, t_learning):
        self.profiling_data['vision'] = self.profiling_data.get('vision', 0) + (t_vision - t_start)
        self.profiling_data['sensory'] = self.profiling_data.get('sensory', 0) + (t_sensory - t_vision)
        self.profiling_data['proto'] = self.profiling_data.get('proto', 0) + (t_proto - t_sensory)
        self.profiling_data['ppo_select'] = self.profiling_data.get('ppo_select', 0) + (t_ppo_select - t_proto)
        self.profiling_data['motor'] = self.profiling_data.get('motor', 0) + (t_motor - t_ppo_select)
        self.profiling_data['cognitive'] = self.profiling_data.get('cognitive', 0) + (t_cognitive - t_motor)
        self.profiling_data['learning'] = self.profiling_data.get('learning', 0) + (t_learning - t_cognitive)
        self.profiling_data['total'] = self.profiling_data.get('total', 0) + (t_learning - t_start)
        self.profiling_counter += 1

        if self.profiling_counter >= 500:
            avg_total = (self.profiling_data['total'] / self.profiling_counter) * 1000
            print(f"\n--- Brain Step Profile (avg over {self.profiling_counter} steps): {avg_total:.2f} ms ---")
            for key, value in self.profiling_data.items():
                if key != 'total':
                    avg_time = (value / self.profiling_counter) * 1000
                    percentage = (avg_time / avg_total) * 100 if avg_total > 0 else 0
                    print(f"  - {key:<12}: {avg_time:.3f} ms ({percentage:.1f}%)")
            self.profiling_counter = 0
            self.profiling_data = {}
# === MOVE TO brain/core/14_update_profiling_data_(line_864-885).py ===

    def get_status(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current status of all brain modules.
        Returns:
            A dictionary containing the status of each module.
        """
        return {
            "current_stage": self.current_stage,
            "hippocampus_stats": self.hippocampus.get_memory_stats(),
            "thalamus_info_sources": len(self.thalamus.information_sources),
            "imitator_status": {
                "exploration_rate": getattr(self.rl_agent, "exploration_rate", None),
                "num_states_learned": len(self.rl_agent.q_table)
            },
            "dopamine_system_status": {
                "current_level": self.dopamine_system.current_dopamine_level
            },
            "world_model_status": {
                "model_error": np.mean(np.abs(self.world_model.transition_model))
            },
            "meta_controller_status": {
                "intrinsic_weight": self.meta_controller.intrinsic_weight
            },
            "working_memory_status": self.working_memory.get_status(),
            "global_workspace_content": self.global_workspace.get_broadcast(),
            "cerebellum_status": {
                "coordination_noise": self.cerebellum.coordination_noise
            },
            "proto_cortex_mean_activity": self.proto_cortex.mean_activity(),
            "sleep_engine_status": self.sleep_engine.get_sleep_summary(),
            "salience_network_attention": self.salience_network.get_attention_weights().tolist(),
            "dmn_status": self.dmn.get_status(),
            "dna_controller_metrics": self.dna_controller.get_performance_metrics(),
            "llm_ik_stats": self.llm_ik.get_learning_stats() if hasattr(self, 'llm_ik') else {},
            "llm_manipulation_stats": self.llm_manipulation_planner.get_planning_stats() if hasattr(self, 'llm_manipulation_planner') else {},
            "training_data_status": {
                "ik_solutions_loaded": len(self._ik_training_data.get('solutions', [])),
                "manipulation_demos_loaded": len(self._manipulation_training_data.get('demonstrations', [])),
                "unified_dataset_samples": self._unified_training_data.get('dataset_statistics', {}).get('total_training_samples', 0)
            },
            "safety_guardian_status": self.safety_guardian.get_status(),
            "brain_stem_status": self.brain_stem.get_status(),
            "adapters": {
                "ruckig": (self.ruckig.get_status() if self.ruckig else {"available": False}),
                "toppra": (self.toppra.get_status() if self.toppra else {"available": False}),
                "ompl": (self.ompl.get_status() if self.ompl else {"available": False}),
                "towr": (self.towr.get_status() if self.towr else {"available": False}),
                "ocs2": (self.ocs2.get_status() if self.ocs2 else {"available": False}),
                "visp": (self.visp.get_status() if self.visp else {"available": False}),
                "gtsam": (self.gtsam.get_status() if self.gtsam else {"available": False}),
                "rtabmap": (self.rtabmap.get_status() if self.rtabmap else {"available": False}),
                "pcl": (self.pcl.get_status() if self.pcl else {"available": False}),
                "pinocchio": (self.pin.get_status() if self.pin else {"available": False}),
                "dart": (self.dart.get_status() if self.dart else {"available": False}),
                "drake": (self.drake.get_status() if self.drake else {"available": False}),
                "sophus": (self.sophus.get_status() if self.sophus else {"available": False}),
                "manif": (self.manif.get_status() if self.manif else {"available": False}),
                "spatialmath": (self.spatialmath.get_status() if self.spatialmath else {"available": False}),
                "casadi": (self.casadi.get_status() if self.casadi else {"available": False}),
                "osqp": (self.osqp_backend.get_status() if self.osqp_backend else {"available": False}),
                "ceres": (self.ceres.get_status() if self.ceres else {"available": False}),
                "qpsolvers": (self.qps.get_status() if self.qps else {"available": False}),
                "nlopt": (self.nlopt.get_status() if self.nlopt else {"available": False}),
                "ipopt": (self.ipopt.get_status() if self.ipopt else {"available": False}),
                "proxsuite": (self.proxsuite.get_status() if self.proxsuite else {"available": False}),
                "pagmo": (self.pagmo.get_status() if self.pagmo else {"available": False}),
                "pymoo": (self.pymoo.get_status() if self.pymoo else {"available": False}),
                "trajopt": (self.trajopt.get_status() if self.trajopt else {"available": False}),
                "topico": (self.topico.get_status() if self.topico else {"available": False}),
                "fuse": (self.fuse.get_status() if self.fuse else {"available": False}),
            }
        }
# === MOVE TO brain/core/15_get_status_(line_886-957).py ===
