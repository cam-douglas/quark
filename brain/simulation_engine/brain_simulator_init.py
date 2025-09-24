
from typing import Any, Optional, Dict
import os
import json
import numpy as np
# Planning
from brain.architecture.neural_core.planning.hrm_adapter import HRMPlanner
from brain.architecture.neural_core.fundamental.brain_stem import BrainStem
from brain.architecture.safety.safety_guardian import SafetyGuardian
from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
from brain.architecture.neural_core.cognitive_systems.callback_hub import hub
# Task bridge for roadmap integration
from brain.tools.task_bridge import TASK_BRIDGE


class BrainSimulator:
    """The master controller for orchestrating all brain modules."""

    def __init__(self, use_hrm: bool = False, obs_dim: int = None, act_dim: int = None, embodiment: Any = None, dataset_integration: Any | None = None):
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
                    # Run a 100-step simulation to develop the brain structure
                    self.brain_bio_spec = self.bio_simulator.run_simulation(steps=100)
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

        # Dynamically construct the brain from the biological simulation specification.
        # The implementation lives in `brain/core/construct_brain.py` and is imported lazily
        # here to avoid circular-import issues with large sub-modules.
        from brain.core.construct_brain import _construct_brain_from_bio_spec

        _construct_brain_from_bio_spec(self, self.brain_bio_spec)

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

    # ------------------------------------------------------------------
    # Public API: step
    # ------------------------------------------------------------------

    def step(self, inputs: Dict[str, Any], stage: int = 0) -> Dict[str, Any]:
        """Single control step of the unified brain simulator.

        This thin wrapper delegates to the standalone step logic kept in
        `brain/core/step_part1.py`, which in turn calls `step_part2.py`.

        Keeping the heavy logic outside the class avoids circular-import
        issues while allowing the entry-point to call `BrainSimulator.step()`
        directly.
        """
        from brain.core import step_part1  # local import to avoid startup cost

        return step_part1.step(self, inputs, stage)

    # ------------------------------------------------------------------
    # Dataset loading (placeholder)
    # ------------------------------------------------------------------

    def _load_training_datasets_if_needed(self) -> None:
        """Lazy dataset loader stub.

        In the original modular design this method pulled AMASS/IK datasets
        only on first use.  For now we keep a no-op placeholder so the step
        logic can call it safely.  Future work: integrate `self.dataset_integration`
        once that subsystem is revived.
        """
        return
