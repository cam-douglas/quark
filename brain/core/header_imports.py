"""Master Brain Simulator
Orchestrates all brain modules, managing their interactions and the flow of information.
This is the central hub for the integrated brain simulation.

Integration: Primary simulator entry; integrates all neural subsystems at runtime.
Rationale: Primary simulator entry point for neural runtime.
"""

import sys
import os
import numpy as np
from typing import Dict, Any, Optional
import time # Added for profiling
import json # For caching the biological specification
from brain.tools.task_bridge import TASK_BRIDGE

# Ensure parent directories are in the path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from brain.architecture.neural_core.sensory_input.thalamus.thalamic_relay import ThalamicRelay
from brain.architecture.neural_core.hippocampus.episodic_memory import EpisodicMemory
from brain.architecture.neural_core.basal_ganglia.simple_gate import ActionGate # Will be replaced
from brain.architecture.neural_core.motor_control.basal_ganglia.rl_agent import QLearningAgent
from brain.architecture.neural_core.motor_control.basal_ganglia.dopamine_system import DopamineSystem
from brain.architecture.neural_core.proto_cortex.layer_sheet import LayerSheet
from brain.architecture.neural_core.hippocampus.sleep_consolidation_engine import SleepConsolidationEngine
from brain.architecture.neural_core.salience_networks.basic_salience_network import BasicSalienceNetwork
# Updated path: default mode network moved under neural_core/default_mode_network
from brain.architecture.neural_core.default_mode_network.proto_dmn import ProtoDMN
from brain.architecture.neural_core.cognitive_systems.world_model import SimpleWorldModel
from brain.architecture.neural_core.cognitive_systems.limbic_system import LimbicSystem
from brain.architecture.neural_core.prefrontal_cortex.meta_controller import MetaController
from brain.architecture.neural_core.working_memory.working_memory import WorkingMemory
from brain.architecture.neural_core.conscious_agent.global_workspace import GlobalWorkspace
from brain.architecture.neural_core.cerebellum.cerebellum import Cerebellum
from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager  # NEW
from brain.architecture.neural_core.memory.episodic_store import EpisodicMemoryStore
from brain.architecture.neural_core.memory.longterm_store import LongTermMemoryStore
from brain.architecture.neural_core.memory.memory_synchronizer import MemorySynchronizer
from brain.architecture.neural_core.learning.long_term_memory import LongTermMemory
from brain.architecture.neural_core.memory.persistence_manager import MemoryPersistenceManager
from brain.architecture.neural_core.cognitive_systems.knowledge_hub import KnowledgeHub
from brain.architecture.neural_core.cognitive_systems.callback_hub import hub  # NEW

# AlphaGenome Integration for biologically compliant brain construction
# AlphaGenome integrations are optional and can be disabled via env.
try:
    from brain.modules.alphagenome_integration.cell_constructor import CellType
except Exception:
    class _CellTypeFallback:
        class NEURON:
            value = "neuron"
    CellType = _CellTypeFallback
from brain.architecture.neural_core.language.language_cortex import LanguageCortex
from brain.architecture.neural_core.sensory_processing.visual_cortex import VisualCortex
from brain.architecture.neural_core.sensory_processing.somatosensory_cortex import SomatosensoryCortex
from brain.architecture.neural_core.sensory_processing.auditory_cortex import AuditoryCortex
from brain.architecture.neural_core.motor_control.motor_cortex import MotorCortex
from brain.architecture.neural_core.motor_control.oculomotor_cortex import OculomotorCortex
from brain.architecture.neural_core.learning.curiosity_driven_agent import CuriosityDrivenAgent
from brain.architecture.neural_core.learning.long_term_memory import LongTermMemory
from brain.architecture.neural_core.learning.ppo_agent import PPOAgent
from brain.architecture.neural_core.learning.simple_imitator import SimpleImitator
from brain.architecture.neural_core.planning.hrm_adapter import HRMPlanner
from brain.architecture.neural_core.motor_control.llm_inverse_kinematics import LLMInverseKinematics
from brain.architecture.neural_core.planning.llm_manipulation_planner import LLMManipulationPlanner
from brain.architecture.neural_core.learning.dataset_integration import dataset_integration
from brain.architecture.safety.safety_guardian import SafetyGuardian
from brain.architecture.neural_core.fundamental.brain_stem import BrainStem
# Adapters (feature-flagged)
from brain.architecture.integrations.motion.ruckig_adapter import RuckigAdapter
from brain.architecture.integrations.motion.toppra_adapter import ToppraAdapter
from brain.architecture.integrations.planning.ompl_adapter import OMPLAdapter
from brain.architecture.integrations.locomotion.towr_adapter import TOWRAdapter
from brain.architecture.integrations.control.ocs2_adapter import OCS2Adapter
from brain.architecture.integrations.servoing.visp_adapter import ViSPAdapter
# Perception / SLAM / Dynamics / Math adapters
from brain.architecture.integrations.perception.slam.gtsam_adapter import GTSAMAdapter
from brain.architecture.integrations.perception.slam.rtabmap_adapter import RTABMapAdapter
from brain.architecture.integrations.perception.pcl_adapter import PCLAdapter
from brain.architecture.integrations.dynamics.pinocchio_adapter import PinocchioAdapter
from brain.architecture.integrations.dynamics.dart_adapter import DARTAdapter
from brain.architecture.integrations.dynamics.drake_adapter import DrakeAdapter
from brain.architecture.integrations.math.sophus_adapter import SophusAdapter
from brain.architecture.integrations.math.manif_adapter import ManifAdapter
from brain.architecture.integrations.math.spatialmath_adapter import SpatialMathAdapter
# Optimizers / backends adapters (stubs)
# These provide availability and configuration surfaces; core usage will be wired per-module.
try:
	from brain.architecture.integrations.optim.casadi_adapter import CasadiAdapter
	from brain.architecture.integrations.optim.osqp_adapter import OSQPAdapter
	from brain.architecture.integrations.optim.ceres_adapter import CeresAdapter
	from brain.architecture.integrations.optim.qpsolvers_adapter import QPSolversAdapter
	from brain.architecture.integrations.optim.nlopt_adapter import NLOptAdapter
	from brain.architecture.integrations.optim.ipopt_adapter import IpoptAdapter
	from brain.architecture.integrations.optim.proxsuite_adapter import ProxSuiteAdapter
	from brain.architecture.integrations.optim.pagmo_adapter import PagmoAdapter
	from brain.architecture.integrations.optim.pymoo_adapter import PyMooAdapter
	from brain.architecture.integrations.optim.trajopt_adapter import TrajOptAdapter
except Exception:
	CasadiAdapter = OSQPAdapter = CeresAdapter = QPSolversAdapter = NLOptAdapter = IpoptAdapter = ProxSuiteAdapter = PagmoAdapter = PyMooAdapter = TrajOptAdapter = None

# Motion extra
try:
	from brain.architecture.integrations.motion.topico_adapter import TopiCoAdapter
except Exception:
	TopiCoAdapter = None

# Perception fusion
try:
	from brain.architecture.integrations.perception.fuse_adapter import FuseAdapter
except Exception:
	FuseAdapter = None

from transformers import AutoTokenizer, AutoModelForCausalLM
from brain.tools.goal_poll import log_next_goal

log_next_goal()
