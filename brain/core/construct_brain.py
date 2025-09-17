from __future__ import annotations

"""Helper used by `BrainSimulator` to build cognitive modules from biological spec."""

# Standard Library
from typing import Any, Dict

# Neural core modules (sensory)
from brain.architecture.neural_core.sensory_processing.visual_cortex import VisualCortex
from brain.architecture.neural_core.sensory_processing.somatosensory_cortex import SomatosensoryCortex
from brain.architecture.neural_core.sensory_processing.auditory_cortex import AuditoryCortex

# Motor / language / eye movement
from brain.architecture.neural_core.motor_control.motor_cortex import MotorCortex
from brain.architecture.neural_core.language.language_cortex import LanguageCortex
from brain.architecture.neural_core.motor_control.oculomotor_cortex import OculomotorCortex

# Memory & persistence
from brain.architecture.neural_core.hippocampus.episodic_memory import EpisodicMemory
from brain.architecture.neural_core.memory.longterm_store import LongTermMemory
from brain.architecture.neural_core.memory.memory_synchronizer import MemorySynchronizer
from brain.architecture.neural_core.memory.episodic_store import EpisodicMemoryStore
from brain.architecture.neural_core.memory.longterm_store import LongTermMemoryStore
from brain.architecture.neural_core.memory.persistence_manager import MemoryPersistenceManager

# Knowledge hub
from brain.architecture.neural_core.cognitive_systems.knowledge_hub import KnowledgeHub

# Reinforcement / basal ganglia
from brain.architecture.neural_core.motor_control.basal_ganglia.rl_agent import QLearningAgent

# Other cognitive / control modules
from brain.architecture.neural_core.sensory_input.thalamus.thalamic_relay import ThalamicRelay
from brain.architecture.neural_core.cerebellum.cerebellum import Cerebellum

from brain.architecture.neural_core.prefrontal_cortex.meta_controller import MetaController
from brain.architecture.neural_core.hippocampus.sleep_consolidation_engine import SleepConsolidationEngine
from brain.architecture.neural_core.salience_networks.basic_salience_network import BasicSalienceNetwork
from brain.architecture.neural_core.default_mode_network.proto_dmn import ProtoDMN
from brain.architecture.neural_core.cognitive_systems.world_model import SimpleWorldModel
from brain.architecture.neural_core.cognitive_systems.limbic_system import LimbicSystem
from brain.architecture.neural_core.working_memory.working_memory import WorkingMemory
from brain.architecture.neural_core.conscious_agent.global_workspace import GlobalWorkspace

# Shared enums / definitions
from brain.modules.alphagenome_integration.cell_constructor import CellType


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
    neuron_count = final_cell_dist.get(CellType.NEURON.value, 0)
    print(f"   - Neuron count in biological spec: {neuron_count}")
    
    if neuron_count > 0:
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
    else:
        print("   - ⚠️ WARNING: No neurons in biological specification!")
        print("   - Neural modules (including language cortex) will not be initialized.")
        print("   - This preserves biological constraints but limits functionality.")


    # Instantiate general cognitive systems that are not tied to a specific cell type count
    self.meta_controller = MetaController(extrinsic_weight=0.7, intrinsic_weight=0.3)
    self.sleep_engine = SleepConsolidationEngine()
    self.salience_network = BasicSalienceNetwork(num_sources=2)
    self.dmn = ProtoDMN()
    self.world_model = SimpleWorldModel(num_states=10, num_actions=self.act_dim if self.act_dim is not None else 16)
    self.limbic_system = LimbicSystem()
    self.working_memory = WorkingMemory(capacity=10)
    self.global_workspace = GlobalWorkspace()

    # --- AlphaGenome Integration ---
    try:
        import os  # Add missing import
        from brain.modules.alphagenome_integration.dna_controller import create_dna_controller
        from brain.modules.alphagenome_integration.compliance_engine import ComplianceEngine

        # Initialize DNA controller with default config
        self.dna_controller = create_dna_controller()
        self.compliance_engine = ComplianceEngine()
        print("🧬 AlphaGenome DNA Controller and Compliance Engine integrated")

    except Exception as e:
        print(f"⚠️ AlphaGenome integration unavailable: {e}")
        self.dna_controller = None
        self.compliance_engine = None

    print("✅ Brain modules constructed.")
