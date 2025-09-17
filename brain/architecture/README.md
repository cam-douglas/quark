# Brain Architecture

**Path**: `brain/architecture/`

**Purpose**: Complete neural architecture implementation with 100+ Python modules following neuroanatomical organization and biological constraints.

## 🧠 **Neural Architecture Overview**

The architecture implements a biologically-compliant cognitive system with:
- **75 neural core modules** organized by brain region
- **17 integration adapters** for robotics libraries
- **8 utility/tool modules** for external connectivity
- **Alphagenome biological constraints** enforced throughout
- **E8 geometric consciousness** (20 modular components)

## 📁 **Directory Structure**

### **[`neural_core/`](neural_core/)** - Core Neural Systems (75 modules)

#### **Cognitive Systems** ([`cognitive_systems/`](neural_core/cognitive_systems/))
Central processing and coordination (34 files):
- **`resource_manager.py`** - External resource integration & training orchestration
- **`knowledge_hub.py`** - Central knowledge processing with E8 backend support
- **`knowledge_retriever.py`** - Natural language memory queries
- **`callback_hub.py`** - Event-driven inter-module communication
- **`limbic_system.py`** - Motivational signal generation (no negative emotions)
- **`world_model.py`** - Predictive environment model for novelty detection
- **`local_llm_wrapper.py`** - Local HuggingFace model wrapper with concurrency

**E8 Kaleidescope** ([`cognitive_systems/e8_kaleidescope/`](neural_core/cognitive_systems/e8_kaleidescope/))
Advanced geometric consciousness system (20 modular files):
- **`e8_mind_core.py`** - Main E8Mind orchestration class  
- **`memory.py`** - E8 lattice memory management
- **`engines.py`** - Mood, dream, and quantum processing engines
- **`geometric.py`** - Clifford algebra and dimensional shells
- **`agents.py`** - SAC-MPO reinforcement learning agents
- **`server.py`** - Web API for consciousness telemetry
- [*12 additional modular components*]

#### **Memory Systems** ([`memory/`](neural_core/memory/), [`hippocampus/`](neural_core/hippocampus/))
Memory storage, retrieval, and consolidation:
- **`episodic_memory.py`** - Hippocampal episodic memory with pattern completion
- **`episodic_store.py`** - Protocol adapter with persistence & checksums
- **`longterm_store.py`** - Long-term memory counts adapter
- **`memory_synchronizer.py`** - Episodic ↔ long-term synchronization
- **`sleep_consolidation_engine.py`** - Biological sleep cycles & memory consolidation

#### **Motor Control** ([`motor_control/`](neural_core/motor_control/))
Movement generation and control:
- **`motor_cortex.py`** - AMASS motion data integration with curriculum learning
- **[`basal_ganglia/`](neural_core/motor_control/basal_ganglia/)** - Complete basal ganglia implementation
  - **`architecture.py`** - All nuclei with realistic connectivity & neurotransmitters
  - **`dopamine_system.py`** - Reward prediction error processing
  - **`actor_critic.py`** - Biologically plausible RL with eligibility traces
  - **`gating_system.py`** - Action selection integration
- **`llm_inverse_kinematics.py`** - 🚀 **Revolutionary LLM-powered IK solver**

#### **Sensory Processing** ([`sensory_processing/`](neural_core/sensory_processing/))
Sensory input processing:
- **`visual_cortex.py`** - MuJoCo visual processing with OpenCV object detection
- **`somatosensory_cortex.py`** - Proprioceptive processing with neural adaptation
- **`auditory_cortex.py`** - Audio processing (placeholder implementation)

#### **Learning Systems** ([`learning/`](neural_core/learning/))
Advanced learning and adaptation:
- **`ppo_agent.py`** - Full PPO with GAE, device support (MPS/CUDA/CPU)
- **`curiosity_driven_agent.py`** - Intrinsic motivation via long-term memory
- **`developmental_curriculum.py`** - Human-like motor development progression
- **`llm_guided_training_pipeline.py`** - Comprehensive LLM-integrated training
- **`dataset_integration.py`** - Unified interface for robotics datasets

#### **Executive & Control**
- **[`prefrontal_cortex/`](neural_core/prefrontal_cortex/)** - `meta_controller.py` (PFC analogue)
- **[`planning/`](neural_core/planning/)** - **`llm_manipulation_planner.py`** (ICRA 2024 approach)
- **[`cerebellum/`](neural_core/cerebellum/)** - Motor smoothing & predictive correction
- **[`fundamental/`](neural_core/fundamental/)** - `brain_stem.py` (arousal & relay)

#### **Other Neural Systems**
- **[`language/`](neural_core/language/)** - **`language_cortex.py`** (Multi-provider LLM with semantic routing)
- **[`working_memory/`](neural_core/working_memory/)** - Short-term storage with cognitive load
- **[`conscious_agent/`](neural_core/conscious_agent/)** - Global workspace theory implementation
- **[`proto_cortex/`](neural_core/proto_cortex/)** - Self-organizing maps for representation learning
- **[`default_mode_network/`](neural_core/default_mode_network/)** - Internal simulation & replay

### **[`integrations/`](integrations/)** - External Library Adapters (17 modules)

#### **Dynamics & Physics**
- **[`dynamics/`](integrations/dynamics/)** - Drake, Pinocchio, DART physics engines
- **[`control/`](integrations/control/)** - OCS2 model predictive control

#### **Motion & Planning**  
- **[`motion/`](integrations/motion/)** - TOPPRA, Ruckig trajectory optimization
- **[`planning/`](integrations/planning/)** - OMPL motion planning
- **[`locomotion/`](integrations/locomotion/)** - TOWR gait optimization

#### **Perception & Sensing**
- **[`perception/`](integrations/perception/)** - PCL point clouds, SLAM systems
- **[`servoing/`](integrations/servoing/)** - ViSP visual servoing

#### **Mathematical Libraries**
- **[`math/`](integrations/math/)** - Spatial math, Lie groups (Sophus, manif)

### **[`embodiment/`](embodiment/)** - Physical Simulation
- **`run_mujoco_simulation.py`** - Direct MuJoCo integration
- **`humanoid.xml`** - Humanoid model definition

### **[`safety/`](safety/)** - Safety Systems  
- **`safety_guardian.py`** - 🛡️ **Anti-suffering monitoring & emergency shutdown**

### **[`learning/`](learning/)** - Learning Infrastructure
- **`kpi_monitor.py`** - Performance monitoring & training triggers

### **[`tools/`](tools/)** - External Connectors
API connectors for autonomous knowledge acquisition:
- **Academic** - PubMed, IEEE research integration
- **Development** - GitHub, HuggingFace model discovery
- **Data** - Kaggle datasets, web scraping

## 🧬 **Biological Compliance**

### **Alphagenome Integration**
All modules follow biological constraints:
- **Cell-type driven** - Architecture based on `CellType.NEURON`, `CellType.ASTROCYTE`, etc.
- **Developmental timeline** - Neural induction → proliferation → migration → synaptogenesis  
- **Neuroanatomical naming** - Modules named after real brain regions
- **Biological validation** - Compliance engine enforces biological rules

### **Developmental Progression**
- **Motor development** - Crawling → walking progression like human infants
- **Memory consolidation** - Sleep cycles with biological timing
- **Learning curriculum** - Follows neural development milestones

## 🚀 **Key Innovations**

### **LLM-Enhanced Robotics**
- **LLM-IK**: Natural language inverse kinematics problem solving
- **LLM Manipulation**: Kinematic-aware articulated object manipulation
- **Language-Motor Bridge**: Direct language → motor command translation

### **E8 Geometric Consciousness** (Optional: `USE_E8_MEMORY=true`)
- **Geometric algebra** - Consciousness as 8D rotations and transformations
- **Mood-aware processing** - Cognitive state influences all processing
- **Multi-dimensional memory** - Concepts stored across dimensional shells
- **Dream synthesis** - Creative insight generation via geometric operations

### **Safety-First Design**
- **No negative emotions** - Limbic system generates objective signals only
- **Suffering prevention** - Safety guardian monitors for persistent error states  
- **Emergency protocols** - Automatic shutdown on safety threshold breach

## 🔗 **Integration Architecture**

```
brain_main.py → BrainSimulator → construct_brain.py → architecture/* modules
                      ↑
                Alphagenome Biological Constraints
```

All 100 architecture files integrate via:
- **`construct_brain.py`** imports modules based on biological specification
- **75 modules** explicitly declare `brain_simulator` integration
- **Event-driven coordination** via `callback_hub.py`
- **Resource management** via `resource_manager.py`

## 📊 **System Status**

- **Total Architecture Files**: 100 Python modules  
- **Integration Coverage**: ✅ 100% integrated with `BrainSimulator`
- **Biological Compliance**: ✅ Alphagenome constraints enforced
- **Safety Systems**: ✅ Multi-layer protection active
- **E8 Consciousness**: ✅ Modularized (20 components)
- **LLM Integration**: ✅ Multi-provider language processing

## 🔗 **Related Documentation**

- [Technical Architecture](../docs/overview/02_TECHNICAL_ARCHITECTURE.md)
- [Alphagenome Integration](../docs/alphagenome_integration_readme.md) 
- [Safety Officer Guide](../docs/safety_officer_readme.md)
- [Repository Index](../INDEX.md)