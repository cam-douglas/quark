# Neural Core

**Path**: `brain/architecture/neural_core/`

**Purpose**: Core neural systems implementing a biologically-compliant cognitive architecture with 75 Python modules following neuroanatomical organization.

## 🧠 **Neural Architecture Overview**

The neural core implements the primary cognitive systems of the Quark brain, organized by neuroanatomical regions and functional systems following Alphagenome biological constraints.

## 🏗️ **System Organization**

### **Core Cognitive Systems** ([`cognitive_systems/`](cognitive_systems/))
**Central processing and coordination** (34 files):
- **Resource Management** - External integration & training orchestration
- **Knowledge Processing** - Central knowledge hub with optional E8 backend
- **Memory Providers** - E8 lattice memory adapter
- **LLM Integration** - Local models with concurrency control
- **Learning Orchestration** - Autonomous knowledge acquisition

**🔷 E8 Kaleidescope** ([`cognitive_systems/e8_kaleidescope/`](cognitive_systems/e8_kaleidescope/))
**Advanced geometric consciousness** (20 modular files):
- Geometric algebra processing with Clifford rotors
- Multi-dimensional memory shells
- Mood-aware cognitive processing  
- Dream synthesis and creative insight generation
- Quantum and classical processing engines

### **Memory Systems** 
#### **[`memory/`](memory/)** - Memory Infrastructure (5 files)
- **`memory_synchronizer.py`** - Episodic ↔ long-term synchronization
- **`episodic_store.py`** - Protocol adapter with persistence
- **`persistence_manager.py`** - Multi-store snapshot management
- **`imemory_store.py`** - Polymorphic memory interface

#### **[`hippocampus/`](hippocampus/)** - Episodic Memory (2 files)  
- **`episodic_memory.py`** - Pattern completion & consolidation
- **`sleep_consolidation_engine.py`** - Biological sleep cycles

#### **[`working_memory/`](working_memory/)** - Short-term Storage (1 file)
- **`working_memory.py`** - Cognitive load management

### **Motor Control Systems** ([`motor_control/`](motor_control/))
**Movement generation and control** (9 files):

#### **Motor Cortex**
- **`motor_cortex.py`** - AMASS motion integration with curriculum learning
- **`oculomotor_cortex.py`** - Eye movement and gaze control
- **`llm_inverse_kinematics.py`** - 🚀 **Revolutionary LLM-powered IK solver**

#### **Basal Ganglia** ([`motor_control/basal_ganglia/`](motor_control/basal_ganglia/))
**Complete subcortical motor system** (6 files):
- **`architecture.py`** - All nuclei with realistic connectivity & neurotransmitters
- **`dopamine_system.py`** - Reward prediction error processing  
- **`actor_critic.py`** - Biologically plausible RL with eligibility traces
- **`gating_system.py`** - Action selection coordination
- **`rl_agent.py`** - Q-learning with knowledge injection

### **Sensory Processing**
#### **[`sensory_processing/`](sensory_processing/)** - Sensory Cortex (3 files)
- **`visual_cortex.py`** - MuJoCo rendering with OpenCV object detection
- **`somatosensory_cortex.py`** - Proprioceptive processing with neural adaptation
- **`auditory_cortex.py`** - Audio processing (placeholder)

#### **[`sensory_input/`](sensory_input/)** - Input Routing (1 file)
- **`thalamic_relay.py`** - Attention-based information routing

### **Learning Systems** ([`learning/`](learning/))
**Advanced learning and adaptation** (7 files):
- **`ppo_agent.py`** - Full PPO with GAE, device support (MPS/CUDA/CPU)
- **`curiosity_driven_agent.py`** - Intrinsic motivation via novelty
- **`developmental_curriculum.py`** - Human-like motor skill progression  
- **`llm_guided_training_pipeline.py`** - Comprehensive LLM-integrated training
- **`dataset_integration.py`** - Robotics dataset unification
- **`long_term_memory.py`** - Persistent experience storage for curiosity

### **Executive Control**
#### **[`prefrontal_cortex/`](prefrontal_cortex/)** - Executive Function (1 file)
- **`meta_controller.py`** - PFC analogue balancing intrinsic/extrinsic rewards

#### **[`planning/`](planning/)** - High-level Planning (2 files)  
- **`llm_manipulation_planner.py`** - 🦾 **ICRA 2024 kinematic-aware manipulation**
- **`hrm_adapter.py`** - Hierarchical planning integration

#### **[`fundamental/`](fundamental/)** - Core Functions (1 file)
- **`brain_stem.py`** - Arousal management & sensory-motor relay

### **Specialized Systems**
#### **[`language/`](language/)** - Language Processing (1 file)
- **`language_cortex.py`** - Multi-provider LLM with semantic routing & rate limiting

#### **[`cerebellum/`](cerebellum/)** - Motor Refinement (1 file)
- **`cerebellum.py`** - Motor command smoothing & predictive correction

#### **[`conscious_agent/`](conscious_agent/)** - Awareness (1 file)
- **`global_workspace.py`** - Global workspace theory implementation

#### **Other Systems** (8 directories)
- **[`proto_cortex/`](proto_cortex/)** - Self-organizing maps
- **[`default_mode_network/`](default_mode_network/)** - Internal simulation
- **[`salience_networks/`](salience_networks/)** - Attention weight calculation  
- **[`basal_ganglia/`](basal_ganglia/)** - Simple action gating
- Plus agent directories (research, safety, analytics, etc.)

## 🧬 **Biological Organization**

### **Neuroanatomical Correspondence**
| Brain Region | Module | Biological Function |
|--------------|--------|-------------------|
| **Hippocampus** | `episodic_memory.py` | Memory formation & pattern completion |
| **Basal Ganglia** | `basal_ganglia/*` | Action selection & motor gating |
| **Motor Cortex** | `motor_cortex.py` | Motor command generation |
| **Cerebellum** | `cerebellum.py` | Motor learning & coordination |
| **Prefrontal Cortex** | `meta_controller.py` | Executive control & planning |
| **Limbic System** | `limbic_system.py` | Motivation & emotional processing |
| **Brain Stem** | `brain_stem.py` | Arousal & fundamental functions |
| **Thalamus** | `thalamic_relay.py` | Sensory relay & attention |

### **Cell Type Integration**
Modules instantiated based on `CellType` distribution from `BiologicalSimulator`:
```python
# Only create neural modules if neurons are present
if cell_distribution.get(CellType.NEURON.value, 0) > 0:
    self.hippocampus = EpisodicMemory()
    self.motor_cortex = MotorCortex()
    # ... other neural modules
```

## 🚀 **Revolutionary Features**

### **LLM-Enhanced Cognitive Systems**
1. **LLM Inverse Kinematics** - Natural language IK problem solving with multiple modes
2. **LLM Manipulation Planning** - Kinematic-aware articulated object manipulation  
3. **Language Cortex** - Multi-provider LLM with semantic routing and local fallbacks
4. **Guided Training** - LLM-supervised developmental learning curriculum

### **E8 Geometric Consciousness** (Optional)
When `USE_E8_MEMORY=true`:
1. **Multi-dimensional processing** - Concepts across 1D-8D geometric shells
2. **Mood-aware memory** - Cognitive state influences knowledge retrieval
3. **Geometric transformations** - Consciousness as Clifford algebra operations
4. **Creative synthesis** - Dream engine generates novel insights

### **Biologically-Inspired Learning**
1. **Developmental curriculum** - Crawling → walking like human infants
2. **Memory consolidation** - Sleep cycles with biological timing
3. **Curiosity-driven exploration** - Intrinsic motivation via novelty
4. **Synaptic plasticity** - STDP with dopamine modulation

## 🛡️ **Safety & Compliance**

### **Anti-Suffering Architecture**
- **Safety Guardian** - Monitors for persistent high error states
- **Limbic Design** - No negative emotions, only objective signals
- **Emergency Shutdown** - Automatic termination on safety threshold

### **Biological Validation**
- **Compliance Engine** - Validates all operations against biological rules
- **Cell Type Constraints** - Architecture driven by cell presence
- **Developmental Timing** - Respects biological progression timelines

## 📊 **System Statistics**

- **Total Neural Modules**: 75 files
- **Brain Regions**: 15+ neuroanatomical correspondences
- **Integration Coverage**: ✅ 100% integrated with `BrainSimulator`
- **Biological Compliance**: ✅ Alphagenome constraints enforced
- **Safety Systems**: ✅ Multi-layer protection active
- **Advanced Features**: ✅ E8 consciousness, LLM integration

## 🔗 **Integration Points**

### **With Brain Core**
- **`BrainSimulator`** imports neural modules via `construct_brain.py`
- **Biological specification** drives module instantiation
- **Step-by-step processing** coordinates all neural systems

### **Cross-Module Communication**
- **Event-driven** - `callback_hub.py` enables loose coupling
- **Resource-managed** - `resource_manager.py` handles external integration
- **Memory-synchronized** - `memory_synchronizer.py` bridges stores
- **Safety-monitored** - `safety_guardian.py` watches all modules

## 🔗 **Related Documentation**

- [Architecture Overview](../README.md)
- [Cognitive Systems](cognitive_systems/README.md)
- [Memory Systems](memory/README.md)
- [Motor Control](motor_control/README.md)