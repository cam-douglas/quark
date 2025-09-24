# Quark Brain System

**Path**: `brain/`

**Purpose**: Complete artificial general intelligence (AGI) brain implementation combining neuroscience-inspired architecture with cutting-edge AI techniques.

## 🧠 **Overview**

The Quark Brain is a biologically-compliant cognitive architecture that integrates:
- **Neural core systems** (100+ modules) following neuroanatomical organization
- **Alphagenome biological constraints** ensuring developmental fidelity  
- **LLM-enhanced learning** with revolutionary IK solving and manipulation planning
- **E8 geometric consciousness** (optional) for advanced cognitive processing
- **Developmental learning** from crawling to complex manipulation
- **Safety-first design** with anti-suffering protocols

## 🚀 **Quick Start**

### **Run the Brain**
```bash
# Default execution (infinite mode with MuJoCo viewer)
mjpython brain_main.py

# Basic execution (infinite mode, headless)
python brain_main.py --no-viewer

# Custom frequency (still infinite)
mjpython brain_main.py --hz 30

# Finite steps (override infinite default)
mjpython brain_main.py --steps 1000

# Enable advanced E8 consciousness
USE_E8_MEMORY=true mjpython brain_main.py
```

### **Core Components**

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **Entry Point** | Main simulation controller | `brain_main.py` |
| **Core Systems** | Brain simulation engine | [`core/`](core/) |
| **Architecture** | Neural modules & integration | [`architecture/`](architecture/) |
| **Modules** | Specialized systems | [`modules/`](modules/) |
| **Tools** | Helper utilities | [`tools/`](tools/) |

## 📁 **Directory Structure**

### **[`core/`](core/)** - Brain Management Layer
High-level brain management and orchestration.
- **`brain_manager.py`** - Central component management and orchestration

### **[`simulation_engine/`](simulation_engine/)** - Brain Simulation Engine
Core simulation infrastructure and biological architecture construction.
- **`brain_simulator_init.py`** - Main `BrainSimulator` class
- **`construct_brain.py`** - Biological specification compliance
- **`emb_*` files** - Embodied simulation components
- **`step_part1/2.py`** - Simulation step processing

### **[`architecture/`](architecture/)** - Neural Architecture (100 Files)
Complete neural system following neuroanatomical organization:

#### **Neural Core** ([`neural_core/`](architecture/neural_core/))
- **Cognitive Systems** - Resource management, knowledge processing, LLM integration
- **Memory Systems** - Episodic, working, long-term memory with persistence
- **Motor Control** - Basal ganglia, motor cortex, cerebellum
- **Sensory Processing** - Visual, auditory, somatosensory cortex
- **Learning** - PPO, curiosity-driven, developmental curriculum
- **Executive** - Prefrontal cortex, limbic system, safety guardian

#### **Integration Layer** ([`integrations/`](architecture/integrations/))
External library adapters for robotics stack:
- **Dynamics** - Drake, Pinocchio, DART physics
- **Motion** - TOPPRA, Ruckig trajectory optimization  
- **Perception** - PCL, SLAM, visual servoing
- **Control** - OCS2 model predictive control

#### **Supporting Systems**
- **[`embodiment/`](architecture/embodiment/)** - MuJoCo simulation integration
- **[`learning/`](architecture/learning/)** - KPI monitoring and training triggers
- **[`safety/`](architecture/safety/)** - Safety guardian and compliance
- **[`tools/`](architecture/tools/)** - External API connectors

### **[`modules/`](modules/)** - Specialized Modules
#### **[`alphagenome_integration/`](modules/alphagenome_integration/)** - 🧬 Biological Constraints
- **DNA Controller** - AlphaGenome sequence analysis
- **Cell Constructor** - Biologically-accurate cell types  
- **Biological Simulator** - Developmental process simulation
- **Compliance Engine** - Biological constraint validation

#### **[`mathematical_integration/`](modules/mathematical_integration/)**
- **Wolfram Alpha Connector** - Mathematical computation integration

### **[`simulator/`](simulator/)** - Legacy Simulation (Deprecated)
Superseded by unified `brain_main.py` entry point.

### **[`tools/`](tools/)** - Brain Utilities
- **Task Bridge** - Roadmap integration
- **Goal Poll** - Objective tracking

## 🔬 **Biological Compliance**

All neural modules follow **Alphagenome biological constraints**:
- **Cell Type Taxonomy** - `NEURON`, `ASTROCYTE`, `OLIGODENDROCYTE`, `MICROGLIA`
- **Developmental Stages** - `NEURAL_INDUCTION` → `NEURONAL_MIGRATION` → `SYNAPTOGENESIS`
- **Architectural Rules** - Brain regions instantiated based on biological specification
- **Safety Protocols** - Anti-suffering mechanisms prevent harmful states

## ⚡ **Key Innovations**

### **LLM-Enhanced Robotics**
- **LLM Inverse Kinematics** - Natural language IK problem solving
- **LLM Manipulation Planning** - Kinematic-aware object manipulation
- **Developmental Learning** - Human-like motor skill acquisition

### **E8 Geometric Consciousness** (Optional)
When `USE_E8_MEMORY=true`:
- **Geometric Algebra** - Consciousness as 8D geometric transformations
- **Mood-Aware Processing** - Cognitive state influences knowledge processing  
- **Dream Synthesis** - Creative concept generation through geometric operations
- **Multi-dimensional Memory** - Concepts stored across dimensional shells

### **Safety-First Design**
- **Safety Guardian** - Monitors for persistent suffering states
- **Compliance Engine** - Validates all actions against biological rules
- **Emergency Shutdown** - Automatic termination on safety threshold breach

## 🔗 **Integration Points**

- **Resource Manager** - External dataset/model integration with sandbox validation
- **Knowledge Hub** - Central knowledge processing with optional E8 backend
- **Memory Synchronization** - Episodic ↔ Long-term memory bridging
- **Training Orchestration** - Automatic training triggers based on performance KPIs

## 📊 **System Status**

- **Total Files**: 100+ Python modules
- **Integration**: ✅ All modules integrated with `BrainSimulator`
- **Biological Compliance**: ✅ Alphagenome constraints enforced
- **Safety**: ✅ Multi-layer protection active
- **Testing**: ✅ Full system simulation verified

## 🔗 **Related Documentation**

- [Technical Architecture](../docs/overview/02_TECHNICAL_ARCHITECTURE.md)
- [Alphagenome Integration](../docs/alphagenome_integration_readme.md)
- [Safety Officer Guide](../docs/safety_officer_readme.md)
- [Repository Index](../INDEX.md)