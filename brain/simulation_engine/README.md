# Brain Simulation Engine

**Path**: `brain/simulation_engine/` (formerly `brain/core/`)

**Purpose**: Core brain simulation engine and biological architecture construction system.

## 🎯 **Overview**

The brain core provides the fundamental simulation infrastructure that orchestrates all neural architecture modules according to biological specifications from Alphagenome integration.

## 🔧 **Core Components**

### **Primary Systems**

| File | Purpose | Description |
|------|---------|-------------|
| **`brain_simulator_init.py`** | 🧠 **Main Controller** | The master `BrainSimulator` class that orchestrates all neural modules |
| **`construct_brain.py`** | 🧬 **Biological Builder** | Constructs brain architecture based on Alphagenome cell type distributions |
| **`brain_simulator_adapter.py`** | 📊 **Metrics Wrapper** | Ensures simulation steps return reward and loss metrics |

### **Simulation Lifecycle**

| File | Purpose | Description |
|------|---------|-------------|
| **`emb_init_sim_and_reset.py`** | ⚡ **Initialization** | MuJoCo simulation setup and reset logic |
| **`step_part1.py`** | 🔄 **Processing Part 1** | First half of simulation step (sensory processing) |
| **`step_part2.py`** | 🔄 **Processing Part 2** | Second half of simulation step (motor output) |
| **`emb_run_and_main.py`** | 🚀 **Execution** | Main simulation loop coordination |

### **Specialized Processing**

| File | Purpose | Description |
|------|---------|-------------|
| **`calculate_pose_error.py`** | 📐 **Pose Analysis** | Calculate pose error for imitation learning |
| **`emb_curriculum_logic.py`** | 🎓 **Curriculum** | Developmental learning progression logic |
| **`update_stage.py`** | 📈 **Progression** | Curriculum stage advancement |
| **`generate_subtasks.py`** | 🎯 **Task Planning** | Decompose high-level goals into subtasks |

### **LLM Integration Properties**

| File | Purpose | Description |
|------|---------|-------------|
| **`llm_ik_property.py`** | 🤖 **IK Integration** | LLM-powered inverse kinematics property access |
| **`llm_manipulation_property.py`** | 🦾 **Manipulation** | LLM manipulation planner property access |

### **System Utilities**

| File | Purpose | Description |
|------|---------|-------------|
| **`ask_method.py`** | 💬 **Query Interface** | Natural language query method for brain state |
| **`get_status.py`** | 📊 **Status Reporting** | Comprehensive system status collection |
| **`set_viewer.py`** | 👁️ **Viewer Setup** | MuJoCo viewer configuration |
| **`load_training_datasets.py`** | 📚 **Data Loading** | Training dataset integration |
| **`update_profiling.py`** | ⚙️ **Performance** | Performance profiling and optimization |

### **Import Management**

| File | Purpose | Description |
|------|---------|-------------|
| **`header_imports.py`** | 📦 **Core Imports** | Standard simulation imports |
| **`emb_header_imports.py`** | 📦 **Embodied Imports** | MuJoCo-specific imports |
| **`emb_init_and_io.py`** | 🔧 **I/O Setup** | Input/output system initialization |

## 🧬 **Biological Architecture Construction**

### **`construct_brain.py` Process**
```python
def _construct_brain_from_bio_spec(self, bio_spec: Dict[str, Any]):
    # 1. Extract cell type distribution from AlphaGenome simulation
    final_cell_dist = bio_spec["final_state"]["cell_type_distribution"]
    
    # 2. Instantiate modules based on biological presence
    if final_cell_dist.get(CellType.NEURON.value, 0) > 0:
        # Initialize neural modules only if neurons present
        self.hippocampus = EpisodicMemory()
        self.motor_cortex = MotorCortex(...)
        # ... other neural modules
    
    # 3. Always instantiate general cognitive systems
    self.meta_controller = MetaController(...)
    self.limbic_system = LimbicSystem()
    # ... executive systems
```

### **Module Mapping**
```python
MODULE_MAP = {
    "cortex": [VisualCortex, SomatosensoryCortex, MotorCortex, LanguageCortex],
    "hippocampus": [EpisodicMemory],  
    "basal_ganglia": [QLearningAgent],
    "thalamus": [ThalamicRelay],
    "cerebellum": [Cerebellum],
    "general": [MetaController, LimbicSystem, WorkingMemory, ...]
}
```

## ⚡ **BrainSimulator Integration**

### **Initialization Chain**
1. **`brain_main.py`** → **`BrainSimulator`** → **`construct_brain.py`**
2. **Biological specification** drives architecture instantiation
3. **34 architecture modules** imported and configured
4. **Alphagenome compliance** validated throughout

### **Simulation Loop**
```python
# Each simulation step processes:
sensory_input → neural_modules → motor_output
      ↑              ↑              ↓
 embodiment    brain_architecture   actions
```

## 🛡️ **Safety & Compliance**

### **Biological Constraints**
- **Cell type validation** - Only instantiate modules for present cell types
- **Developmental timing** - Respect biological development timelines
- **Neural naming** - Use neuroanatomical terminology
- **Constraint validation** - Compliance engine enforces rules

### **Safety Systems**
- **Safety Guardian** - Monitor for persistent error states  
- **Emergency shutdown** - Automatic termination on safety threshold
- **Biological compliance** - Validate against Alphagenome constraints

## 🔗 **Key Relationships**

- **→ Architecture**: Imports and orchestrates all 100 architecture modules
- **→ Alphagenome**: Uses biological specification for architecture decisions
- **→ MuJoCo**: Integrates with physical simulation for embodiment
- **→ Brain Main**: Called by `brain_main.py` as primary entry point

## 📊 **System Metrics**

- **Architecture Modules Integrated**: 34/100 directly imported
- **Biological Compliance**: ✅ Alphagenome constraints enforced
- **Safety Systems**: ✅ Multi-layer protection active
- **Simulation Performance**: ✅ Real-time capable with MuJoCo integration

## 🔗 **Related Documentation**

- [Main Brain README](../README.md)
- [Architecture Overview](../architecture/README.md)
- [Alphagenome Integration](../modules/alphagenome_integration/README.md)
