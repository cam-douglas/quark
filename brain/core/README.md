# Brain Core

**Path**: `brain/core/`

**Purpose**: High-level orchestration and management layer for the brain system.

## 🎯 **Overview**

The brain core directory contains high-level orchestration components that coordinate across all brain subsystems without disrupting their architectural organization. These components provide unified interfaces and coordination capabilities.

## 📁 **Core Components**

### **`brain_orchestrator.py`** - Master Orchestrator
- **Purpose**: High-level orchestration of all brain managers and subsystems
- **Features**:
  - Unified interface to all brain managers
  - Coordinated multi-system operations
  - Startup/shutdown orchestration
  - Cross-system operation coordination
  - Component lifecycle management

### **`component_registry.py`** - Dynamic Component Registry
- **Purpose**: Dynamic discovery and registration of brain components
- **Features**:
  - Automatic component discovery
  - Dynamic module loading
  - Component categorization
  - Manager identification
  - Lazy instantiation

## 🏗️ **Architecture Design**

```
Brain Core Orchestration Layer
==============================

┌─────────────────────────────────────────────┐
│          Brain Orchestrator                 │  ← High-level coordination
├─────────────────────────────────────────────┤
│         Component Registry                  │  ← Dynamic discovery
└─────────────────────────────────────────────┘
                     ↓
        Orchestrates and coordinates
                     ↓
┌─────────────────────────────────────────────┐
│        Brain Architecture Components        │
├─────────────────────────────────────────────┤
│ • Resource Manager    (cognitive systems)   │
│ • Knowledge Hub       (cognitive systems)   │
│ • Meta Controller     (prefrontal cortex)   │
│ • Persistence Manager (memory systems)      │
│ • Curriculum Manager  (learning systems)    │
│ • Motor Cortex        (motor control)       │
│ • Visual Cortex       (sensory processing)  │
└─────────────────────────────────────────────┘
```

## 🔧 **Key Benefits**

1. **Non-Invasive Integration** - Works with existing architecture without refactoring
2. **Unified Interface** - Single entry point for complex brain operations
3. **Coordinated Operations** - Orchestrates multi-system operations
4. **Dynamic Discovery** - Automatically finds and registers components
5. **Lazy Loading** - Components loaded only when needed

## 💡 **Usage Examples**

### Basic Orchestration
```python
from brain.core.brain_orchestrator import BrainOrchestrator

# Initialize orchestrator
orchestrator = BrainOrchestrator(brain_dir)

# Start brain systems
orchestrator.orchestrate_startup("full")

# Coordinate a complex operation
result = orchestrator.coordinate_managers(
    "store_knowledge",
    {"content": "new information", "type": "episodic"}
)
```

### Component Discovery
```python
from brain.core.component_registry import ComponentRegistry

# Create registry
registry = ComponentRegistry(brain_dir)

# Get all memory managers
memory_managers = registry.get_managers_by_category("memory")

# Load a specific component
knowledge_hub = registry.instantiate_component(
    "architecture.neural_core.cognitive_systems.knowledge_hub"
)
```

## 🔗 **Integration Points**

The orchestrator integrates with:

1. **TODO System** - Called by `state/todo/core/brain_handler.py`
2. **Brain Main** - Can be used by `brain/brain_main.py` for startup
3. **Simulation Engine** - Coordinates with `brain/simulation_engine/`
4. **All Managers** - Provides unified access to all brain managers

## 📊 **Coordinated Operations**

The orchestrator can coordinate complex operations across systems:

### Knowledge Storage
- Knowledge Hub → structures information
- Episodic Memory → stores context
- Persistence Manager → saves to disk
- Meta Controller → updates world model

### Action Planning
- Meta Controller → sets goals
- Motor Cortex → plans movements
- Cerebellum → refines timing
- Working Memory → maintains context

### Sensory Processing
- Thalamus → relays input
- Sensory Cortices → process data
- Cognitive Systems → interpret meaning
- Attention Systems → filter stimuli

### Skill Learning
- Curriculum Manager → structures progression
- Motor Systems → learn patterns
- Memory Systems → store episodes
- Meta Controller → monitors progress

## 🚀 **Future Enhancements**

- **Plugin System** - Dynamic loading of new brain components
- **State Persistence** - Save/restore entire brain state
- **Distributed Orchestration** - Coordinate across multiple machines
- **Performance Monitoring** - Real-time component performance tracking
- **Event Bus** - Publish/subscribe for component communication

## 🔗 **Related Documentation**

- [Brain Handler](../../state/todo/core/brain_handler.py) - TODO system interface
- [Architecture Overview](../architecture/README.md) - Neural architecture
- [Simulation Engine](../simulation_engine/README.md) - Core simulation
- [Main Brain README](../README.md) - Overall brain system