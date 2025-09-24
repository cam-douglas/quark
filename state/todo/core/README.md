# TODO Core System

**Path**: `state/todo/core/`

**Purpose**: Core system orchestration and management for the Quark TODO system.

## 🎯 **Overview**

The TODO core directory contains all the central management modules that power the unified TODO system. It serves as the orchestration layer between the user-facing `todo.py` interface and all Quark subsystems.

## 📁 **Core Modules**

### **Context & Command Processing**
- **`context_analyzer.py`** - Natural language command parsing and intent detection
- **`command_router.py`** - Routes commands to appropriate subsystem handlers
- **`state_manager.py`** - Manages persistent TODO system state
- **`workflow_orchestrator.py`** - Orchestrates multi-step workflows

### **Subsystem Handlers**
- **`brain_handler_wrapper.py`** - Wrapper for brain system operations
- **`simulation_handler.py`** - General simulation management
- **`training_handler.py`** - Model training pipeline management
- **`deployment_handler.py`** - Deployment operations (GCP, Docker, Local)
- **`documentation_handler.py`** - Documentation generation and updates
- **`benchmarking_handler.py`** - Performance benchmarking (with GCP focus)

### **Launcher Scripts**
- **`state_system_launcher.py`** - Quark State System entry point
- **`tasks_launcher.py`** - Task management system entry point
- **`validate_launcher.py`** - Validation system entry point

## 🔗 **Integration Architecture**

```
User Command → todo.py
                 ↓
         ContextAnalyzer
                 ↓
         CommandRouter
                 ↓
    Subsystem Handlers/Launchers
                 ↓
         Target Systems
```

## 🚀 **Command Flow Example**

```bash
# User types: "todo simulate cerebellum"

1. ContextAnalyzer:
   - Parses: system='brain', action='simulate', params={'component': 'cerebellum'}

2. CommandRouter:
   - Routes to: brain_handler_wrapper.py

3. BrainHandler Wrapper:
   - Delegates to: brain/core/brain_handler.py

4. Brain Handler:
   - Executes: cerebellum simulation
```

## 📊 **Subsystem Connections**

| Handler | Target System | Primary Location |
|---------|--------------|------------------|
| `brain_handler_wrapper` | Brain Core | `brain/core/brain_handler.py` |
| `state_system_launcher` | Quark State | `state/quark_state_system/` |
| `tasks_launcher` | Task Manager | `state/tasks/core/` |
| `validate_launcher` | Validation | `state/tasks/validation/core/` |
| `training_handler` | Training | `brain/gcp_training_manager.py` |
| `deployment_handler` | Deployment | `scripts/deploy_*.py` |

## 🧠 **Natural Language Processing**

The `ContextAnalyzer` supports patterns for:
- Task management: "plan new task", "work on X", "track progress"
- Validation: "validate foundation", "check metrics", "dashboard"
- Brain simulation: "simulate cerebellum", "brain status"
- Training: "train model --stage 1", "training status"
- Deployment: "deploy to gcp", "deployment logs"
- Documentation: "generate docs", "update readme"
- Benchmarking: "benchmark performance", "profile cpu"
- Workflows: "daily standup", "sprint review"

## 🔧 **Adding New Subsystems**

To add a new subsystem:

1. Create a handler in `state/todo/core/`:
   ```python
   class NewSystemHandler:
       def route_command(self, action: str, params: Dict) -> int:
           # Implementation
   ```

2. Add patterns to `context_analyzer.py`:
   ```python
   self.patterns[r'\bnew_pattern\b'] = ('new_system', 'action')
   ```

3. Add routing in `command_router.py`:
   ```python
   elif system == 'new_system':
       from .new_system_handler import NewSystemHandler
       handler = NewSystemHandler(self.project_root)
       return handler.route_command(action, params)
   ```

## 🔗 **Related Documentation**

- [TODO System Overview](../../../todo.py)
- [Brain Core](../../../brain/core/README.md)
- [Validation System](../../tasks/validation/VALIDATION_GUIDE.md)
- [Task Management](../../tasks/core/README.md)
