# Consolidated Task Directory

This folder contains the **machine-readable task lists** used by the Quark state system.

| File | Purpose |
|------|---------|
| `tasks_high.yaml`   | High-priority, blocking tasks |
| `tasks_medium.yaml` | Medium-priority tasks |
| `tasks_low.yaml`    | Nice-to-have / background tasks |
| `tasks_archive.yaml`| Completed / obsolete tasks |

Each YAML file is a list of task objects:
```yaml
- id: TASK_001
  title: Integrate basal ganglia model
  priority: high        # high | medium | low
  status: pending       # pending | in_progress | completed | archived
  roadmap_stage: Stage-N2
  description: >
    Port the latest basal ganglia micro-circuit and connect to cortex layer 5 …
  source_path: state/tasks/archive/basal_ganglia_todo.md  # traceability
```

Raw historical task markdowns are moved to `state/tasks/archive/` in Phase 2.
