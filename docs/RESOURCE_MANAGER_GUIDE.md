# Quark Resource Manager – Design Guide

> **Status:** Draft (Phase 1)

## Overview
Quark’s *Resource Manager* (RM) is a library callable from `brain_simulator` that
handles discovery, validation, placement, and live integration of external
resources (datasets, models, code, configs).  It is designed to be pluggable and
safe, operating through a sandbox for untrusted packages and providing
real-time callbacks to running simulations.

---

## Component Diagram
```mermaid
flowchart TD
    subgraph Runtime
        BS[BrainSimulator]
        EBS[EmbodiedBrainSimulation]
    end
    BS & EBS -->|observer| CBH[CallbackHub]
    CBH --> RM[ResourceManager]
    RM --> Reg[Registry (YAML)]
    RM --> Sandbox
    RM --> AutoScan
    RM -->|integrates| FS[(Filesystem)]
    AutoScan -->|watches| FS
    Sandbox -->|executes| TestHarness
```

---

## Resource Types → Canonical Directories
| Resource type | Examples | Target Directory |
|--------------|----------|------------------|
| Dataset (≥200 MB) | `.zip`, `.tar`, parquet, netCDF | `data/datasets/<name>/` |
| Trained Model | `.pt`, `.ckpt`, `.onnx`, `.safetensors` | `data_knowledge/models_artifacts/<model>/` |
| Simulator Plugin | `.py`, package directory | `brain/externals/<package>/` |
| Config / Param | `.yaml`, `.toml` | `management/configurations/<domain>/` |
| Documentation | `.md`, `.rst` | `documentation/` |
| Misc <200 MB | any | `data/misc/` |

---

## Registry YAML Schema
```yaml
id: 8b1a9953d1a
path: /abs/path/original/file
integrated_path: /quark/data/datasets/myset/file.zip
size_bytes: 123456789
hash: deadbeef…   # SHA-256 of content
resource_type: dataset
license: MIT
added_at: 2025-08-30T10:15:00Z
notes: Optional free-text
```

---

## Configuration & Environment Variables
* Default config file searched in `management/configurations/resource_manager/default.yaml`.
* Override with `QUARK_RM_CONFIG=/custom/path.yaml`.
* Auto-scan toggle: `QUARK_RM_AUTOSCAN=1`.

---

## Placement Rules (v0.1)
1. If `size_bytes > 200 MB` → treat as *large* and place under `data/`.
2. Infer domain by filename keywords (dataset, model, mesh, memory).
3. Fall back to `misc/`.
4. Refuse integration if license **unknown** and manual approval pending.

---

## Error Handling & Logging
* Python `logging` integration; RM logger name `quark.resource_manager`.
* Log file at `logs/resource_manager.log` (rotating, 10 MB × 5).
* Severity mapping: DEBUG (dev), INFO (success path), WARNING (recoverable), ERROR + raise (failed integration).

---

## Dependencies
Add to `pyproject.toml`:
```toml
pyyaml = "^6.0"
license-checker = "^0.9"  # Optional for license detection
```

---

## Plugin Interface (v0.1)
```python
from abc import ABC, abstractmethod

class ResourcePlugin(ABC):
    """Hook for domain-specific integration."""
    @abstractmethod
    def can_handle(self, meta: dict) -> bool: ...
    @abstractmethod
    def integrate(self, meta: dict) -> None: ...
```
Plugins registered via `entry_points.group = "quark.resource_plugins"` in
`pyproject.toml`.

---

## Next Steps
* Implement Auto-Scan worker (Phase 2).
* Harden sandbox & license detection (Phase 3).
* Wire callback hub into simulators (Phase 4).
* Add tests & CI (Phase 5).

---

## Local LLM Integration (v0.1)
Quark can load *local* HuggingFace-format language models (e.g., `llama2_7b_chat_uncensored`) without network calls.  Drop the model folder under `data/models/<name>` and register it:

```python
from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
rm = ResourceManager()
rid = rm.register_resource("/abs/path/to/model_dir", {"type": "model", "name": "llama2_7b_chat_uncensored"})
```

The registry entry exposes `integrated_path`, which `LocalLLMWrapper` consumes to create a transformers pipeline.  The Advanced Planner automatically discovers the first registered model whose `name` matches its candidate list, ensuring offline task-planning capability.

Profile limits (batch=1, ≤2 concurrent requests) are enforced in the wrapper to avoid GPU/CPU oversubscription.

---

## Sandbox & License Policy (v0.1)

### Sandbox Validation
When a new Python code resource (< 200 MB) is approved, the Resource Manager
runs a lightweight sandbox compile test:

1. Copies the file into a temporary directory.
2. Executes `python -m py_compile <file>` inside the temp dir.
3. Enforces timeout (`cpu_limit_sec`, default 120 s) and memory cap
   (`mem_limit_mb`, default 1024 MB).
4. Captures stdout/stderr → stored at
   `logs/resource_manager/sandbox/<resource_id>/compile.log`.
5. If the compile step fails or times out, integration is aborted and the
   resource remains un-integrated.

> Road-map: replace with Docker/Firejail isolation and full unit-test runs.

### License Detection & Gate

A naïve detector scans the first 20 lines of the resource (or LICENSE file) for
SPDX tags or keywords:

* Allowed by default: `MIT`, `Apache-2.0`, `BSD-3-Clause`
* Blocked by default: `GPL-3.0`

Blocked licenses trigger a **manual approval requirement** (set
`force=True` in metadata or use a future CLI flag).  The policy is configured
in `management/configurations/resource_manager/default.yaml` under
`licensing.allowed` / `licensing.blocked`.

---
