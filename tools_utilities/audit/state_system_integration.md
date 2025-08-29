# State-System Integration Plan

This document specifies how `quark_state_system`, `tasks/`, and
`autonomous_agent.py` will migrate into the new `/state/` namespace while
preserving functionality and CI guarantees.

---
## 1. Target Layout
```
state/
├── __init__.py  # re-exports key helpers for convenience
├── quark_state_system/        # existing package
│   └── ...
├── tasks/                     # markdown + python tasks
│   └── ...
└── autonomous_agent.py        # moved file; will import from state.quark_state_system
```

## 2. Import Path Changes
| Old Import                        | New Import                         |
|----------------------------------|------------------------------------|
| `import quark_state_system ...`   | `from state import quark_state_system` |
| `from tasks...`                   | `from state.tasks ...`                 |
| `import autonomous_agent`         | `from state import autonomous_agent`   |

These mappings will be appended to `mapping_bucket_C.csv` and handled by
`rewrite_bucket.py` during the live run.

## 3. Runtime Contracts
1. `QUARK_STATE_SYSTEM.py` (in `/core/`) uses `quark_state_system` to fetch
   recommendations.  After move it will `import state.quark_state_system as qss`.
2. Existing CLI scripts (`python tasks/<task>.py`) will continue to work via a
   stub shim inserted into `tasks/__init__.py` that forwards to `state.tasks`.

## 4. CI & Tests
* **Unit Test:** ensure `state.quark_state_system.AutonomousPlanner` still
  executes.
* **Integration Test:** run `QUARK_STATE_SYSTEM.py` end-to-end and assert
  `get_next_recommendation()` returns non-empty.
* **Live Visual Test:** schedule a minimal task and display via live-stream
  plugin.

## 5. Migration Steps (Bucket C live run)
1. Execute `rewrite_bucket.py mapping_bucket_C.csv --execute` (live).
2. Run `move_bucket.py mapping_bucket_C.csv --confirm`.
3. Execute `link_updater.py --map audit_outputs/path_mapping_draft.csv`.
4. Run full test-suite & live visual validation.

---
Ready-when criteria:
* All imports resolved.
* All state tests pass.
* Agent performs at least one recommendation cycle.
