# state

Path: `state`

Purpose: The `state` package is Quark’s **runtime orchestration layer**.  It owns all
road-map/task management logic, prompt-safety validation, autonomous execution
pipelines, cloud-storage helpers and reporting scripts that keep the rest of the
repository in sync.

Key runtime components now imported automatically by `brain/brain_main.py`:

| Component | Role in execution |
|-----------|------------------|
| `quark_state_system.autonomous_agent.AutonomousAgent` | Executes the next highest-priority roadmap task in a background thread. |
| `quark_state_system.prompt_guardian.PromptGuardian` | Validates each roadmap prompt / proposed action against bio-safety & compliance rules before the agent acts. |
| `quark_state_system.quark_driver.QuarkDriver` | Provides an API (`process_prompt`) for interactive sessions, re-using the agent & guardian. |
| `quark_state_system.quantum_decision_engine` + `quantum_router` | Decide & route heavy compute tasks to AWS Braket when quantum advantage is likely. |
| `quark_state_system.task_loader` / `goal_manager` | Load YAML task lists and expose `next_goal()` for the brain to consume. |

Ops / maintenance utilities (CLI scripts):

* `upload_heavy_directories.py` — pushes large `models/` & `datasets/` trees to S3.
* `sync_quark_state.py` — rewrites state docs so dates & statuses remain consistent.
* `generate_state_report.py` — generates the markdown state dashboard.

These tools are not required at runtime but remain useful for CI and manual
maintenance.

> **Tip:** Run `python -m state.quark_state_system.upload_heavy_directories --help` for any CLI helper.

## Subdirectories
- __pycache__/
- memory/
- quark_state_system/  # runtime state engine
- tasks/
  - roadmap_tasks/  # detailed roadmap task documentation

## Files
- README.md
- quark_state_system/README.md  # details
- __init__.py
- _bootstrap.py

## Links
- [Root README](README.md)
- [Repo Index](repo_index.json)
