# Quark State System – Architectural Overview

This document captures the **nine-step execution pipeline** that governs every
Quark session.  Keeping it in `/docs/architecture.md` ensures both human
contributors and Cursor agents share a single source of truth.

---
## 1. Entry-Point Command Parsing  (`QUARK_STATE_SYSTEM.py`)
* Accepts natural-language or CLI keywords (`status`, `recommendations`,
  `tasks`, `to-do’s`, `next steps`, etc.).
* Routes the request either to the **Recommendation Path** or the
  **Active-Driver Path**.

## 2. Recommendation Path  (`state/quark_state_system/quark_recommendations.py`)
* Hot-reloads `task_loader` to ensure task YAML files are current.
* Pulls a roadmap snapshot via `management.rules.roadmap.roadmap_controller`.
* Syncs roadmap items into task YAMLs and optionally explodes
  "🚧 In Progress" sections into fine-grained tasks.
* Returns a formatted guidance block containing `next_tasks`.

## 3. Active-Driver Path  (`state/quark_state_system/quark_driver.py`)
* Instantiates once per session; owns a persistent `AutonomousAgent` and its
  `PromptGuardian` compliance engine.
* Calls `_print_todo_snapshot()` at the very start of every `process_prompt`
  cycle (rule-enforced).

## 4. Prompt Handling
* Detects `continuous N` → runs N roadmap or ad-hoc goals sequentially until
  the phase ends.
* Detects generic `proceed`, `execute`, `continue`, `evolve` → hands off to the
  agent.
* For specific prompts → generates a provisional action and sends it to the
  guardian.

## 5. Compliance Validation  (`PromptGuardian`)
* Verifies the proposed `action_type` against biological, safety, and repo
  rules.  Blocks non-compliant actions.

## 6. Autonomous Execution  (`AutonomousAgent`)
* Fetches the next actionable roadmap goal.
* Decomposes the goal into sub-tasks; performs code edits, tests, docs.
* Calls `todo_write` on every branch; marks goals complete only after
  validation.

## 7. Roadmap Sync
* Upon success, reports progress to the Roadmap Controller which updates the
  YAML task files (high / medium / low).

## 8. Timing & Feedback
* `quark_driver.process_prompt` refreshes the current goal, prints a timing
  breakdown via `utilities.performance_utils`, and waits for the next user
  command.

## 9. Continuous Mode
* Can run an entire roadmap phase unattended while still logging each step’s
  latency and compliance status in the foreground.

---
### Key Take-Aways
* Two parallel modes: read-only **Guidance** and state-changing **Active Driver**.
* Strict rule enforcement gates every prompt, task, and code change.
* Tasks and roadmap items are synced on every invocation—no stale guidance.
* Continuous mode can execute full phases while remaining transparent.
