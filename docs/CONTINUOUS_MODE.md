# Continuous Phase Execution Mode

Quark’s chat-driven **continuous mode** lets you chain multiple development
steps in a single command without pausing for each individual task.

```text
"continuous"             → run up to 5 tasks of the current chat-phase
"continuous 3"           → run 3 tasks then halt
"proceed continuous"     → alias; same as above (default 5)
"continue continously 2" → minor misspellings are tolerated
```

## How it Works

1. The chat middleware forwards the user message to `QuarkDriver`.
2. `QuarkDriver` detects any word containing **continuous/continous/**… and
   extracts an optional integer `N`.
3. It calls `run_phase_tasks(N)` which:
   * Invokes `AutonomousAgent.execute_next_goal()` for each task.
   * Retries a failing task up to **3 ×** before skipping.
   * Stops early if the agent reports no actionable tasks.
4. Once `N` tasks finish (or nothing remains) the driver prints a summary and
   waits for the next chat instruction.

Errors inside a task **do not abort** the phase loop – they are logged and
retried automatically. Only after three consecutive failures is the task
skipped and the loop continues.

## CLI Alias

You can trigger the same behaviour from the command line:

```bash
python QUARK_STATE_SYSTEM.py continuous 3
```

Without a number, the limit defaults to **5**.

---
*See `state/quark_state_system/quark_driver.py` for implementation details.*
