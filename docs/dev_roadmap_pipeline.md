# Developer Guide – Roadmap → Task Pipeline

This short guide explains how a new roadmap item becomes an actionable task in the Quark State System.

## 1  Add / Edit a Roadmap Document
• Create or edit a Markdown/YAML file under `management/rules/roadmaps/`.
• Make sure the first heading (`# Title …`) is present.
• Optionally add a line containing `Current Status:` or an emoji status (✅, 🚧, 📋 Planned).

## 2  Ask Quark to “update roadmap”
From chat *or* shell:
```bash
python QUARK_STATE_SYSTEM.py "update roadmap"
```
Steps performed:
1. Regenerates `ROADMAPS_INDEX.md`.
2. Derives a status snapshot.
3. Calls `task_loader.sync_with_roadmaps()`, appending new tasks to YAMLs.

## 3  Check Recommendations
```bash
python QUARK_STATE_SYSTEM.py recommendations
```
Quark lists the top pending tasks (high → low priority).

## 3.1 Auto-Expansion & Task Bridge

When `recommendations` or the brain simulator requests tasks, the **Task Bridge** performs:
1. Sync roadmap YAMLs → pending tasks.
2. If a task title exceeds ~12 words, it is auto-split into subtasks via the local uncensored-Llama model (see `advanced_planner.py`).
3. The subtasks are queued to the `goal_manager`; leftovers are pushed back so nothing is lost.
4. On completion the simulator/agent calls `goal_manager.complete(id)` → `TASK_BRIDGE.mark_done()` → appends `DONE` in the originating roadmap bullet and updates status.

No manual step is required—just mark tasks complete in chat (“task complete”).

## 3.2 Ad-Hoc Chat Tasks

If you ask, “Are these roadmap tasks or ad-hoc tasks?” and answer “ad-hoc”, Quark creates a new `chat_tasks_<title>_N.yaml` under `state/tasks/`. Use:
```bash
python QUARK_STATE_SYSTEM.py create-chat-tasks "My quick fixes" fix_database bug_123
```
To view progress:
```bash
python QUARK_STATE_SYSTEM.py task-status
```
Completed items move to `state/tasks/tasks_archive.yaml` automatically.

## 4  Mark Completion
Once a task is done you can:
```bash
python QUARK_STATE_SYSTEM.py complete <task_id>
```
(Completion helper TBD) or edit the YAML to `status: completed`.

## 5  Troubleshooting
• If a roadmap link renders “TODO: file not found”, create or rename the referenced file.
• Duplicate protection: syncing twice will not create duplicate tasks – ensured by `task_loader.sync_with_roadmaps()`.

Happy roadmap-driven development! 📈

## Pre-Push Hook

Install once:
```bash
chmod +x tools_utilities/scripts/pre_push_update.py
ln -s ../../tools_utilities/scripts/pre_push_update.py .git/hooks/pre-push
```
What it does on every `git push`:
1. Regenerates `ROADMAPS_INDEX.md` and validates links in `master_roadmap.md`.
2. Syncs roadmap snapshot → YAML tasks.
3. Updates the README “Roadmap Status” block with an abstract & metrics.
4. Runs existing README generator + large-file checks.
If any step fails, the push is aborted so you can fix issues locally first.

### Link Validation
The pre-push hook (and CI workflow) run `tools_utilities/scripts/link_validate.py` to ensure every internal markdown link resolves to an existing file.  If a link is broken, the push/CI job fails and lists the offending source file + target.  Fix the path or create a stub before retrying.
