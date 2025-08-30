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
