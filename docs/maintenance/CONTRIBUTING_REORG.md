# Contributing to the Quark Repository Re-Org

This document defines the workflow for reviewing and executing the repository
re-organization.  **All contributors must follow these rules until the `reorg`
branch is merged.**

---
## Branches
* `main` – production, no direct commits.
* `reorg_phase2` – artefact & script staging (current).
* `reorg_phase3` – live import-rewrite & file moves (opened after approval).

## PR Checklist
- [ ] References a GitHub issue `reorg-<id>`.
- [ ] Touches only files in the approved bucket for this PR.
- [ ] Includes updated mapping CSV in `audit_outputs/`.
- [ ] CI (`reorg-validation.yml`) passes.
- [ ] `reorg_validation_plan.md` unchanged or updated with new tests.

## Local Execution Steps
```
# dry-run again if you modified mappings
python tools_utilities/audit/dry_run.py A --mapping new_map.csv

# execute live rewrite & move
python tools_utilities/audit/rewrite_bucket.py new_map.csv --execute
python tools_utilities/audit/move_bucket.py new_map.csv --confirm

# update docs links
python tools_utilities/audit/link_updater.py --map audit_outputs/path_mapping_draft.csv

# run full validation
pytest -q
```

## Roll-Back Procedure
```
git reset --hard origin/reorg_phase2
git clean -fd
```

## Review Roles
* **Lead Systems Architect** – approves bucket order & mapping accuracy.
* **QA & Repro Engineer** – signs off CI + determinism.
* **Safety Officer** – scans for data/path leaks.
