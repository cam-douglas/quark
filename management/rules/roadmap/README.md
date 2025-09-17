# Quark Roadmap Suite

Path: `management/rules/roadmap/`

This directory contains the **canonical development roadmap** for the Quark project, broken into modular Markdown files.  All engineering planning, biological stages, benchmarks, safety rules, and deliverables are maintained here and consumed automatically by tooling (see `roadmap_controller.py`, `QUARK_STATE_SYSTEM.py`, CI scripts).

## Key Concepts
- **MASTER_ROADMAP.md** – Single entry-point index listing every roadmap section.  Update this file whenever you add / rename roadmap files.
- **Stage-specific files** – `stage1_embryonic_rules.md` … `stage6_adult_rules.md` track biological/engineering milestones per developmental stage.
- **Cross-cutting files** – System design, benchmark validation, deliverables, appendices A–D (risks, glossary, etc.).
- **Cursor Rules** – Each file begins with one or more HTML comments (`<!-- CURSOR RULE: ... -->`) that instruct Cursor/CI which tests or linters to run before edits.

## Directory Layout (2025-09-01)
```
roadmap/
│  MASTER_ROADMAP.md           ← Canonical index
│  README.md                   ← This file
│  roadmap_controller.py       ← Helper loader for tooling
│
├─ main_integrations_rules.md
├─ stage1_embryonic_rules.md
├─ stage2_fetal_rules.md
├─ stage3_early_post-natal_rules.md
├─ stage4_childhood_rules.md
├─ stage5_adolescence_rules.md
├─ stage6_adult_rules.md
│
├─ benchmark_validation_rules.md
├─ system_design_and_orchestration_rules.md
├─ deliverables_rules.md
│
├─ appendix_a_rules.md
├─ appendix_b_rules.md
├─ appendix_c_rules.md
├─ appendix_d_rules.md
└─ archive_superseded/          ← Read-only historical snapshots
```

## Editing Workflow
1. **Start with `MASTER_ROADMAP.md`** – ensure your section link exists.
2. Open the target roadmap file; obey its Cursor Rule tests before committing.
3. Keep each file ≤ ~400 lines for readability; create a new file if scope grows.
4. After merging, CI regenerates `ROADMAPS_INDEX.md` and validates all links.

## Tooling
- `roadmap_controller.get_all_roadmaps()` – returns structured metadata for every roadmap file.
- `QUARK_STATE_SYSTEM` – uses `MASTER_ROADMAP` to generate tasks and checkpoints.
- `pre_push_update.py` – pre-commit hook validating links and formatting.

## Links
- [Repository Root README](../../README.md)
- [Master Roadmap Index](MASTER_ROADMAP.md)
