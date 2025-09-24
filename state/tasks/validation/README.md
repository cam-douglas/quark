2025-09-24 — Validation & Evidence (Single Source of Truth)

This directory contains Quark’s validation system: master checklist, per-roadmap checklists, rubric/evidence templates, and dashboard specs. It enforces the Validation Golden Rule (see .quark/rules/validation_golden_rule.mdc).

Files
- [MASTER_VALIDATION_CHECKLIST.md](./MASTER_VALIDATION_CHECKLIST.md) — Master gate across all roadmaps
- [checklists/](./checklists/) — Per-roadmap and per-milestone checklists
- [templates/](./templates/) — Rubric, evidence, and metric templates
- [dashboards/](./dashboards/) — Dashboard specs (Grafana/HELM-style)
- [evidence/](./evidence/) — Stored artefacts referenced by checklists

Status Flags
- ✔ active — up-to-date and enforced in CI
- ⚠ review — needs update or partial coverage
- ✖ deprecated — kept for traceability only

Verification
- All checklist items must link to: KPI(s) with thresholds, benchmark IDs, rubric doc, and evidence artefacts.
- CI blocks merges on missing/invalid items.
