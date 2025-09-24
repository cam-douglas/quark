# MASTER VALIDATION CHECKLIST

**Status**: REQUIRED gate for all merges  
**Owner**: Validation/QA Lead  
**Version**: 1.1 (2025-09-24)  
**Golden Rule**: [.quark/rules/validation_golden_rule.mdc](../.quark/rules/validation_golden_rule.mdc)  

## Dependencies
- [Validation Golden Rule](../.quark/rules/validation_golden_rule.mdc)
- [CI Validation Gate](../../.github/workflows/validation-gate.yml)
- [Validation Gate Script](../../tools_utilities/validation_gate.py)
- [All Domain Checklists](./checklists/)
- [Dashboard Specifications](./dashboards/)

## Required Elements Per Domain

For each roadmap domain, **ALL** required items must be present with links:

### Mandatory Components
- **KPIs** with explicit targets and comparison operators (≥, ≤, =, etc.)
- **Standard benchmarks/datasets** with ID and version specification
- **Rubric link** to domain-specific validation rubric
- **Evidence artefacts** path specification for metrics, plots, logs, configs, seeds, hashes
- **Calibration metrics** including ECE, CI coverage, selective risk
- **Reproducibility metadata** covering seeds, env, dataset hashes

## Domain Coverage Matrix

### Development Stages
- [ ] **[MASTER_ROADMAP](./checklists/MASTER_ROADMAP_CHECKLIST.md)** — checklist complete and consistent
- [ ] **[Stage 1 — Embryonic](./checklists/STAGE1_EMBRYONIC_CHECKLIST.md)** — morphogen gradients, ventricular mapping
- [ ] **[Stage 2 — Fetal](./checklists/STAGE2_FETAL_CHECKLIST.md)** — neurogenesis, lamination, migration
- [ ] **[Stage 3 — Early Post-natal](./checklists/STAGE3_EARLY_POSTNATAL_CHECKLIST.md)** — synaptogenesis, critical periods
- [ ] **[Stage 4 — Childhood](./checklists/STAGE4_CHILDHOOD_CHECKLIST.md)** — myelination, circuit refinement
- [ ] **[Stage 5 — Adolescence](./checklists/STAGE5_ADOLESCENCE_CHECKLIST.md)** — pruning, neuromodulatory maturation
- [ ] **[Stage 6 — Adult](./checklists/STAGE6_ADULT_CHECKLIST.md)** — stable networks, lifelong plasticity

### Integration Domains
- [ ] **[Main Integrations](./checklists/MAIN_INTEGRATIONS_CHECKLIST.md)** — cross-domain orchestration
- [ ] **[System Design & Orchestration](./checklists/SYSTEM_DESIGN_CHECKLIST.md)** — infrastructure, scalability
- [ ] **[Deliverables](./checklists/DELIVERABLES_CHECKLIST.md)** — artifacts, documentation
- [ ] **[Appendix C — Benchmarks & Probes](./checklists/APPENDIX_C_BENCHMARKS_CHECKLIST.md)** — comprehensive domain testing

## Global Gating Criteria

### Quality Gates
- [ ] **Calibration Standards**
  - ECE ≤ 0.02 OR documented, accepted exception with mitigation plan
  - CI coverage ≥ 95% for statistical claims
  - Selective risk curves demonstrate monotone trade-off

- [ ] **Evidence Infrastructure**
  - All artefacts stored under [state/tasks/validation/evidence/<run_id>/](./evidence/)
  - Rubrics versioned and linked from each checklist
  - Evidence paths resolve and contain required files

- [ ] **Reproducibility Standards**
  - Seeds recorded for all stochastic processes
  - Configuration files versioned and stored
  - Environment specifications captured (OS, package versions)
  - Dataset hashes computed and verified

- [ ] **CI Integration**
  - [Validation gate script](../../tools_utilities/validation_gate.py) executes successfully
  - GitHub Actions [workflow](../../.github/workflows/validation-gate.yml) blocks merges on failures
  - All required links resolve successfully

## Approval Protocol

### Sign-off Requirements
- **Technical Reviewer**: [To be assigned per domain]
- **QA Lead Approval**: Required for all domains
- **Safety Review**: Required for full brain simulation components
- **Performance Validation**: Required for implementation domains

### Final Gate Status
- **Overall Decision**: Pass / Conditional Pass / Fail
- **Conditional Items**: [List any items requiring follow-up]
- **Next Review Date**: [Scheduled follow-up if conditional]
