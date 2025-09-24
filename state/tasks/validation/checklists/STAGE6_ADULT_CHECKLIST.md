# STAGE 6 — ADULT VALIDATION CHECKLIST

**Source**: [management/rules/roadmap/stage6_adult_rules.md](../../../management/rules/roadmap/stage6_adult_rules.md)  
**Version**: 1.1 (2025-09-24)  
**Golden Rule**: [.quark/rules/validation_golden_rule.mdc](../../../.quark/rules/validation_golden_rule.mdc)  

## Dependencies
- [Master Validation Checklist](../MASTER_VALIDATION_CHECKLIST.md)
- [Stage 5 Adolescence Checklist](./STAGE5_ADOLESCENCE_CHECKLIST.md) (prerequisite)
- [All Prior Stage Checklists](./)(prerequisites)
- [Rubric Template](../templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md)
- [Evidence Directory](../evidence/)
- [HELM/MMLU/MT-Bench](https://example.com/helm-mmlu) (external benchmarks)

## Milestone Gates
- [ ] **Whole-brain integration** (message bus <10 ms, retries)
- [ ] **Cloud infrastructure** (Ray+Kubeflow, 99.9% uptime)
- [ ] **Vascular model** + BOLD simulation validated
- [ ] **BBB-pericyte transport policy** operational
- [ ] **Sleep consolidation** + nightly replay functional
- [ ] **Tractography templates** complete; ion channel & receptor catalogue linkages

## KPI Specifications

### agi_domain_score_avg
- **Target**: ≥ 90%
- **Benchmark**: HELM/MMLU/MT-Bench
- **Measurement**: Composite
- **Rubric**: [../templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md](../templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### energy_per_synaptic_event_nJ
- **Target**: ≤ 0.5
- **Benchmark**: Energy logs
- **Measurement**: nJ/event
- **Rubric**: [../templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md](../templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### uptime_pct
- **Target**: ≥ 99.9%
- **Benchmark**: Runtime monitors
- **Measurement**: Uptime
- **Rubric**: [../templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md](../templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### vascular_perfusion_accuracy_pct
- **Target**: ≥ target
- **Benchmark**: BOLD comparisons
- **Measurement**: % accuracy
- **Rubric**: [../templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md](../templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### bbb_transport_fidelity_pct
- **Target**: ≥ target
- **Benchmark**: Transport assays
- **Measurement**: % fidelity
- **Rubric**: [../templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md](../templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

## Calibration & Reproducibility Requirements
- [ ] **ECE (Expected Calibration Error)** reported ≤ 0.02
- [ ] **CI coverage** ≥ 95% for statistical claims
- [ ] **Seeds/configs/env/dataset hashes** recorded for reproducibility
- [ ] **Full AGI capability metrics** calibrated across all domains

## Sign-off Protocol
- **Decision**: Pass / Conditional Pass / Fail
- **Reviewer**: [To be assigned]
- **Date**: [To be filled]
- **Notes**: [Stable networks and lifelong plasticity observations]