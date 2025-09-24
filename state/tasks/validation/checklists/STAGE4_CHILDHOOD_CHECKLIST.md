# STAGE 4 — CHILDHOOD VALIDATION CHECKLIST

**Source**: [management/rules/roadmap/stage4_childhood_rules.md](../../../management/rules/roadmap/stage4_childhood_rules.md)  
**Version**: 1.1 (2025-09-24)  
**Golden Rule**: [.quark/rules/validation_golden_rule.mdc](../../../.quark/rules/validation_golden_rule.mdc)  

## Dependencies
- [Master Validation Checklist](../MASTER_VALIDATION_CHECKLIST.md)
- [Stage 3 Early Post-natal Checklist](./STAGE3_EARLY_POSTNATAL_CHECKLIST.md) (prerequisite)
- [Rubric Template](../templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md)
- [Evidence Directory](../evidence/)
- [DTI/Myelin References](https://example.com/dti-myelin) (external)
- [WISC Cognitive Battery](https://example.com/wisc) (external)

## Milestone Gates
- [ ] **Connectomics networks** + small-world metrics established
- [ ] **Myelination to conduction-latency** model integrated
- [ ] **CSF/glymphatic flow simulator** + metabolic controller linkage
- [ ] **Graph pruning** to ≤ 30% redundant synapses
- [ ] **WISC-like cognitive battery** streaming to KPIs

## KPI Specifications

### average_conduction_latency_ms
- **Target**: ≤ 2x adult baseline
- **Benchmark**: DTI/myelin curves 2–12y
- **Measurement**: Latency model
- **Rubric**: [../templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md](../templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### small_world_sigma
- **Target**: ≥ 2.5
- **Benchmark**: Connectome refs
- **Measurement**: σ small-world
- **Rubric**: [../templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md](../templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### tractography_completion_pct
- **Target**: ≥ planned coverage
- **Benchmark**: Tract targets
- **Measurement**: % completion
- **Rubric**: [../templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md](../templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### clearance_efficiency_pct
- **Target**: ≥ target
- **Benchmark**: CSF model
- **Measurement**: % cleared
- **Rubric**: [../templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md](../templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### metabolic_budget_mJ
- **Target**: ≤ budget
- **Benchmark**: Energy logs
- **Measurement**: mJ per unit time
- **Rubric**: [../templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md](../templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

## Calibration & Reproducibility Requirements
- [ ] **ECE (Expected Calibration Error)** reported ≤ 0.02
- [ ] **CI coverage** ≥ 95% for statistical claims
- [ ] **Seeds/configs/env/dataset hashes** recorded for reproducibility
- [ ] **Small-world network metrics** validated against human connectome

## Sign-off Protocol
- **Decision**: Pass / Conditional Pass / Fail
- **Reviewer**: [To be assigned]
- **Date**: [To be filled]
- **Notes**: [Myelination and circuit refinement observations]