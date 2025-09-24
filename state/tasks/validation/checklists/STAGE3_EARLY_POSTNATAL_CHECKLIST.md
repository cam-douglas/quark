# STAGE 3 — EARLY POST-NATAL VALIDATION CHECKLIST

**Source**: [management/rules/roadmap/stage3_early_post-natal_rules.md](../../../management/rules/roadmap/stage3_early_post-natal_rules.md)  
**Version**: 1.1 (2025-09-24)  
**Golden Rule**: [.quark/rules/validation_golden_rule.mdc](../../../.quark/rules/validation_golden_rule.mdc)  

## Dependencies
- [Master Validation Checklist](../MASTER_VALIDATION_CHECKLIST.md)
- [Stage 2 Fetal Checklist](./STAGE2_FETAL_CHECKLIST.md) (prerequisite)
- [Rubric Template](../templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md)
- [Evidence Directory](../evidence/)
- [V1 ODI Datasets](https://example.com/v1-odi) (external reference)

## Milestone Gates
- [ ] **Neuromodulatory gating** (ACh/GABA) for critical periods
- [ ] **Columnar microcircuits** with feedforward/feedback probes
- [ ] **SSL sensory encoder** + replay buffer functional
- [ ] **Synapse diversity palette** integrated (AMPA/NMDA/GABA/Gly/gap)
- [ ] **Cranial nerve I–XII stubs** wired to brainstem nuclei
- [ ] **Critical period closure** confirmed after GABA maturation toggle

## KPI Specifications

### synapse_density_ratio
- **Target**: 1.8±0.1 at peak (decay post-pruning)
- **Benchmark**: Primate curves
- **Measurement**: Density vs age
- **Rubric**: [../templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md](../templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### ocular_dominance_dprime
- **Target**: ≥ 1.5 during window; drop after closure
- **Benchmark**: V1 ODI dataset
- **Measurement**: d' trajectory
- **Rubric**: [../templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md](../templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### cranial_nerve_stub_pct
- **Target**: ≥ planned coverage
- **Benchmark**: Internal checklist
- **Measurement**: % wired
- **Rubric**: [../templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md](../templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### neurotransmitter_catalog_complete
- **Target**: 100% planned list
- **Benchmark**: Catalogs
- **Measurement**: Presence check
- **Rubric**: [../templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md](../templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### critical_period_closure
- **Target**: Confirmed post GABA maturation
- **Benchmark**: Internal toggle logs
- **Measurement**: Pre/post ODI change
- **Rubric**: [../templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md](../templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

## Calibration & Reproducibility Requirements
- [ ] **ECE (Expected Calibration Error)** reported ≤ 0.02
- [ ] **CI coverage** ≥ 95% for statistical claims
- [ ] **Seeds/configs/env/dataset hashes** recorded for reproducibility
- [ ] **Selective risk curves** demonstrate monotone trade-off

## Sign-off Protocol
- **Decision**: Pass / Conditional Pass / Fail
- **Reviewer**: [To be assigned]
- **Date**: [To be filled]
- **Notes**: [Critical period and synaptogenesis validation observations]
