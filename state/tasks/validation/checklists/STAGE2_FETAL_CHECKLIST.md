# STAGE 2 — FETAL VALIDATION CHECKLIST

**Source**: [management/rules/roadmap/stage2_fetal_rules.md](../../../management/rules/roadmap/stage2_fetal_rules.md)  
**Version**: 1.1 (2025-09-24)  
**Golden Rule**: [.quark/rules/validation_golden_rule.mdc](../../../.quark/rules/validation_golden_rule.mdc)  

## Dependencies
- [Master Validation Checklist](../MASTER_VALIDATION_CHECKLIST.md)
- [Stage 1 Embryonic Checklist](./STAGE1_EMBRYONIC_CHECKLIST.md) (prerequisite)
- [Rubric Template](../templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md)
- [Evidence Directory](../evidence/)
- [Foundation Layer Tasks](../../roadmap_tasks/foundation_layer_tasks.md)

## Milestone Gates
- [ ] **Six-layer cortical template** + thalamic relay stubs
- [ ] **Neurogenesis engine targets** achieved
- [ ] **RL radial migration** (PPO + curiosity) functional
- [ ] **Synapse schema extensions** (AMPA/NMDA/GABA/Gly/gap) implemented
- [ ] **Ablation study**: remove curiosity bonus → accuracy drops as expected
- [ ] **Migration stability**: monotonic improvement across curriculum

## KPI Specifications

### neuron_count_error_pct
- **Target**: ≤ 5%
- **Benchmark**: Internal counts vs target
- **Measurement**: Error %
- **Rubric**: [../templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md](../templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### laminar_accuracy
- **Target**: ≥ 0.80
- **Benchmark**: Labeled laminar patches
- **Measurement**: Classification accuracy
- **Rubric**: [../templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md](../templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### laminar_thickness
- **Target**: Match dMRI ranges
- **Benchmark**: dMRI fetal (30–38 w)
- **Measurement**: Thickness MAE
- **Rubric**: [../templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md](../templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### ablation_curiosity_effect
- **Target**: Δ accuracy < 0 (drop)
- **Benchmark**: Internal ablation
- **Measurement**: Δ accuracy vs baseline
- **Rubric**: [../templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md](../templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### migration_stability
- **Target**: Monotone across curriculum
- **Benchmark**: Training logs
- **Measurement**: Accuracy vs steps
- **Rubric**: [../templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md](../templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

## Calibration & Reproducibility Requirements
- [ ] **Confidence intervals** reported for all statistical claims
- [ ] **Seeds/configs/env/dataset hashes** recorded for reproducibility
- [ ] **ECE (Expected Calibration Error)** reported ≤ 0.02
- [ ] **Brier score** calculated for probabilistic predictions

## Sign-off Protocol
- **Decision**: Pass / Conditional Pass / Fail
- **Reviewer**: [To be assigned]
- **Date**: [To be filled]
- **Notes**: [Neurogenesis and migration validation observations]
