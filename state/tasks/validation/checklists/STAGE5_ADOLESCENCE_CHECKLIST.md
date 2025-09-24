# STAGE 5 — ADOLESCENCE VALIDATION CHECKLIST

**Source**: [management/rules/roadmap/stage5_adolescence_rules.md](../../../management/rules/roadmap/stage5_adolescence_rules.md)  
**Version**: 1.1 (2025-09-24)  
**Golden Rule**: [.quark/rules/validation_golden_rule.mdc](../../../.quark/rules/validation_golden_rule.mdc)  

## Dependencies
- [Master Validation Checklist](../MASTER_VALIDATION_CHECKLIST.md)
- [Stage 4 Childhood Checklist](./STAGE4_CHILDHOOD_CHECKLIST.md) (prerequisite)
- [Rubric Template](../templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md)
- [Evidence Directory](../evidence/)
- [Stroop/WCST Task References](https://example.com/stroop-wcst) (external)

## Milestone Gates
- [ ] **Finalize pruning** to ~50% peak with OGD stability
- [ ] **DMN/SN/DAN switching controller** operational
- [ ] **Neuromodulatory maturation** (DA/NE/5-HT/ACh) complete
- [ ] **Hierarchical RL planner** validated on Stroop & WCST
- [ ] **Gene markers** + synaptic proteome annotations integrated

## KPI Specifications

### pruning_completion_pct
- **Target**: ≥ 98%
- **Benchmark**: Internal pruning target
- **Measurement**: % completion
- **Rubric**: [../templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md](../templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### stroop_accuracy
- **Target**: ≥ 95%
- **Benchmark**: Stroop task
- **Measurement**: Accuracy
- **Rubric**: [../templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md](../templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### risk_adjusted_return
- **Target**: ≥ baseline_human
- **Benchmark**: RL tasks
- **Measurement**: Utility
- **Rubric**: [../templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md](../templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### neuromod_tuning_pct
- **Target**: ≥ planned coverage
- **Benchmark**: Receptor map
- **Measurement**: % tuned
- **Rubric**: [../templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md](../templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### gene_marker_coverage_pct
- **Target**: ≥ planned coverage
- **Benchmark**: Markers table
- **Measurement**: % coverage
- **Rubric**: [../templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md](../templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

## Calibration & Reproducibility Requirements
- [ ] **ECE (Expected Calibration Error)** reported ≤ 0.02
- [ ] **CI coverage** ≥ 95% for statistical claims
- [ ] **Seeds/configs/env/dataset hashes** recorded for reproducibility
- [ ] **Executive function metrics** validated against adolescent cohorts

## Sign-off Protocol
- **Decision**: Pass / Conditional Pass / Fail
- **Reviewer**: [To be assigned]
- **Date**: [To be filled]
- **Notes**: [Pruning and executive function maturation observations]