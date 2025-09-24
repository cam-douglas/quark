# STAGE 1 — EMBRYONIC VALIDATION CHECKLIST

**Source**: [management/rules/roadmap/stage1_embryonic_rules.md](../../../management/rules/roadmap/stage1_embryonic_rules.md)  
**Version**: 1.1 (2025-09-24)  
**Golden Rule**: [.quark/rules/validation_golden_rule.mdc](../../../.quark/rules/validation_golden_rule.mdc)  

## Dependencies
- [Master Validation Checklist](../MASTER_VALIDATION_CHECKLIST.md)
- [Rubric Template](../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md)
- [Evidence Directory](../evidence/)
- [Foundation Layer Tasks](../../roadmap_tasks/foundation_layer_tasks.md)

## Milestone Gates
- [ ] **Morphogen gradients** → coarse 3-axis voxel map with region labels
- [ ] **Ventricular system map** (lateral/third/fourth/aqueduct)
- [ ] **Meninges scaffold integrity**
- [ ] **Brainstem subdivision labels** (midbrain/pons/medulla)
- [ ] **Field physics plausibility** (PDE residuals; monotonic gradients)
- [ ] **Network structure metrics** (small-worldness, hubs)
- [ ] **Oscillation/ERP probes** under simple stimuli

## KPI Specifications

### segmentation_dice
- **Target**: ≥ baseline 0.267 (improve from baseline)
- **Benchmark**: Allen Brain Atlas embryonic
- **Measurement**: Region-wise Dice/VOE
- **Rubric**: [../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md](../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### experimental_accuracy
- **Target**: ≥ 0.705 acceptable
- **Benchmark**: ABA subsets
- **Measurement**: Accuracy with CI
- **Rubric**: [../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md](../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### grid_resolution_mm
- **Target**: Achieved (record resolution)
- **Benchmark**: Internal spec
- **Measurement**: Mesh/grid report
- **Rubric**: [../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md](../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### meninges_mesh_integrity
- **Target**: No self-intersections; manifold
- **Benchmark**: Mesh sanity
- **Measurement**: Topology checks
- **Rubric**: [../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md](../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### field_pde_residual
- **Target**: ≤ threshold; monotonic checks pass
- **Benchmark**: Internal PDE diagnostics
- **Measurement**: Residual norms; gradient tests
- **Rubric**: [../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md](../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### small_world_hub_match
- **Target**: ≥ 90% match to refs
- **Benchmark**: Human Connectome refs
- **Measurement**: σ, centrality, participation
- **Rubric**: [../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md](../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### oscillation_bands_present
- **Target**: Expected delta/theta bands
- **Benchmark**: Evoked/spontaneous
- **Measurement**: Spectral peaks; ERP latency
- **Rubric**: [../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md](../templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

## Calibration & Reproducibility Requirements
- [ ] **Reliability curves** computed and documented
- [ ] **Brier score** calculated for probabilistic predictions
- [ ] **ECE (Expected Calibration Error)** reported ≤ 0.02
- [ ] **Seeds/configs/env/dataset hashes** recorded for reproducibility

## Sign-off Protocol
- **Decision**: Pass / Conditional Pass / Fail
- **Reviewer**: [To be assigned]
- **Date**: [To be filled]
- **Notes**: [Any additional observations or conditions]
