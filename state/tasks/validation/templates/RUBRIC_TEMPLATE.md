# Rubric Template

## Metadata
**Version**: 1.0 (2025-09-24)  
**Owner**: Validation/QA Lead  
**Scope**: `<component or full-simulation>`

## 1. KPIs and Benchmarks

### KPIs (with targets)
- **KPI 1**: `<KPI name>` → Target: `<threshold>` (e.g., ECE ≤ 0.02)
- **KPI 2**: `<KPI name>` → Target: `<threshold>`
<!-- Add more KPIs as needed -->

### Benchmarks/Datasets
- **Benchmark 1**: `<Benchmark ID>` - Version: `<version>`
- **Dataset 1**: `<Dataset ID>` - Version: `<version>`
<!-- Add more benchmarks/datasets as needed -->

## 2. Scoring Criteria (Weights)

### Correctness/Accuracy
- **Weight**: `<weight>%`
- **Measured via**: `<metric>`

### Calibration/Uncertainty
- **Weight**: `<weight>%`
- **Measured via**: ECE, CI coverage, selective risk

### Robustness/OOD/Adversarial
- **Weight**: `<weight>%`
- **Measured via**: Δ accuracy, robustness margin

### Biological Fidelity (if applicable)
- **Weight**: `<weight>%`
- **Measured via**: laminar fidelity, small-worldness, ODI

### Performance/Cost
- **Weight**: `<weight>%`
- **Measured via**: p95 latency, energy per inference, utilization

**Total Weight**: 100%

## 3. Pass/Fail Gates

### Pass Criteria (ALL must be satisfied)
- ✅ KPIs meet or exceed all targets
- ✅ Calibration criteria satisfied
- ✅ Evidence artefacts complete and reproducible
- ✅ Rubric weights yield overall score ≥ Pass threshold

### Pass Threshold
- **Overall Score Required**: `≥ <threshold>%`

## 4. Edge Cases & Error Budgets

### Known Failure Modes
- **Failure Mode 1**: `<description>` - Acceptable Error: `<band>`
- **Failure Mode 2**: `<description>` - Acceptable Error: `<band>`

### Error Tolerances
- **Component A**: `±<tolerance>`
- **Component B**: `±<tolerance>`

## 5. Abstention & Selective Prediction Policy

### Abstention Conditions
- **Condition 1**: `<when to abstain>`
- **Condition 2**: `<when to abstain>`

### Expected Behavior
- Monotone selective risk curve required
- Coverage-accuracy trade-off documented

## 6. Evidence Requirements

### Required Artefacts
- 📄 `metrics.json` or `metrics.csv`
- 📊 `plots/` directory with visualizations
- ⚙️ `configs/` with all configuration files
- 🎲 `seeds.txt` with random seeds used
- 📦 `environment.txt` with full environment export
- 🔐 `dataset_hashes.txt` with content hashes

### Evidence Storage
**Path**: `state/tasks/validation/evidence/<run_id>/`

## 7. Review & Sign-off

### Reviewer Information
- **Name**: `<name>`
- **Role**: `<role>`
- **Date**: `<YYYY-MM-DD>`

### Decision
**Verdict**: `[Pass / Conditional Pass / Fail]`

### Notes
`<Any additional review comments or conditions>`
