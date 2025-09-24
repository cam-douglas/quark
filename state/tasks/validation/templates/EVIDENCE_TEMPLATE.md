# Evidence Record

## Metadata
**Run ID**: `<yyyy-mm-dd_hhmmss_identifier>`  
**Date**: `<YYYY-MM-DD>`  
**Owner**: `<name>`  
**Related Roadmap File(s)**: `<file paths>`  
**Milestone(s)**: `<milestone identifiers>`

## KPIs & Targets
- **KPI 1**: `<KPI name>` → Target: `<value>`
- **KPI 2**: `<KPI name>` → Target: `<value>`
<!-- Add more KPIs as needed -->

## Benchmarks/Datasets
- **Benchmark 1**: `<benchmark id/version>`
- **Dataset 1**: `<dataset name/version>`
<!-- Add more benchmarks/datasets as needed -->

## Configuration
- **Seeds**: `<list of random seeds>`
- **Config Files**: `<paths to configuration files>`
- **Environment**: `<hash/specification>`
- **Dataset Hashes**: `<content-addressed hashes>`

## Results

### Metrics (with 95% CI)
- **Metric 1**: `<metric name>` = `<value>` [`<low>`, `<high>`]
- **Metric 2**: `<metric name>` = `<value>` [`<low>`, `<high>`]

### Calibration
- **ECE**: `<value>`
- **CI Coverage**: `<value>%`
- **Selective Risk**: `<curve reference>`

### Robustness
- **OOD Δ**: `<value>`
- **Adversarial Margin**: `<value>`

## Artefacts
- **Metrics**: `state/tasks/validation/evidence/<run_id>/metrics.json`
- **Plots**: `state/tasks/validation/evidence/<run_id>/plots/`
- **Logs**: `state/tasks/validation/evidence/<run_id>/logs/`
- **Configs**: `state/tasks/validation/evidence/<run_id>/configs/`

## Notes
- **Anomalies**: `<any anomalies observed>`
- **Failures**: `<any failures encountered>`
- **Justifications**: `<explanations for deviations>`

## Verdict
**Decision**: `[Pass / Conditional Pass / Fail]`

## Sign-offs
- **Reviewer**: `<name>` - Date: `<YYYY-MM-DD>`
- **QA Lead**: `<name>` - Date: `<YYYY-MM-DD>`
