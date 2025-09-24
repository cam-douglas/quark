# Quark Validation System Usage Guide

**Generated**: 2025-09-24  
**Status**: Active  
**Version**: 1.0

## Quick Start

### Interactive Sprint Validation
```bash
# Start interactive sprint guide
make validate

# Or directly:
python quark_validate.py sprint
```

### Quick Validation of Current Changes
```bash
# Validate git diff changes
make validate-quick

# Or:
python quark_validate.py validate
```

### CI/CD Validation
```bash
# Run validation gate (for CI)
make validate-ci

# Or:
python tools_utilities/validation_gate.py
```

## Core Concepts

### 1. Validation Pipeline Flow

```
Planning → Scope Selection → Prerequisites → Implementation → Evidence → Validation → Review → Finalize
```

### 2. File Structure

```
state/tasks/validation/
├── MASTER_VALIDATION_CHECKLIST.md    # Top-level gate
├── VALIDATION_MASTER_INDEX.md        # Navigation index
├── checklists/                       # Domain checklists
│   ├── STAGE[1-6]_*.md              # Biological stages
│   ├── MAIN_INTEGRATIONS_*.md       # Cross-domain
│   └── APPENDIX_C_BENCHMARKS.md     # Comprehensive probes
├── templates/                        # Rubric templates
├── dashboards/                       # Visualization specs
├── evidence/                         # Run artifacts
│   └── <run_id>/
│       ├── metrics.json
│       ├── config.yaml
│       ├── seeds.txt
│       ├── environment.txt
│       └── dataset_hashes.txt
└── core/                            # Python modules

```

### 3. Checklist Item Structure

Each checklist item contains:
- **Milestone Gate**: Binary readiness check
- **KPI Specification**: 
  - Target with comparator (≥, ≤, =)
  - Benchmark/dataset
  - Measurement method
  - Rubric link
  - Evidence path

### 4. Evidence Requirements

Every validation run must collect:
- `metrics.json` - KPI measurements
- `config.yaml` - Configuration
- `seeds.txt` - Random seeds
- `environment.txt` - System info
- `dataset_hashes.txt` - Data integrity
- `logs.txt` - Execution logs

## Command Reference

### Main Entry Point

```bash
python quark_validate.py [command] [options]
```

**Commands**:
- `sprint` - Interactive sprint guide
- `validate` - Validate current changes (auto-detects or uses --domain/--stage)
- `verify` - Validate specific domain (requires --domain or --stage)
- `metrics` - Display metrics
- `rubric` - Manage rubrics
- `dashboard` - Generate dashboards
- `evidence` - Manage artifacts
- `rules` - Validate rules index
- `ci` - Run full CI validation pipeline
- `help` - Show help

### Specifying What to Validate

#### Method 1: Auto-Detection (Default)
```bash
python quark_validate.py validate
# Automatically detects changes via git diff and suggests relevant domains
```

#### Method 2: Explicit Domain
```bash
# Using full domain name
python quark_validate.py verify --domain MAIN_INTEGRATIONS

# Using aliases (case-insensitive)
python quark_validate.py verify --domain foundation
python quark_validate.py verify --domain "foundation layer"
python quark_validate.py verify --domain integration
```

#### Method 3: By Stage Number
```bash
python quark_validate.py verify --stage 1  # Validates STAGE1_EMBRYONIC
python quark_validate.py verify --stage 2  # Validates STAGE2_FETAL
# ... etc up to stage 6
```

#### Method 4: Interactive Selection
```bash
python quark_validate.py sprint
# Presents interactive menu to choose scope
```

### Domain Aliases

The system recognizes these common aliases:

| Alias | Maps To |
|-------|---------|
| `foundation`, `foundation_layer` | STAGE1_EMBRYONIC |
| `embryonic` | STAGE1_EMBRYONIC |
| `fetal` | STAGE2_FETAL |
| `postnatal` | STAGE3_EARLY_POSTNATAL |
| `childhood` | STAGE4_CHILDHOOD |
| `adolescence` | STAGE5_ADOLESCENCE |
| `adult` | STAGE6_ADULT |
| `integration`, `integrations` | MAIN_INTEGRATIONS |
| `system` | SYSTEM_DESIGN |
| `benchmarks` | APPENDIX_C_BENCHMARKS |
| `deliverables` | DELIVERABLES |
| `roadmap` | MASTER_ROADMAP |

### Makefile Targets

```bash
make validate          # Interactive sprint
make validate-quick    # Quick validation
make validate-ci       # CI gate
make validate-dashboard # Generate dashboard
make validate-metrics  # Show metrics
make validate-rubrics  # Generate rubrics
make validate-full     # Complete pipeline
```

## Sprint Validation Workflow

### Phase 1: Planning
1. Review `MASTER_VALIDATION_CHECKLIST.md`
2. Select validation targets for sprint
3. Choose category (biological/integration/system)

### Phase 2: Scope Selection
1. Auto-detect scope from git diff
2. Or manually select from checklist menu
3. Scope determines which KPIs to validate

### Phase 3: Prerequisites
1. Check stage dependencies (1→6 order)
2. Verify prior stage completions
3. Review explicit dependencies

### Phase 4: Implementation & Measurement
1. Implement code changes
2. Run automated KPI measurements
3. Or enter manual measurements

### Phase 5: Evidence Collection
1. Generate run ID (timestamp)
2. Create `evidence/<run_id>/` directory
3. Collect all required artifacts

### Phase 6: Validation Gates
1. Run `validation_gate.py`
2. Check KPI targets met
3. Verify evidence completeness

### Phase 7: Review
1. Generate HTML dashboard
2. Review completion report
3. Analyze trends

### Phase 8: Finalize
1. Update master checklist
2. Create PR with results
3. CI auto-validates on merge

## Integration with Quark State System

### Activation Words
The validation system activates on these keywords:
- validate, validation
- verify, verification
- KPI, metrics
- rubric, benchmark
- calibration, evidence
- checklist, milestone, gate

### Example Integration
```python
from state.quark_state_system.validation_integration import ValidationIntegration

# Create integration
validation = ValidationIntegration(workspace_root)

# Check if should activate
if validation.should_activate(prompt_text):
    result = validation.process_validation_prompt(prompt_text)
```

## KPI Measurement Scripts

### Creating a Measurement Script
```python
#!/usr/bin/env python3
import json

def measure_my_kpi():
    # Measurement logic
    value = compute_metric()
    
    return {
        "kpi": "my_kpi_name",
        "value": value,
        "unit": "ratio",
        "status": "success"
    }

if __name__ == "__main__":
    result = measure_my_kpi()
    print(json.dumps(result))
```

### Registering in KPI Runner
Add to `kpi_runner.py`:
```python
self.kpi_scripts = {
    "my_kpi_name": "measure_my_kpi.py",
    # ...
}
```

## Dashboard Generation

### HTML Dashboard
```bash
make validate-dashboard
# Opens: state/tasks/validation/dashboards/validation_dashboard.html
```

### Grafana Integration
```bash
# Import dashboard config
cat state/tasks/validation/dashboards/grafana_dashboard.json
```

## CI/CD Integration

### GitHub Actions
The validation system is fully integrated with GitHub Actions:

```yaml
# .github/workflows/validation-gate.yml
- name: Run Full CI Validation Pipeline
  run: |
    python quark_validate.py ci
```

This runs:
1. Rules index validation
2. Validation gate checks
3. Evidence completeness verification
4. Dashboard generation
5. PR commenting with results

### Pre-commit Hook
A pre-commit hook is installed at `.git/hooks/pre-commit`:
```bash
# Runs automatically before each commit
python quark_validate.py validate  # Quick validation
python quark_validate.py rules     # Rules validation
```

### Rules Validation
```bash
# Validate rules index
python quark_validate.py rules

# Sync rules between .cursor and .quark
python quark_validate.py rules --action sync
# OR
make validate-sync
```

## Troubleshooting

### Missing Evidence Files
```bash
# Check evidence completeness
ls -la state/tasks/validation/evidence/<run_id>/
```

### Validation Gate Failures
```bash
# Run with verbose output
python tools_utilities/validation_gate.py --verbose
```

### KPI Measurement Errors
```bash
# Test individual KPI
python -c "from state.tasks.validation.core.kpi_runner import KPIRunner; 
k = KPIRunner(Path('state/tasks/validation')); 
print(k.run_kpi_measurement({'name': 'your_kpi'}))"
```

## Best Practices

1. **Always run validation before PR**
   ```bash
   make validate-ci
   ```

2. **Collect evidence for every run**
   - Even manual tests should generate evidence
   - Use consistent run IDs

3. **Update checklists immediately**
   - Mark gates when behavior implemented
   - Add evidence paths after runs

4. **Review trends regularly**
   ```bash
   make validate-dashboard
   ```

5. **Keep rubrics versioned**
   - Update version on changes
   - Document edge cases

## Advanced Usage

### Custom Validation Scope
```python
from state.tasks.validation.core.sprint_guide import SprintGuide

guide = SprintGuide(validation_root)
result = guide.quick_validate(scope="MY_CUSTOM_CHECKLIST")
```

### Batch KPI Runs
```python
from state.tasks.validation.core.kpi_runner import KPIRunner

runner = KPIRunner(validation_root)
for checklist in ["STAGE1", "STAGE2", "STAGE3"]:
    results = runner.run_all_kpis(f"{checklist}_CHECKLIST")
    print(f"{checklist}: {results}")
```

### Programmatic Evidence Collection
```python
from state.tasks.validation.core.evidence_collector import EvidenceCollector

collector = EvidenceCollector(validation_root)
run_id = collector.create_run_id()
run_dir = collector.setup_evidence_directory(run_id)

# Collect custom metrics
my_metrics = {"custom_kpi": 0.95}
collector.collect_metrics(run_dir, my_metrics)
```

## Support

For issues or questions:
1. Check `VALIDATION_MASTER_INDEX.md`
2. Review roadmap rules in `management/rules/roadmap/`
3. Run `python quark_validate.py help`
