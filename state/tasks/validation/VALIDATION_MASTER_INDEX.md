# VALIDATION MASTER INDEX

**Created**: 2025-09-24  
**Version**: 1.0  
**Purpose**: Complete directory of all validation files with dependencies and symlinks  

## Core Validation Documents

### Master Controls
- [**MASTER_VALIDATION_CHECKLIST.md**](./MASTER_VALIDATION_CHECKLIST.md) — Central gating document for all merges
- [**README.md**](./README.md) — Directory overview and usage guide
- [**INDEX.md**](./INDEX.md) — Original index file
- [**VALIDATION_MASTER_INDEX.md**](./VALIDATION_MASTER_INDEX.md) — This comprehensive index

### Golden Rule Reference
- [**Validation Golden Rule**](../.quark/rules/validation_golden_rule.mdc) — Core validation principles

## Development Stage Checklists

### Developmental Biology Stages
- [**STAGE1_EMBRYONIC_CHECKLIST.md**](./checklists/STAGE1_EMBRYONIC_CHECKLIST.md)
  - **Focus**: Morphogen gradients, ventricular mapping, meninges scaffold
  - **Dependencies**: [Foundation Layer Tasks](../roadmap_tasks/foundation_layer_tasks.md), [Allen Brain Atlas](https://atlas.brain-map.org/)
  - **Rubric**: [RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md](./templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md)

- [**STAGE2_FETAL_CHECKLIST.md**](./checklists/STAGE2_FETAL_CHECKLIST.md)
  - **Focus**: Neurogenesis, lamination, radial migration
  - **Dependencies**: Stage 1 completion, cortical template systems
  - **Rubric**: [RUBRIC_STAGE2_FETAL_CHECKLIST.md](./templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md)

- [**STAGE3_EARLY_POSTNATAL_CHECKLIST.md**](./checklists/STAGE3_EARLY_POSTNATAL_CHECKLIST.md)
  - **Focus**: Synaptogenesis, critical periods, neuromodulation
  - **Dependencies**: Stage 2 completion, SSL sensory encoders
  - **Rubric**: [RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md](./templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md)

- [**STAGE4_CHILDHOOD_CHECKLIST.md**](./checklists/STAGE4_CHILDHOOD_CHECKLIST.md)
  - **Focus**: Myelination, circuit refinement, small-world networks
  - **Dependencies**: Stage 3 completion, connectomics pipelines
  - **Rubric**: [RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md](./templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md)

- [**STAGE5_ADOLESCENCE_CHECKLIST.md**](./checklists/STAGE5_ADOLESCENCE_CHECKLIST.md)
  - **Focus**: Pruning, neuromodulatory maturation, executive function
  - **Dependencies**: Stage 4 completion, hierarchical RL systems
  - **Rubric**: [RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md](./templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md)

- [**STAGE6_ADULT_CHECKLIST.md**](./checklists/STAGE6_ADULT_CHECKLIST.md)
  - **Focus**: Stable networks, lifelong plasticity, vascular homeostasis
  - **Dependencies**: All prior stages, cloud infrastructure
  - **Rubric**: [RUBRIC_STAGE6_ADULT_CHECKLIST.md](./templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md)

## Integration & System Checklists

### Cross-Domain Integration
- [**MAIN_INTEGRATIONS_CHECKLIST.md**](./checklists/MAIN_INTEGRATIONS_CHECKLIST.md)
  - **Focus**: Domain subsections covering all capability areas
  - **Dependencies**: All development stages, capability blueprints
  - **Rubric**: [RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](./templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
  - **Subsections**:
    - Core Cognitive (working memory, reasoning, math)
    - Perception & World Modeling (linear probe, cross-modal recall)
    - Action & Agency (Sokoban, ALFWorld, risk calibration)
    - Communication & Language (MT-Bench, RAG grounding)
    - Social & Cultural (ToM, negotiation, norms)
    - Robustness & Continual Learning (OOD, adversarial, retention)
    - Implementation & Performance (latency, utilization, energy)

### System Architecture
- [**SYSTEM_DESIGN_CHECKLIST.md**](./checklists/SYSTEM_DESIGN_CHECKLIST.md)
  - **Focus**: Infrastructure, scalability, orchestration
  - **Dependencies**: Ray+Kubeflow, vLLM, distributed systems
  - **Rubric**: [RUBRIC_SYSTEM_DESIGN_CHECKLIST.md](./templates/RUBRIC_SYSTEM_DESIGN_CHECKLIST.md)

### Meta Checklists
- [**MASTER_ROADMAP_CHECKLIST.md**](./checklists/MASTER_ROADMAP_CHECKLIST.md)
  - **Focus**: Cross-file consistency and presence checks
  - **Dependencies**: All roadmap documents
  - **Rubric**: [RUBRIC_MASTER_ROADMAP_CHECKLIST.md](./templates/RUBRIC_MASTER_ROADMAP_CHECKLIST.md)

- [**DELIVERABLES_CHECKLIST.md**](./checklists/DELIVERABLES_CHECKLIST.md)
  - **Focus**: Documentation, CI/CD, observability artifacts
  - **Dependencies**: All implementation components
  - **Rubric**: [RUBRIC_DELIVERABLES_CHECKLIST.md](./templates/RUBRIC_DELIVERABLES_CHECKLIST.md)

## Comprehensive Domain Testing

- [**APPENDIX_C_BENCHMARKS_CHECKLIST.md**](./checklists/APPENDIX_C_BENCHMARKS_CHECKLIST.md)
  - **Focus**: Complete domain subsections for all capability areas
  - **Dependencies**: External benchmarks (HELM, MMLU, MT-Bench, WILDS, AutoAttack)
  - **Rubric**: [RUBRIC_APPENDIX_C_BENCHMARKS_CHECKLIST.md](./templates/RUBRIC_APPENDIX_C_BENCHMARKS_CHECKLIST.md)
  - **Domain Coverage**:
    - Core Cognitive, Memory, Reasoning
    - Perception & World Modeling
    - Action & Agency
    - Communication & Language
    - Social & Cultural Intelligence
    - Robustness, Calibration & Continual Learning
    - Networks (DMN/SN, small-world metrics)
    - Implementation Performance
    - Full Brain Simulation

## Templates & Supporting Files

### Validation Templates
- [**RUBRIC_TEMPLATE.md**](./templates/RUBRIC_TEMPLATE.md) — Base template for all rubrics
- [**EVIDENCE_TEMPLATE.md**](./templates/EVIDENCE_TEMPLATE.md) — Evidence record format
- [**KPI_MAPPING_TEMPLATE.md**](./templates/KPI_MAPPING_TEMPLATE.md) — KPI specification format

### Domain-Specific Rubrics
- [**RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md**](./templates/RUBRIC_STAGE1_EMBRYONIC_CHECKLIST.md)
- [**RUBRIC_STAGE2_FETAL_CHECKLIST.md**](./templates/RUBRIC_STAGE2_FETAL_CHECKLIST.md)
- [**RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md**](./templates/RUBRIC_STAGE3_EARLY_POSTNATAL_CHECKLIST.md)
- [**RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md**](./templates/RUBRIC_STAGE4_CHILDHOOD_CHECKLIST.md)
- [**RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md**](./templates/RUBRIC_STAGE5_ADOLESCENCE_CHECKLIST.md)
- [**RUBRIC_STAGE6_ADULT_CHECKLIST.md**](./templates/RUBRIC_STAGE6_ADULT_CHECKLIST.md)
- [**RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md**](./templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- [**RUBRIC_SYSTEM_DESIGN_CHECKLIST.md**](./templates/RUBRIC_SYSTEM_DESIGN_CHECKLIST.md)
- [**RUBRIC_MASTER_ROADMAP_CHECKLIST.md**](./templates/RUBRIC_MASTER_ROADMAP_CHECKLIST.md)
- [**RUBRIC_DELIVERABLES_CHECKLIST.md**](./templates/RUBRIC_DELIVERABLES_CHECKLIST.md)
- [**RUBRIC_APPENDIX_C_BENCHMARKS_CHECKLIST.md**](./templates/RUBRIC_APPENDIX_C_BENCHMARKS_CHECKLIST.md)

## Dashboard Specifications

### Monitoring & Visualization
- [**validation_dashboard_spec.yaml**](./dashboards/validation_dashboard_spec.yaml) — Dashboard configuration
- [**grafana_panels.json**](./dashboards/grafana_panels.json) — Grafana panel definitions

## Evidence Storage

### Artifact Repository
- [**evidence/**](./evidence/) — Evidence artifact storage directory
  - Structure: `evidence/<run_id>/`
  - Contents: metrics.json, plots/, logs/, configs/, seeds, dataset hashes

## CI/CD Integration

### Automated Validation
- [**Validation Gate Script**](../../tools_utilities/validation_gate.py) — Automated checklist parser
- [**GitHub Actions Workflow**](../../.github/workflows/validation-gate.yml) — CI validation pipeline

## External Dependencies

### Reference Datasets & Benchmarks
- **Allen Brain Atlas** (embryonic/adult references)
- **Human Connectome Project** (structural/functional connectivity)
- **HELM Evaluation Framework** (comprehensive ML evaluation)
- **MMLU & GSM8K** (cognitive capability probes)
- **MT-Bench** (dialogue quality assessment)
- **WILDS/Wild-Time** (OOD generalization testing)
- **AutoAttack/FGSM** (adversarial robustness testing)

### Related Quark Components
- [**Management Rules**](../../management/rules/) — Project governance
- [**Roadmap Tasks**](../roadmap_tasks/) — Implementation tracking
- [**State System**](../../state/) — Global orchestration
- [**Brain Architecture**](../../brain/) — Core implementation

## Usage Instructions

### For Developers
1. Start with [MASTER_VALIDATION_CHECKLIST.md](./MASTER_VALIDATION_CHECKLIST.md)
2. Navigate to relevant stage/domain checklist
3. Follow KPI specifications and evidence requirements
4. Use domain-specific rubric for evaluation
5. Store artifacts in evidence/ directory structure

### For QA/Validation Leads
1. Review master checklist for completeness
2. Ensure CI validation gate passes
3. Verify all rubrics are properly linked
4. Validate evidence artifacts exist and are accessible
5. Confirm calibration metrics meet standards

### For CI/CD Integration
1. Validation gate runs automatically on PRs
2. Blocks merges if required items missing
3. Auto-generates rubric stubs where needed
4. Validates all symlinks resolve correctly