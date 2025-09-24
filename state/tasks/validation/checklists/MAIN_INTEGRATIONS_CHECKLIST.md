# MAIN INTEGRATIONS VALIDATION CHECKLIST

Source: `management/rules/roadmap/main_integrations_rules.md`
Version: 1.1 (2025-09-24)

Milestone Gates
- [ ] Core Cognitive orchestrator present and validated
- [ ] Perception & World Modeling loop validated (linear probe, cross-modal recall@k)
- [ ] Action & Agency gating validated (Sokoban, ALFWorld, risk-aware policies)
- [ ] Language cortex integration validated (MT-Bench, RAG grounding probes)
- [ ] Social cognition multi-agent integration validated (ToM, negotiation)
- [ ] Metacognition & capability management validated (calibration surfaced to policy)
- [ ] Knowledge integration via state system validated
- [ ] Robustness/adaptivity module validated (OOD, adversarial, continual)
- [ ] Creativity & exploration engine validated
- [ ] Implementation/connectome configuration validated (perf & cost dashboards)

## Domain Subsections

### Core Cognitive
- Working memory (n-back/RT, digit span)
- Formal reasoning (limited depth proving)
- Math (GSM8K/MATH subsets with process scoring)
- Puzzle success (baseline → improvement toward stretch)

### Perception & World Modeling
- Linear probe on frozen sensory encoders
- Cross-modal retrieval recall@k (text↔vision↔audio)
- Predictive coding rollouts (video PSNR/SSIM; uncertainty grows with horizon)
- Physics invariants (mass/energy conserved in toy envs)

### Action & Agency
- Sokoban multi-step success, option discovery efficiency
- ALFWorld composite task success; plan length optimality
- Risk-calibrated abstention; monotone selective risk

### Communication & Language
- MT-Bench pairwise preference evaluations
- RAG grounding exactness vs closed-book with provenance
- Process supervision: CoT self-critique quality

### Social & Cultural
- Theory of Mind (Sally-Anne/Faux-Pas)
- Norm conflicts (moral dilemmas) with explanation alignment
- Multi-agent negotiation (win-rate; Pareto efficiency)

### Robustness, Calibration & Continual
- OOD Δ (WILDS/Wild-Time); adversarial margin (AutoAttack/FGSM)
- Calibration: ECE ≤ 0.02; CI ≥ 95%; selective risk monotone
- Continual learning retention ≥ 95% with EWC+replay; forgetting Δ ≤ threshold

### Implementation & Performance
- p50/p95 latency; utilization ≥ 90%; energy per inference ↓ vs baseline
- Reliability: failover ≤ 1s; load balancing ≥ 95%; snapshot/rollback correctness

## KPI Specifications

### Core Cognitive Domain

#### reasoning_accuracy
- **Target**: Directional ↑ (long-term 99%)
- **Benchmark**: Formal suites
- **Measurement**: Prover accuracy
- **Rubric**: [../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### Perception & World Modeling Domain

#### multimodal_accuracy
- **Target**: ≥ 95%
- **Benchmark**: Multimodal tasks
- **Measurement**: Recall@k / linear probe
- **Rubric**: [../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

#### video_psnr_ssim
- **Target**: ↑ vs baseline
- **Benchmark**: Video toy sets
- **Measurement**: PSNR/SSIM vs horizon
- **Rubric**: [../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### Action & Agency Domain

#### plan_success_rate
- **Target**: ≥ target
- **Benchmark**: ALFWorld/Sokoban
- **Measurement**: % success
- **Rubric**: [../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

#### risk_calibrated_abstention
- **Target**: Monotone selective risk
- **Benchmark**: Hazard sets
- **Measurement**: Selective risk curve
- **Rubric**: [../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### Communication & Language Domain

#### mtbench_score
- **Target**: ≥ target
- **Benchmark**: MT-Bench
- **Measurement**: Composite
- **Rubric**: [../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

#### rag_grounding_exactness
- **Target**: ≥ target
- **Benchmark**: Closed-book vs RAG
- **Measurement**: Exactness/provenance
- **Rubric**: [../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### Robustness Domain

#### ECE (Expected Calibration Error)
- **Target**: ≤ 0.02
- **Benchmark**: Held-out
- **Measurement**: Calibration
- **Rubric**: [../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

#### ood_delta
- **Target**: Bounded drop
- **Benchmark**: WILDS/Wild-Time
- **Measurement**: Δ accuracy
- **Rubric**: [../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

### Performance Domain

#### latency_p95_ms
- **Target**: ≤ 100 ms
- **Benchmark**: Runtime
- **Measurement**: p95 latency
- **Rubric**: [../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

#### utilization_pct
- **Target**: ≥ 90%
- **Benchmark**: Runtime
- **Measurement**: Utilization
- **Rubric**: [../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md](../templates/RUBRIC_MAIN_INTEGRATIONS_CHECKLIST.md)
- **Evidence**: [../evidence/<run_id>/](../evidence/)

## Dependencies
- [Master Validation Checklist](../MASTER_VALIDATION_CHECKLIST.md)
- [Main Integrations Roadmap](../../../management/rules/roadmap/main_integrations_rules.md)
- [Capability Blueprints](../../../management/rules/roadmap/appendix_a_rules.md)
- [All Development Stage Checklists](./)

## Sign-off Protocol
- **Decision**: Pass / Conditional Pass / Fail
- **Technical Reviewer**: [To be assigned per domain]
- **QA Lead Approval**: Required
- **Date**: [To be filled]
- **Notes**: [Cross-domain integration observations]
