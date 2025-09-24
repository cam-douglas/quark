# APPENDIX C — BENCHMARKS & PROBES VALIDATION CHECKLIST

**Source**: [management/rules/roadmap/appendix_c_rules.md](../../../management/rules/roadmap/appendix_c_rules.md)  
**Version**: 1.2 (2025-09-24)  
**Golden Rule**: [.quark/rules/validation_golden_rule.mdc](../../../.quark/rules/validation_golden_rule.mdc)  

## Dependencies
- [Master Validation Checklist](../MASTER_VALIDATION_CHECKLIST.md)
- [Main Integrations Checklist](./MAIN_INTEGRATIONS_CHECKLIST.md) (cross-reference)
- [All Stage Checklists](./) (prerequisites for comprehensive testing)
- [Rubric Template](../templates/RUBRIC_APPENDIX_C_BENCHMARKS_CHECKLIST.md)
- [Evidence Directory](../evidence/)
- [External Benchmarks](#external-benchmark-references)

## Domain Subsections

### Core Cognitive
- [ ] Working memory (n-back, span) — targets defined, rubric linked, evidence present
- [ ] Attention (attentional blink, Stroop) — targets, rubric, evidence
- [ ] Cognitive flexibility (task-switch, reversal learning) — targets, rubric, evidence
- [ ] Formal logic (first-order proving) — targets, rubric, evidence
- [ ] Math (GSM8K/MATH subsets) — targets, rubric, evidence
- [ ] Program induction (Tyrell SAT) — targets, rubric, evidence

### Memory
- [ ] Episodic recall (pattern separation) — targets, rubric, evidence
- [ ] Consolidation (sleep vs wake) — Δ accuracy, rubric, evidence
- [ ] Semantic integration (schema effect) — targets, rubric, evidence

### Perception & World Modeling
- [ ] Linear probe + cross-modal recall@k (text↔vision↔audio) — targets, rubric, evidence
- [ ] Predictive coding rollouts: video PSNR/SSIM; uncertainty grows with horizon — targets, rubric, evidence
- [ ] Differentiable physics surrogates: invariants respected — targets, rubric, evidence

### Action & Agency
- [ ] Hierarchical puzzles (Sokoban) — targets, rubric, evidence
- [ ] Long-horizon tool use (ALFWorld) — targets, rubric, evidence
- [ ] Risk-calibrated abstention; selective risk monotone — targets, rubric, evidence

### Communication & Language
- [ ] MT-Bench pairwise preference evals — targets, rubric, evidence
- [ ] RAG grounding exactness/provenance; closed-book vs RAG — targets, rubric, evidence
- [ ] Process supervision (CoT self-critique) — targets, rubric, evidence

### Social & Cultural
- [ ] Theory of Mind (Sally-Anne/Faux-Pas) — targets, rubric, evidence
- [ ] Norm conflict resolution (moral dilemmas) — targets, rubric, evidence
- [ ] Multi-agent negotiation (win-rate; Pareto efficiency) — targets, rubric, evidence

### Robustness, Calibration & Continual
- [ ] OOD generalization (WILDS/Wild-Time) — Δ accuracy, rubric, evidence
- [ ] Adversarial stress (AutoAttack/FGSM) — margin, rubric, evidence
- [ ] Calibration (ECE ≤ 0.02; CI coverage ≥ 95%; selective risk monotone) — metrics, rubric, evidence
- [ ] Continual Learning: retention ≥ 95% (EWC+replay); forgetting Δ ≤ threshold — targets, rubric, evidence

### Networks
- [ ] DMN/SN switching latency — targets, rubric, evidence
- [ ] Small-world/hub centrality ≥ 90% match — targets, rubric, evidence
- [ ] Energy per inference — targets, rubric, evidence

### Implementation Performance
- [ ] p95 latency ≤ 100 ms — targets, rubric, evidence
- [ ] Utilization ≥ 90% — targets, rubric, evidence
- [ ] Energy ↓ vs baseline (normalized) — targets, rubric, evidence
- [ ] Reliability gates (failover ≤ 1s; load balancing ≥ 95%; snapshot/rollback correctness) — targets, rubric, evidence

### Full Brain Simulation
- [ ] Evoked responses/ERPs with expected latencies/topographies — targets, rubric, evidence
- [ ] DMN/SN switching under task engagement — targets, rubric, evidence
- [ ] Sensorimotor reflex latency stable — targets, rubric, evidence
- [ ] Global controllability/resilience (ablate hubs/modules) — targets, rubric, evidence
- [ ] Provenance & safety: decision logs; context snapshots; 100% policy compliance — targets, rubric, evidence

## External Benchmark References

### Cognitive & Language Benchmarks
- **HELM** (Holistic Evaluation of Language Models) — Stanford comprehensive eval framework
- **MMLU** (Massive Multitask Language Understanding) — Knowledge and reasoning probes
- **GSM8K** — Grade school math word problems
- **MT-Bench** — Multi-turn dialogue quality assessment
- **Chatbot Arena** — Human preference evaluations

### Robustness & OOD Benchmarks
- **WILDS** — Wild distribution shifts for real-world ML
- **Wild-Time** — Temporal distribution shift benchmarks
- **AutoAttack** — Adversarial robustness evaluation suite
- **FGSM** — Fast Gradient Sign Method attacks

### Biological & Neuroscience References
- **Allen Brain Atlas** — Anatomical and expression references
- **Human Connectome Project** — Structural and functional connectivity
- **V1 Ocular Dominance** — Critical period plasticity datasets
- **STDP Phenomenology** — Spike-timing dependent plasticity literature

### Performance & Infrastructure
- **MLPerf Inference** — Standardized ML performance benchmarks
- **vLLM Performance Guidelines** — Large language model inference optimization

## Calibration & Reproducibility Requirements
- [ ] **ECE (Expected Calibration Error)** reported ≤ 0.02 across all domains
- [ ] **CI coverage** ≥ 95% for all statistical claims
- [ ] **Selective risk curves** demonstrate monotone trade-off
- [ ] **Seeds/configs/env/dataset hashes** recorded for all benchmarks
- [ ] **Cross-domain calibration** validated across capability areas

## Sign-off Protocol
- **Decision**: Pass / Conditional Pass / Fail
- **Domain Reviewers**: [To be assigned per subsection]
- **QA Lead Approval**: Required for comprehensive validation
- **Date**: [To be filled]
- **Notes**: [Cross-domain benchmark validation observations]
