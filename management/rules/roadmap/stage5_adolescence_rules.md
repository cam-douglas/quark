*Stage 5 - Adolescence* 

**Version 2.1 – 2025-09-01**

> **Canonical** – This file supersedes all earlier Stage 5 roadmaps. 

## Adolescence — Pruning & neuromodulatory maturation consolidating executive function.

Adolescence (~12–20 yr): pruning, neuromodulator maturation, and final myelination consolidate executive, reward, and abstract reasoning circuits.

**Roadmap Status:** 🚨 **FOUNDATION LAYER ✅ COMPLETE** - Other tasks 🚨 PENDING

Adolescence (~12–20 years) finalizes cortical pruning, strengthens long-range connectivity, and sees peak dopaminergic and serotonergic remodeling, underpinning reward sensitivity and cognitive flexibility.

**Engineering Milestones — implementation tasks driving Stage 5 goals**
* [multi-scale-integration] 🚨 **PENDING** - Continue multi-scale-integration P1: finalize synaptic pruning to ~50 % peak while ensuring stability via OGD constraints.
* [functional-networks] 🚨 **PENDING** - Launch functional-networks P0: DMN/SN/DAN switching controller with salience-gated task sets.
* [neuromodulatory-systems] 🚨 **PENDING** - Mature neuromodulatory-systems P2: receptor-level tuning for DA, NE, 5-HT, ACh based on reward and flexibility assays.
* [planning-agents] 🚨 **PENDING** - Deploy hierarchical RL planner (Hier-PPO + curiosity) over fronto-striatal loops; validate on Stroop and WCST tasks.
* [developmental-biology] 🚨 **PENDING** - Load gene-marker table (NeuN, GFAP, OLIG2, IBA1) and synaptic proteome annotations into connectome metadata.
* [metabolic-constraints] 🚨 **PENDING** - Implement energy metabolism layer (glucose/ketone switch & astrocyte–neuron lactate shuttle) influencing training-time robustness metrics.
* [vascular-placeholder] 🚨 **PENDING** - Prepare perfusion monitoring hooks for upcoming vascular model integration in Stage 6.

**Biological Goals — desired biological outcomes for Stage 5**

* [graph-pruning] 🚨 **PENDING** - Complete synaptic pruning to achieve adult sparsity (~50% of peak synapses). **(KPI: pruning_completion_pct)**
* [neuromodulatory-systems] 🚨 **PENDING** - Mature neuromodulatory systems (DA, NE, 5-HT, ACh) with receptor-level tuning. **(KPI: neuromod_tuning_pct)**
* [planning-agents] 🚨 **PENDING** - Optimize fronto-striatal loops for goal-directed planning and risk evaluation. **(KPI: risk_adjusted_return)**
* [evaluation] 🚨 **PENDING** - Validate reward-learning curves and cognitive control tasks (Stroop, WCST). **(KPI: stroop_accuracy)**
* [developmental-biology] 🚨 **PENDING** - Integrate gene-expression markers (NeuN, GFAP, OLIG2, IBA1) for cell-type tagging across cortex and subcortex. **(KPI: gene_marker_coverage_pct)**
* [proteomics] 🚨 **PENDING** - Annotate synaptic proteome (SNARE, PSD-95, synaptophysin) to inform plasticity kinetics. **(KPI: proteome_annotation_pct)**

**SOTA ML Practices (2025)**
* [planning-agents] 🚨 **PENDING** - Reinforcement learning with curiosity + uncertainty bonuses (InfoBarlow).
* [large-models] 🚨 **PENDING** - Large-scale MoE transformers fine-tuned via Direct Preference Optimization.
* [continual-learning] 🚨 **PENDING** - Continual learning with orthogonal gradient descent to prevent forgetting while pruning.
* [self-critique] 🚨 **PENDING** - Self-critique agents to iteratively refine decision policies.


```yaml
modules:
  pruning_finalizer:
    impl: "Edge-drop with OGD stability constraint"
    target_sparsity: "~50% peak"
  neuromod_maturator:
    impl: "Parameter scheduler for DA/NE/5HT/ACh receptors"
  rl_planner:
    impl: "Hierarchical PPO + curiosity"
  gene_marker_table: "gene_markers.csv"
  synaptic_proteome: "synaptic_proteins.yaml"

kpis:
  pruning_completion_pct: ">=98%"
  stroop_accuracy: ">=95%"
  risk_adjusted_return: ">=baseline_human"

validation:
  - "Compare WCST performance to adolescent cohort"
```
**Functional Mapping (links to Appendix A):**
- [Core Cognitive — Memory, Learning, Reasoning, Problem Solving](#$cap-1-core-cognitive)
- [Action & Agency — Planning, Decision-Making, Tool Use, Self-Improvement](#$cap-3-action-agency)
- [Communication & Language — Natural Language, Dialogue, Symbolic Manipulation](#$cap-4-communication-language)
- [Social & Cultural Intelligence — Theory of Mind, Cultural Learning, Ethics & Alignment](#$cap-5-social-cultural)
- [Metacognition & Self-Modeling — Self-Representation, Goal Management, Introspection](#$cap-6-metacognition)
- [Knowledge Integration — Domain Breadth, Transfer, External Knowledge](#$cap-7-knowledge-integration)
- [Robustness & Adaptivity — Adversarial Resistance, Uncertainty Handling](#$cap-8-robustness-adaptivity)
- [Creativity & Exploration — Generativity, Curiosity, Innovation](#$cap-9-creativity-exploration)

---
→ Continue to: [Stage 6 – Adult](stage6_adult_rules.md)

