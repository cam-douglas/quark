*Stage 4 - Childhood* 

**Version 2.1 – 2025-09-01**

> **Canonical** – This file supersedes all earlier Stage 4 roadmaps. 

### Childhood — Myelination & circuit refinement enabling long-range networks.

Childhood (~2–12 yr): rapid myelination and pruning refine circuits, strengthen long-range tracts, and support language, fine-motor skills, and emerging self-regulation.

**Roadmap Status:** 🚨 **FOUNDATION LAYER ✅ COMPLETE** - Other tasks 🚨 PENDING 

From ages ~2–12, progressive myelination accelerates conduction velocity, while pruning and long-range association growth sculpt efficient small-world networks. Sensorimotor milestones, language acquisition, and executive functions rapidly improve.

**Engineering Milestones — implementation tasks driving Stage 4 goals**
* [connectomics-networks] 🚨 **PENDING** - Initiate connectomics-networks P0: generate tractography targets (corpus callosum, anterior/posterior commissures, internal capsule, corticospinal tract) and compute small-world metrics.
* [multi-scale-integration] 🚨 **PENDING** - Start multi-scale-integration P0: register myelination fields to conduction-latency model and expose latency KPIs.
* [metabolism-clearance] 🚨 **PENDING** - Implement CSF/glymphatic flow simulator; link waste-clearance metrics to metabolic controller.
* [graph-pruning] 🚨 **PENDING** - Apply graph-sparsification pruning manager to reach ≤30 % redundant synapses while preserving hub topology.
* [evaluation] 🚨 **PENDING** - Benchmark child-level cognition (WISC puzzle battery) and stream results to KPI dashboard.

**Biological Goals — desired biological outcomes for Stage 4**

* [myelination] 🚨 **PENDING** - Simulate oligodendrocyte-driven myelination using energy-aware rules; record conduction latency gains. **(KPI: average_conduction_latency_ms)**
* [graph-pruning] 🚨 **PENDING** - Prune redundant synapses to approach adult network sparsity while preserving key motifs. **(KPI: small_world_sigma)**
* [connectomics-networks] 🚨 **PENDING** - Establish default mode, salience, and attentional networks at mesoscale resolution. **(KPI: small_world_sigma)**
* [evaluation] 🚨 **PENDING** - Benchmark task performance on child-level cognitive batteries (e.g., WISC subtests). **(KPI: cognitive_score_percentile)**
* [connectomics-networks] 🚨 **PENDING** - Incorporate tractography targets: corpus callosum, anterior/posterior commissures, internal capsule, corticospinal tract. **(KPI: tractography_completion_pct)**
* [metabolism-clearance] 🚨 **PENDING** - Model CSF flow & glymphatic clearance for waste removal. **(KPI: clearance_efficiency_pct)**
* [metabolism-clearance] 🚨 **PENDING** - Embed energy-metabolism constraints (glucose, lactate shuttle, ketone utilization). **(KPI: metabolic_budget_mJ)**
* [myelination-timeline] 🚨 **PENDING** - Develop age-resolved myelin growth curve and incorporate conduction-velocity scaling per tract.
* [csf-flow] 🚨 **PENDING** - Introduce CSF flow & glymphatic clearance subsystem placeholder with perfusion-linked parameters.
* [energy-metabolism] 🚨 **PENDING** - Add metabolic constraint layer (glucose/ketone utilisation, astrocyte-neuron lactate shuttle) affecting robustness KPIs.

**SOTA ML Practices (2025) — recommended methods**

* [efficiency-distillation] 🚨 **PENDING** - Knowledge distillation + network pruning to mimic biological efficiency gains.
* [sparse-routing] 🚨 **PENDING** - Sparse Mixture-of-Experts routing for scalable yet efficient inference.
* [curriculum-learning] 🚨 **PENDING** - Curriculum scheduling: gradually increase task complexity and working-memory load.
* [graph-pruning] 🚨 **PENDING** - Graph sparsification algorithms (edge-drop with spectral constraints) reflecting pruning.



```yaml
modules:
  myelination_sim:
    impl: "Energy-aware myelin growth model"
    metrics: ["axon_diameter", "spike_latency"]
  pruning_manager:
    impl: "Graph sparsifier (spectral)"
    target_sparsity: "<=30% remaining redundant synapses"
  network_emergence:
    impl: "GNN that self-organises DMN/SN/DAN"
  tractography_targets: ["corpus_callosum", "anterior_commissure", "posterior_commissure", "internal_capsule", "corticospinal_tract"]
  csf_flow_model: "csf_glymphatic_model.yaml"
  metabolic_limits:
    glucose_mmol_l: 2.5
    lactate_support: true
    ketone_utilization: true

kpis:
  average_conduction_latency_ms: "<=2x adult baseline"
  small_world_sigma: ">=2.5"

validation:
  - "Match DTI myelination curves ages 2-12"
  - "Test WISC-like puzzle tasks ≥85th percentile vs child norms"
```
**Functional Mapping (links to Appendix A):**
- [Core Cognitive — Memory, Learning, Reasoning, Problem Solving](#$cap-1-core-cognitive)
- [Perception & World Modeling — Multimodal Perception, World Models, Embodiment](#$cap-2-perception-world-modeling)
- [Action & Agency — Planning, Decision-Making, Tool Use, Self-Improvement](#$cap-3-action-agency)
- [Communication & Language — Natural Language, Dialogue, Symbolic Manipulation](#$cap-4-communication-language)
- [Social & Cultural Intelligence — Theory of Mind, Cultural Learning, Ethics & Alignment](#$cap-5-social-cultural)
- [Metacognition & Self-Modeling — Self-Representation, Goal Management, Introspection](#$cap-6-metacognition)
- [Knowledge Integration — Domain Breadth, Transfer, External Knowledge](#$cap-7-knowledge-integration)
- [Robustness & Adaptivity — Adversarial Resistance, Uncertainty Handling](#$cap-8-robustness-adaptivity)
- [Creativity & Exploration — Generativity, Curiosity, Innovation](#$cap-9-creativity-exploration)

---
→ Continue to: [Stage 5 – Adolescence](stage5_adolescence_rules.md)

