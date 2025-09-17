*Stage 3 - Early Post-natal* 

**Version 2.1 – 2025-09-01**

> **Canonical** – This file supersedes all earlier Stage 3 roadmaps. 

## Early Post-natal - Synaptogenesis and critical-period plasticity driving early sensory organisation 

Birth–24 mo: exuberant synaptogenesis and critical periods refine sensory circuits before large-scale pruning.

**Roadmap Status:** 🚨 **FOUNDATION LAYER ✅ COMPLETE** - Other tasks 🚨 PENDING

During this period, balanced excitation/inhibition and neuromodulatory signals (ACh, NE) gate plasticity windows, while spontaneous activity primes circuits ahead of rich sensory experience.

**Engineering Milestones — implementation tasks driving Stage 3 goals**
* [neuromodulatory-systems] 🚨 **PENDING** - Initiate neuromodulatory-systems P0: establish ACh/GABA critical-period gating.
* [hierarchical-processing] 🚨 **PENDING** - Expand hierarchical-processing P1: embed columnar microcircuits with feedforward/feedback probes.
* [sensory-encoding] 🚨 **PENDING** - Train sensory encoder SSL pipeline on infant-style multimodal corpus; attach replay buffer for continual learning.
* [synaptogenesis] 🚨 **PENDING** - Integrate synapse-diversity palette (AMPA, NMDA, GABA_A, gap-junction) into synaptogenesis engine.
* [developmental-biology] 🚨 **PENDING** - Wire cranial nerves I–XII stubs to brain-stem nuclei enabling reflex pathway tests.
* [apoptosis] 🚨 **PENDING** - Add programmed cell death module controlling neuron apoptosis rates with developmental curve.
* [pruning-schedule] 🚨 **PENDING** - Implement activity-dependent synaptic pruning scheduler with quantitative targets per sensory cortex.

**Biological Goals — desired biological outcomes for Stage 3**
* [synaptogenesis] 🚨 **PENDING** - Grow synapse graph to ~1.8× target adult count with Hebbian/anti-Hebbian rules. **(KPI: synapse_density_ratio)**
* [neuromodulatory-systems] 🚨 **PENDING** - Implement critical-period controllers (GABA maturation, NMDAR subunit switch) that modulate plasticity windows. **(KPI: ocular_dominance_dprime)**
* [sensory-encoding] 🚨 **PENDING** - Train sensory pathways using self-supervised multimodal corpora. **(KPI: ocular_dominance_dprime)**
* [synaptogenesis] 🚨 **PENDING** - Validate synapse density curves and ocular dominance index vs primate data. **(KPI: synapse_density_ratio)**
* [synaptogenesis] 🚨 **PENDING** - Define synapse diversity palette: excitatory (glutamatergic), inhibitory (GABAergic), electrical (gap junction). **(KPI: synapse_density_ratio)**
* [developmental-biology] 🚨 **PENDING** - Stub out cranial nerve interfaces (I–XII) mapped to brainstem nuclei. **(KPI: cranial_nerve_stub_pct)**
* [neurochemistry] 🚨 **PENDING** - Catalogue primary fast neurotransmitters (glutamate, GABA, glycine) and secondary modulators (histamine, opioid peptides, endocannabinoids). **(KPI: neurotransmitter_catalog_complete)**

**SOTA ML Practices (2025) — recommended methods**
* [sensory-encoding] 🚨 **PENDING** - Self-supervised contrastive learning (SimCLR-style) for sensory encoding.
* [continual-learning] 🚨 **PENDING** - Elastic Weight Consolidation + replay to balance rapid learning vs stability.
* [parameter-efficiency] 🚨 **PENDING** - Parameter-efficient adapters (LoRA) enabling quick domain adaptation.
* [data-centric] 🚨 **PENDING** - Active learning to prioritize high-information infant stimuli samples.



```yaml
modules:
  synaptogenesis_engine:
    impl: "Graph growth w/ Hebbian + STDP"
    target_synapses: "1.8x_adult"
  critical_period_gate:
    impl: "Plasticity mod switch"
    triggers: ["GABA_maturation", "ACh_level"]
  sensory_encoder_ssl:
    impl: "Contrastive ViT + AudioConv"
  synapse_types: ["AMPA", "NMDA", "GABA_A", "GABA_B", "gap_junction"]
  cranial_nerves_spec: "cranial_nerves.yaml"
  fast_nt_catalog: "primary_neurotransmitters.yaml"
  modulators_catalog: "secondary_modulators.yaml"

kpis:
  synapse_density_ratio: "1.8±0.1"
  ocular_dominance_dprime: ">=1.5"

validation:
  - "Compare V1 ODI curve to macaque months 0-24 dataset"
```
**Functional Mapping (links to Appendix A):**
- [Core Cognitive — Memory, Learning, Reasoning, Problem Solving](#$cap-1-core-cognitive)
- [Perception & World Modeling — Multimodal Perception, World Models, Embodiment](#$cap-2-perception-world-modeling)
- [Action & Agency — Planning, Decision-Making, Tool Use, Self-Improvement](#$cap-3-action-agency)
- [Communication & Language — Natural Language, Dialogue, Symbolic Manipulation](#$cap-4-communication-language)
- [Robustness & Adaptivity — Adversarial Resistance, Uncertainty Handling](#$cap-8-robustness-adaptivity)
- [Creativity & Exploration — Generativity, Curiosity, Innovation](#$cap-9-creativity-exploration)

---
→ Continue to: [Stage 4 – Childhood](stage4_childhood_rules.md)

