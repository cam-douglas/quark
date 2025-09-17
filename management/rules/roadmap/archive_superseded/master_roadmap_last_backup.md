## Version: 2.0 — 2025-08-30

> **This document supersedes all previous roadmaps; it is the single canonical source.**


## Stage 1 — Embryonic: establishes foundational morphogen gradients and structural scaffold for all later development

### Embryonic — Neural-tube patterning

**Overall Status:** 📋 Planned 

<!-- CURSOR RULE: ALWAYS run embryonic_atlas_tests before editing this section -->
The embryonic stage (weeks 3-8) forms the primary brain vesicles—prosencephalon, mesencephalon, rhombencephalon—via neural-tube folding and rostro-caudal patterning. Key molecular gradients (SHH, BMP, WNT, FGF) establish dorsoventral and anteroposterior axes, seeding future forebrain, midbrain, and hindbrain territories.

**Engineering Milestones — implementation tasks driving Stage 1 goals**

* [foundation-layer] Establish foundation-layer morphogen solver aligning with SHH/BMP/WNT/FGF gradients.
* [developmental-biology] Generate lineage-tagged neuroepithelial cells for downstream proliferation.
* [foundation-layer] Excavate ventricular cavities (lateral, third, fourth, aqueduct) in voxel map.
* [foundation-layer] Lay meninges scaffold (dura, arachnoid, pia) surrounding neural tube.


**Biological Goals — desired biological outcomes for Stage 1**

* [foundation-layer] Simulate morphogen gradients to generate a coarse 3-axis voxel map (〈1 mm³ resolution) that labels emerging brain regions. **(KPI: segmentation_dice)**
* [developmental-biology] Instantiate proto-cell populations with lineage tags for later differentiation. **(KPI: cell_count_variance)**
* [foundation-layer] Validate regional segmentation against Allen Brain Atlas embryonic reference. **(KPI: segmentation_dice)**
* [foundation-layer] Map primitive ventricular system (lateral, third, fourth ventricles, cerebral aqueduct) for future CSF modelling. **(KPI: grid_resolution_mm)**
* [foundation-layer] Document meninges scaffold (dura, arachnoid, pia) as exostructural context. **(KPI: meninges_mesh_integrity)**


**SOTA ML Practices (2025) — recommended methods**

* [foundation-layer] Diffusion-based generative fields to model spatial morphogen concentration.
* [foundation-layer] Transformer-based graph neural nets (GNN-ViT hybrid) for 3-D segmentation with limited labels.
* [curriculum-learning] Curriculum learning: train on simplified in-silico embryos → full morphology.
* [data-centric] Data-centric augmentation with synthetic embryo images.

**Capability Domain Validation — links SOTA methods to Biological Goals**
* [core-cognitive] Memory segmentation dice evaluated against voxel map labeling → BG segmentation_dice.
* [knowledge-integration] Region labels integrated into knowledge graph build → BG segmentation_dice.
* [robustness-adaptivity] Morphogen model uncertainty measured via segmentation_dice variance.

```yaml
modules:
  morphogen_solver:
    impl: "Diffusion-field PDE (SHH/BMP/WNT/FGF)"
    grid_res_mm: 1
  voxel_segmentation_model:
    impl: "GNN-ViT hybrid"
    pretrained: "embryo_imagery_v1"
  ventricular_mapper:
    impl: "Flood-fill cavity extractor"
  meninges_scaffold:
    impl: "Tri-layer mesh generator"

kpis:
  segmentation_dice: ">=0.85"
  cell_count_variance: "<=10%"
  grid_resolution_mm: "<=1"
  meninges_mesh_integrity: ">=0.9"

capability_domains:
  core_cognitive: ["segmentation_dice"]
  knowledge_integration: ["segmentation_dice"]
  robustness_adaptivity: ["segmentation_dice"]
```

---

## Stage 2 — Fetal: coordinates neurogenesis and radial migration building cortical laminae

### Fetal — Neurogenesis & radial migration

**Overall Status:** 📋 Planned 

<!-- CURSOR RULE: ALWAYS run fetal_lamination_tests before editing this section -->
Weeks 9-38 witness explosive neurogenesis in ventricular zones followed by radial migration that constructs the six-layered neocortex. Intermediate progenitor cells amplify neuron numbers; guidance cues (Reelin, Notch) organize inside-out lamination.

**Engineering Milestones — implementation tasks driving Stage 2 goals**
* [foundation-layer] Advance foundation-layer to P2 Core Modules: instantiate six-layer cortical template and thalamic relay stubs.
* [hierarchical-processing] Activate hierarchical-processing P0 laminar scaffold validation harness.
* [developmental-biology] Populate brainstem (midbrain, pons, medulla) voxel regions with progenitor cell pools.
* [developmental-biology] Carve cerebellar vermis / hemispheres and embed deep nuclei placeholders.
* [developmental-biology] Compile initial ion-channel library (Nav/Kv/Cav/HCN) and attach to neurogenesis engine.

**Biological Goals — desired biological outcomes for Stage 2**

* [foundation-layer] Simulate morphogen gradients to generate a coarse 3-axis voxel map (〈1 mm³ resolution) that labels emerging brain regions. **(KPI: segmentation_dice)**
* [developmental-biology] Instantiate proto-cell populations with lineage tags for later differentiation. **(KPI: cell_count_variance)**
* [foundation-layer] Validate regional segmentation against Allen Brain Atlas embryonic reference. **(KPI: segmentation_dice)**
* [foundation-layer] Map primitive ventricular system (lateral, third, fourth ventricles, cerebral aqueduct) for future CSF modelling. **(KPI: grid_resolution_mm)**
* [foundation-layer] Document meninges scaffold (dura, arachnoid, pia) as exostructural context. **(KPI: meninges_mesh_integrity)**


**SOTA ML Practices (2025)**
* [foundation-layer] Mixture-of-Experts diffusion models to upscale neuron distribution statistically.
* [hierarchical-processing] Reinforcement-learning cellular automata (policy: migration vector; reward: laminar ordering).
* [foundation-layer] Parameter-efficient fine-tuning (LoRA) on in-utero MRI segmentation models.
* [developmental-biology] Use WandB sweeps for hyper-parameter exploration of proliferation rates.

**Capability Domain Validation — links SOTA methods to Biological Goals**
* [perception-world-modeling] Laminar classifier accuracy ≥ laminar_accuracy KPI validates predictive world-model fidelity.
* [core-cognitive] Neuron_count_error_pct informs memory capacity simulations.
* [implementation-pillars] Ion-channel library completeness cross-checked with neurogenesis_engine safety tests.

```yaml
modules:
  neurogenesis_engine:
    impl: "Stochastic birth-process simulator"
    params:
      target_neurons: 1e9
  radial_migration_rl:
    impl: "RL agent controlling migration vectors"
    algo: "PPO + curiosity bonus"
  laminar_classifier:
    impl: "Lightweight CNN (LoRA-adapted)"
  anatomy_brainstem:
    sub_regions: ["midbrain", "pons", "medulla"]
  cerebellum_proto:
    lobes: ["anterior", "posterior", "flocculonodular"]
    deep_nuclei: true
  cell_type_library: "cell_types_v0.yaml"
  ion_channel_catalog: "ion_channel_library.yaml"

kpis:
  laminar_accuracy: ">=0.80"
  neuron_count_error_pct: "<=5%"

validation:
  - "Compare laminar thickness vs dMRI (fetal 30-38 w)"
```

---

## Stage 3 — Early post-natal: synaptogenesis and critical-period plasticity driving early sensory organisation

### Early post-natal — Synaptogenesis & critical-period plasticity

**Overall Status:** 📋 Planned 

<!-- CURSOR RULE: ALWAYS run synaptogenesis_regression_suite before editing this section -->
The first two post-natal years are marked by exuberant synaptogenesis (peak synapse density ~1.5–2× adult) followed by activity-dependent pruning. Critical periods open in sensory cortices—ocular dominance (V1), tonotopy (A1)—driven by balanced excitation/inhibition and neuromodulatory gating.

**Engineering Milestones — implementation tasks driving Stage 3 goals**
* [neuromodulatory-systems] Initiate neuromodulatory-systems P0: establish ACh/GABA critical-period gating.
* [hierarchical-processing] Expand hierarchical-processing P1: embed columnar microcircuits with feedforward/feedback probes.
* [sensory-encoding] Train sensory encoder SSL pipeline on infant-style multimodal corpus; attach replay buffer for continual learning.
* [synaptogenesis] Integrate synapse-diversity palette (AMPA, NMDA, GABA_A, gap-junction) into synaptogenesis engine.
* [developmental-biology] Wire cranial nerves I–XII stubs to brain-stem nuclei enabling reflex pathway tests.

**Biological Goals — desired biological outcomes for Stage 3**
* [synaptogenesis] Grow synapse graph to ~1.8× target adult count with Hebbian/anti-Hebbian rules. **(KPI: synapse_density_ratio)**
* [neuromodulatory-systems] Implement critical-period controllers (GABA maturation, NMDAR subunit switch) that modulate plasticity windows. **(KPI: ocular_dominance_dprime)**
* [sensory-encoding] Train sensory pathways using self-supervised multimodal corpora. **(KPI: ocular_dominance_dprime)**
* [synaptogenesis] Validate synapse density curves and ocular dominance index vs primate data. **(KPI: synapse_density_ratio)**
* [synaptogenesis] Define synapse diversity palette: excitatory (glutamatergic), inhibitory (GABAergic), electrical (gap junction). **(KPI: synapse_density_ratio)**
* [developmental-biology] Stub out cranial nerve interfaces (I–XII) mapped to brainstem nuclei. **(KPI: cranial_nerve_stub_pct)**
* [neurochemistry] Catalogue primary fast neurotransmitters (glutamate, GABA, glycine) and secondary modulators (histamine, opioid peptides, endocannabinoids). **(KPI: neurotransmitter_catalog_complete)**

**SOTA ML Practices (2025) — recommended methods**
* [sensory-encoding] Self-supervised contrastive learning (SimCLR-style) for sensory encoding.
* [continual-learning] Elastic Weight Consolidation + replay to balance rapid learning vs stability.
* [parameter-efficiency] Parameter-efficient adapters (LoRA) enabling quick domain adaptation.
* [data-centric] Active learning to prioritize high-information infant stimuli samples.

**Capability Domain Validation — links SOTA methods to Biological Goals**
* [communication-language] Ocular_dominance_dprime improvement maps to early language perception benchmarks.
* [creativity-exploration] Synapse_density_ratio diversity enables generative circuit exploration tasks.
* [robustness-adaptivity] Cranial_nerve_stub_pct coverage ensures reliable reflex testing under noise.

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

---

## Stage 4 — Childhood: myelination & circuit refinement enabling efficient long-range networks

### Childhood — Myelination & circuit refinement

**Overall Status:** 📋 Planned 

<!-- CURSOR RULE: ALWAYS run myelination_benchmarks before editing this section -->
From ages ~2–12, progressive myelination accelerates conduction velocity, while pruning and long-range association growth sculpt efficient small-world networks. Sensorimotor milestones, language acquisition, and executive functions rapidly improve.

**Engineering Milestones — implementation tasks driving Stage 4 goals**
* [connectomics-networks] Initiate connectomics-networks P0: generate tractography targets (corpus callosum, anterior/posterior commissures, internal capsule, corticospinal tract) and compute small-world metrics.
* [multi-scale-integration] Start multi-scale-integration P0: register myelination fields to conduction-latency model and expose latency KPIs.
* [metabolism-clearance] Implement CSF/glymphatic flow simulator; link waste-clearance metrics to metabolic controller.
* [graph-pruning] Apply graph-sparsification pruning manager to reach ≤30 % redundant synapses while preserving hub topology.
* [evaluation] Benchmark child-level cognition (WISC puzzle battery) and stream results to KPI dashboard.

**Biological Goals — desired biological outcomes for Stage 4**

* [myelination] Simulate oligodendrocyte-driven myelination using energy-aware rules; record conduction latency gains. **(KPI: average_conduction_latency_ms)**
* [graph-pruning] Prune redundant synapses to approach adult network sparsity while preserving key motifs. **(KPI: small_world_sigma)**
* [connectomics-networks] Establish default mode, salience, and attentional networks at mesoscale resolution. **(KPI: small_world_sigma)**
* [evaluation] Benchmark task performance on child-level cognitive batteries (e.g., WISC subtests). **(KPI: cognitive_score_percentile)**
* [connectomics-networks] Incorporate tractography targets: corpus callosum, anterior/posterior commissures, internal capsule, corticospinal tract. **(KPI: tractography_completion_pct)**
* [metabolism-clearance] Model CSF flow & glymphatic clearance for waste removal. **(KPI: clearance_efficiency_pct)**
* [metabolism-clearance] Embed energy-metabolism constraints (glucose, lactate shuttle, ketone utilization). **(KPI: metabolic_budget_mJ)**

**SOTA ML Practices (2025) — recommended methods**

* [efficiency-distillation] Knowledge distillation + network pruning to mimic biological efficiency gains.
* [sparse-routing] Sparse Mixture-of-Experts routing for scalable yet efficient inference.
* [curriculum-learning] Curriculum scheduling: gradually increase task complexity and working-memory load.
* [graph-pruning] Graph sparsification algorithms (edge-drop with spectral constraints) reflecting pruning.

**Capability Domain Validation — links SOTA methods to Biological Goals**
* [action-agency] Small_world_sigma increase boosts planning latency benchmarks.
* [robustness-adaptivity] Average_conduction_latency_ms reduction validates adaptive routing efficiency.
* [knowledge-integration] Tractography_completion_pct feeds connectome knowledge graph.

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

---

## Stage 5 — Adolescence: pruning & neuromodulatory maturation consolidating executive functions

### Adolescence — Pruning & neuromodulatory maturation

**Overall Status:** 📋 Planned 

<!-- CURSOR RULE: ALWAYS run pruning_neuromod_tests before editing this section -->
Adolescence (~12–20 years) finalizes cortical pruning, strengthens long-range connectivity, and sees peak dopaminergic and serotonergic remodeling, underpinning reward sensitivity and cognitive flexibility.

**Engineering Milestones — implementation tasks driving Stage 5 goals**
* [multi-scale-integration] Continue multi-scale-integration P1: finalize synaptic pruning to ~50 % peak while ensuring stability via OGD constraints.
* [functional-networks] Launch functional-networks P0: DMN/SN/DAN switching controller with salience-gated task sets.
* [neuromodulatory-systems] Mature neuromodulatory-systems P2: receptor-level tuning for DA, NE, 5-HT, ACh based on reward and flexibility assays.
* [planning-agents] Deploy hierarchical RL planner (Hier-PPO + curiosity) over fronto-striatal loops; validate on Stroop and WCST tasks.
* [developmental-biology] Load gene-marker table (NeuN, GFAP, OLIG2, IBA1) and synaptic proteome annotations into connectome metadata.

**Biological Goals — desired biological outcomes for Stage 5**

* [graph-pruning] Complete synaptic pruning to achieve adult sparsity (~50% of peak synapses). **(KPI: pruning_completion_pct)**
* [neuromodulatory-systems] Mature neuromodulatory systems (DA, NE, 5-HT, ACh) with receptor-level tuning. **(KPI: neuromod_tuning_pct)**
* [planning-agents] Optimize fronto-striatal loops for goal-directed planning and risk evaluation. **(KPI: risk_adjusted_return)**
* [evaluation] Validate reward-learning curves and cognitive control tasks (Stroop, WCST). **(KPI: stroop_accuracy)**
* [developmental-biology] Integrate gene-expression markers (NeuN, GFAP, OLIG2, IBA1) for cell-type tagging across cortex and subcortex. **(KPI: gene_marker_coverage_pct)**
* [proteomics] Annotate synaptic proteome (SNARE, PSD-95, synaptophysin) to inform plasticity kinetics. **(KPI: proteome_annotation_pct)**

**SOTA ML Practices (2025)**
* [planning-agents] Reinforcement learning with curiosity + uncertainty bonuses (InfoBarlow).
* [large-models] Large-scale MoE transformers fine-tuned via Direct Preference Optimization.
* [continual-learning] Continual learning with orthogonal gradient descent to prevent forgetting while pruning.
* [self-critique] Self-critique agents to iteratively refine decision policies.

**Capability Domain Validation — links SOTA methods to Biological Goals**
* [metacognition-self-modeling] Risk_adjusted_return KPI updates self-evaluation modules.
* [social-cultural-intelligence] Neuromod_tuning_pct aligns socio-emotional agent behaviors.
* [robustness-adaptivity] Pruning_completion_pct influences uncertainty calibration in control loops.

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

---

## Stage 6 — Adult: stable networks, lifelong plasticity, metabolic & vascular homeostasis

### Adult — Stable networks, ongoing plasticity, metabolic & vascular homeostasis

**Overall Status:** 📋 Planned 

<!-- CURSOR RULE: ALWAYS run adult_performance_ci before editing this section -->
Adulthood (>20 years) stabilizes network architecture while maintaining targeted plasticity for learning and memory. Metabolic efficiency, vascular regulation, and glial-mediated homeostasis dominate system maintenance.

**Engineering Milestones — implementation tasks driving Stage 6 goals**
* [whole-brain-integration] Complete whole-brain-integration P2: enable cross-system message bus with latency <10 ms and fault-tolerant retries.
* [cloud-infrastructure] Finalize cloud-infrastructure P1: deploy distributed runtime on Ray+Kubeflow cluster; meet 99.9 % uptime target.
* [vascular-model] Integrate vascular-model (Circle of Willis) with neuro-vascular coupling layer; run BOLD-simulation benchmarks.
* [blood-brain-barrier] Activate BBB-pericyte module; expose molecular transport policy for drug/toxin simulations.
* [sleep-consolidation] Launch sleep-consolidator replay GAN; schedule nightly synaptic down-selection and memory compression jobs.

**Biological Goals — desired biological outcomes for Stage 6**

* [sleep-consolidation] Maintain synaptic homeostasis via sleep-dependent consolidation and glymphatic clearance models. **(KPI: agi_domain_score_avg)**
* [energy-metabolism] Implement energy-aware scheduling (glucose, lactate, ketone) and blood-flow coupling. **(KPI: energy_per_synaptic_event_nJ)**
* [cloud-infrastructure] Achieve production-scale distributed simulation with fault-tolerant orchestration. **(KPI: uptime_pct)**
* [evaluation] Benchmark full AGI capability domains with ≥90 % target KPIs. **(KPI: agi_domain_score_avg)**
* [vascular-model] Model cerebral blood supply via Circle of Willis with perfusion constraints. **(KPI: vascular_perfusion_accuracy_pct)**
* [blood-brain-barrier] Implement blood–brain barrier (BBB) and pericyte regulation modules for molecular transport. **(KPI: bbb_transport_fidelity_pct)**

**SOTA ML Practices (2025) — recommended methods**

* [sleep-consolidation] Retrieval-augmented MoE transformers with on-device quantized adapters.
* [knowledge-graphs] Auto-RAG pipelines for continuous knowledge integration.
* [federated-learning] Federated fine-tuning and edge inference (MLC / WebLLM) for energy savings.
* [mlops] MLOps: CI/CD with Kubeflow, Ray Serve, vLLM optimized inference.

**Capability Domain Validation — links SOTA methods to Biological Goals**
* [knowledge-integration] AGI_domain_score_avg consolidates cross-domain capability measurements.
* [implementation-pillars] Uptime_pct ensures cloud-infrastructure reliability supporting all domains.
* [vascular-model] Vascular_perfusion_accuracy_pct couples to energy-metabolism scheduling tests.

```yaml
modules:
  sleep_consolidator:
    impl: "Replay GAN + synaptic down-selection"
  metabolic_controller:
    impl: "Energy budget allocator (glucose ↔ lactate shuttle)"
  vascular_coupler:
    impl: "Neuro-vascular unit model"
  distributed_runtime:
    impl: "Ray + Kubeflow hybrid"
  vascular_model: "circle_of_willis_network.yaml"
  bbb_pericyte_module: "bbb_pericyte_model.yaml"

kpis:
  agi_domain_score_avg: ">=90%"
  energy_per_synaptic_event_nJ: "<=0.5"
  uptime_pct: ">=99.9%"

validation:
  - "Cross-domain benchmark suite (HELM, MMLU, MT-Bench)"
  - "Compare BOLD signal vs simulated neuro-vascular coupling"
```

---

🎯 Executive Summary

This roadmap delivers a biologically grounded, AGI-capable brain simulation program that (1) aligns with contemporary systems neuroscience, (2) implements 10 AGI capability domains end-to-end, and (3) ships on scalable cloud infrastructure. It combines canonical cortical microcircuitry, thalamo-cortico-basal ganglia loops, neuromodulation, hippocampal memory systems, cerebellar prediction, and large-scale networks (DMN, DAN/VAN, Salience) with modern ML orchestration (RAG/KG, hybrid neural-symbolic planning, deterministic workflow engines). Biological assumptions follow widely cited reviews on cortical laminae and microcircuits, dopamine reward prediction error, locus-coeruleus adaptive gain, acetylcholine-gated encoding/retrieval, serotonin and cognitive flexibility, sleep-dependent consolidation, small-world network topology, critical periods and pruning, predictive coding / free-energy, and the global neuronal workspace. 


✅ Integration Status

AGI Capability Domains (10): Integrated across architecture & evaluation

Enhanced Brain Launcher: agi_enhanced_brain_launcher.py _(TODO: file not found)_ (entrypoint)

AGI Connectome Config: [agi_enhanced_connectome.yaml](../../configurations/project/agi_enhanced_connectome.yaml) (module graph + neuromodulation buses)

Optimization Level: High efficiency & robustness (sparse / hierarchical / caching)


Docs: This file is the canonical MASTER_ROADMAP




🧩 The 10 AGI Capability Domains (Unified Index)

Core Cognitive — Memory, Learning, Reasoning, Problem Solving

Perception & World Modeling — Multimodal Perception, World Models, Embodiment

Action & Agency — Planning, Decision-Making, Tool Use, Self-Improvement

Communication & Language — Natural Language, Dialogue, Symbolic Manipulation

Social & Cultural Intelligence — Theory of Mind, Cultural Learning, Ethics & Alignment

Metacognition & Self-Modeling — Self-Representation, Goal Management, Introspection

Knowledge Integration — Domain Breadth, Transfer, External Knowledge

Robustness & Adaptivity — Adversarial Resistance, Uncertainty Handling

Creativity & Exploration — Generativity, Curiosity, Innovation

Implementation Pillars — Scalability, Modularity, Safety, Evaluation



1) Human Overview — Core Cognitive Domains

Neurobiology
Episodic memory relies on hippocampus (DG/CA3 for pattern separation/completion) with cortical consolidation during slow-wave sleep; acetylcholine gates encoding vs consolidation across hippocampal-entorhinal loops. Prefrontal cortex (PFC) supports working memory and executive control, integrating with thalamus via cortico-thalamo-cortical loops; cortex obeys laminar feedforward (L4→L2/3) and feedback (L5/6) motifs. Reasoning benefits from hybrid neural-symbolic processing layered over these substrates. 


Implication for Design
Implement hippocampal episodic store + cortical semantic graph; use cholinergic mode switches for encode/retrieve policies; PFC-gated working memory; hybrid neuro-symbolic stack for reasoning and proof search.

Machine-Readable Spec

memory_architecture:
  episodic_memory:
    implementation: "Hippocampal-like (DG/CA3) + cortical consolidation"
    optimization: "Sparse coding, compressed indices"
    robustness: "Multi-scale replay; error-correcting recall"
  semantic_memory:
    implementation: "Distributed cortical + typed knowledge graph"
    optimization: "Graph compaction + alias resolution"
    robustness: "Redundant assertions; truth-maintenance"
  procedural_memory:
    implementation: "BG-centered skill learning"
    optimization: "Hierarchical skill options"
    robustness: "Cortico-cerebellar backups"
  working_memory:
    implementation: "PFC buffer + thalamo-cortical gating"
    optimization: "Selective attention + chunking"
    robustness: "Dual-stream maintenance + repair"
learning_systems:
  few_shot_learning:
    implementation: "Meta-learning; rapid weight adaptation"
    optimization: "MAML-like inner loops"
    robustness: "Anti-forgetting via EWC + replay"
  meta_learning:
    implementation: "Hierarchical learning-to-learn"
    optimization: "Low-variance gradient estimates"
    robustness: "Cross-task transfer checks"
  continual_learning:
    implementation: "EWC + orthogonal gradient routing"
    optimization: "Selective synaptic protection"
    robustness: "Progressive architecture search"
reasoning_architecture:
  deductive_reasoning:
    implementation: "Neuro-symbolic prover; neural back-end"
    optimization: "Parallel rule application"
    robustness: "Backtracking + consistency checks"
  inductive_reasoning:
    implementation: "Uncertainty-aware generalization"
    optimization: "Bayesian neural inference"
    robustness: "Calibrated estimates"
  abductive_reasoning:
    implementation: "Best-explanation search on causal graphs"
    optimization: "Approximate inference"
    robustness: "Multi-hypothesis tracking"
  counterfactual_reasoning:
    implementation: "World-model simulation"
    optimization: "Efficient rollouts"
    robustness: "Causal-graph validation"

2) Human Overview — Perception & World Modeling

Neurobiology
Cortex organizes perception via hierarchical predictive coding: feedback conveys predictions; feedforward conveys prediction errors; thalamo-cortical loops update beliefs under uncertainty. Attention uses partially segregated DAN (goal-driven, intraparietal/superior frontal) and VAN (stimulus-driven, right-lateralized temporo-parietal), while the Salience Network (anterior insula + dACC) allocates control and mediates network switching. The DMN supports internal simulation and autobiographical/world-model inferences. 

Annual Reviews

Implication for Design
Adopt predictive world models with uncertainty quantification; cross-modal fusion guided by attention controllers (DAN/VAN emulation) and a salience-driven switch. DMN-like background simulation supports counterfactuals and planning.

Machine-Readable Spec

perception_systems:
  vision:
    implementation: "Hierarchical features + attention"
    optimization: "Conv/ViT hybrids; multi-scale fusion"
    robustness: "Occlusion/adversarial defenses"
  audition:
    implementation: "Streaming temporal encoders"
    optimization: "Chunked inference; online ASR"
    robustness: "Noise-robust features"
  language:
    implementation: "Transformer with neural grounding"
    optimization: "KV-caching; attention sparsity"
    robustness: "Context disambiguation + guardrails"
  multimodal_fusion:
    implementation: "Cross-modal attention + shared embeddings"
    optimization: "Late/early gating; missing-modality handling"
    robustness: "Modality dropout + imputation"
world_modeling:
  predictive_simulation:
    implementation: "Latent dynamics + model-based control"
    optimization: "Trajectory sampling with branching"
    robustness: "Uncertainty-aware rollouts"
  causal_reasoning:
    implementation: "Structure learning for causal graphs"
    optimization: "Interventional queries"
    robustness: "Confounder detection"
  physics_simulation:
    implementation: "Differentiable physics"
    optimization: "Learned surrogates; operator nets"
    robustness: "Multi-physics ensembles"

3) Human Overview — Action & Agency 
Neurobiology
Goal-directed behavior emerges from cortico-basal ganglia-thalamo-cortical loops. Dopamine neurons encode reward prediction error that trains striatal policies (direct/indirect pathways), mapping closely to actor–critic learning: dopamine RPEs update the critic; dopamine-dependent plasticity updates the actor. The cerebellum provides forward models for predictive control and contributes to cognition and affect (CCAS), tuning timing and error correction across cognitive/motor domains. PFC orchestrates multi-step plans; thalamus gates cortical working sets; SN (dACC/insula) modulates effort and task set switching. 


Implication for Design
Use hierarchical RL with options (SMDP); actor–critic with model-based planning (MCTS) and cerebellar-like forward models; risk/effort costs learned from salience-weighted signals.

Machine-Readable Spec

action_systems:
  hierarchical_planning:
    implementation: "Multi-level goal decomposition"
    optimization: "HRL + learned subgoals"
    robustness: "Plan repair under uncertainty"
  temporal_abstraction:
    implementation: "Options framework; macro-actions"
    optimization: "Option discovery via empowerment"
    robustness: "Credit assignment across timescales"
  decision_making:
    implementation: "Utility/risk-aware policies"
    optimization: "MCTS + actor–critic hybrids"
    robustness: "Constraint-aware fallback policies"
tool_systems:
  tool_adaptation:
    implementation: "Few-shot tool APIs + affordances"
    optimization: "Meta-learning of tool use"
    robustness: "Safety-constrained exploration"
  self_improvement:
    implementation: "Introspective debugging; skill library"
    optimization: "Self-supervised updates"
    robustness: "Conservative self-modification"

4) Human Overview — Communication & Language 

Neurobiology
Language engages distributed fronto-temporo-parietal networks superimposed on hierarchical cortical microcircuits. Conscious access to linguistic content is well modeled by the Global Neuronal Workspace: information becomes widely broadcast across long-range fronto-parietal hubs (late P3/LPC signatures) enabling cross-module manipulation (e.g., reasoning, planning). Predictive coding explains rapid context integration and ambiguity resolution via top-down expectations. 


Implication for Design
Adopt a GNW-like “Conductor” that broadcasts verified language states to specialists; hybrid neural-symbolic parsing; memory-augmented discourse tracking.

Machine-Readable Spec

language_systems:
  natural_language:
    implementation: "LLM + grounding to memory/actuators"
    optimization: "Sparse attention; retrieval-augmented"
    robustness: "Pragmatics + safety filters"
  dialogue_competence:
    implementation: "Stateful manager + user model"
    optimization: "Memory compaction; policy shaping"
    robustness: "Recovery strategies; ambiguity repair"
  symbolic_manipulation:
    implementation: "Differentiable programming + logic"
    optimization: "Solver integration (SAT/SMT/CAS)"
    robustness: "Proof checking; type safety"

5) Human Overview — Social & Cultural Intelligence 

Neurobiology
Theory of Mind (ToM) consistently recruits medial prefrontal cortex and bilateral TPJ, supporting belief attribution across tasks. Moral/evaluative choices engage ventromedial PFC with contributions from limbic circuits; vmPFC damage alters moral judgments, highlighting its role in integrating social value with outcome evaluation. Salience and control networks coordinate social attention and conflict monitoring. 


Implication for Design
Represent other-agents’ beliefs/preferences; train culturally adaptive norms; align policies with preference learning and normative uncertainty.

Machine-Readable Spec

social_systems:
  theory_of_mind:
    implementation: "Recursive belief modeling (I-POMDP)"
    optimization: "Factorized epistemic states"
    robustness: "Counterfactual checks; deception handling"
  cultural_learning:
    implementation: "Cross-domain pattern mining"
    optimization: "Transfer across cultural corpora"
    robustness: "Bias detection + debiasing"
  ethics_alignment:
    implementation: "Preference learning + rule models"
    optimization: "Value-uncertainty aware planning"
    robustness: "Constraint monitors + audits"

6) Human Overview — Metacognition & Self-Modeling 

Neurobiology
Metacognition draws on fronto-parietal workspaces to represent confidence/uncertainty and to select goals. Predictive coding / free-energy principles provide a normative account: the system monitors model evidence and minimizes expected surprise via action/perception updates; GNW broadcasting explains access to introspective reports. 


Implication for Design
Expose internal belief states and confidence; maintain a self-model (capabilities, limits, provenance); generate faithful post-hoc rationales tied to verifiers.

Machine-Readable Spec

metacognition_systems:
  self_representation:
    implementation: "Architectural self-model + capability map"
    optimization: "Cheap health checks + heartbeats"
    robustness: "Self-model validation + drift alarms"
  goal_management:
    implementation: "Hierarchical goal graph + constraints"
    optimization: "Multi-objective scalarization"
    robustness: "Goal conflict resolution + watchdogs"
  introspection:
    implementation: "Evidence-linked explanations"
    optimization: "Cost-aware introspection policies"
    robustness: "Uncertainty communication"

7) Human Overview — Knowledge Integration 
Neurobiology
The brain exhibits small-world topology with clustered hubs enabling efficient integration across domains; long-range association areas support abstraction and multi-modal binding (DMN, fronto-parietal). Such architectures optimize integration-vs-cost trade-offs. 


Implication for Design
Back a typed knowledge graph with vector memory; enforce consistency via truth-maintenance; enable analogical mapping and schema induction.

Machine-Readable Spec

knowledge_systems:
  domain_breadth:
    implementation: "Multi-domain KG + embeddings"
    optimization: "Approximate nearest-neighbor (ANN)"
    robustness: "Consistency checks; contradiction flags"
  cross_domain_transfer:
    implementation: "Analogical reasoning (structure mapping)"
    optimization: "Graph kernels; subgraph isomorphism"
    robustness: "Transfer validity tests"
  external_integration:
    implementation: "API & tool adapters; cached corpora"
    optimization: "Staleness-aware caching"
    robustness: "Source validation + provenance"

8) Human Overview — Robustness & Adaptivity 
Neurobiology
Perception/action under uncertainty is well described by Bayesian accounts, with population codes representing likelihoods and priors; behavior and neural data support weighting by uncertainty. This motivates explicit uncertainty estimation, calibration, and risk-aware control for robustness and safe adaptation. 


Implication for Design
Use Bayesian/ensemble heads, calibrated logits, selective prediction; adversarial training & detection; active learning for out-of-distribution events.

Machine-Readable Spec

robustness_systems:
  adversarial_resistance:
    implementation: "Adversarial training + detectors"
    optimization: "Certified bounds on critical paths"
    robustness: "Multi-layer defenses; canary prompts"
  noise_tolerance:
    implementation: "Denoising, redundancy, voting"
    optimization: "Adaptive filters; robust loss"
    robustness: "SNR-aware routing"
  uncertainty_handling:
    implementation: "Bayesian/ensemble predictors"
    optimization: "Fast posterior approximations"
    robustness: "Well-calibrated confidence intervals"

9) Human Overview — Creativity & Exploration 

Neurobiology
Creative cognition recruits DMN for generative recombination with dynamic coupling to control networks for evaluation and refinement; sleep consolidates and reorganizes representations, stabilizing pattern separation that supports novel recombination without interference. Curiosity-driven exploration aligns with intrinsic reward for information gain. 


Implication for Design
Dual-system generator+critic; intrinsic-motivation objectives (e.g., information gain/novelty bonuses); safety-bounded exploration.

Machine-Readable Spec

creativity_systems:
  generativity:
    implementation: "Latent exploration + diffusion"
    optimization: "Diversity priors; quality filters"
    robustness: "Constraint-guided sampling"
  curiosity_learning:
    implementation: "Intrinsic motivation (info gain)"
    optimization: "Count-based or predictive error bonuses"
    robustness: "Safe exploration envelopes"
  innovation:
    implementation: "Combinatorial search + analogies"
    optimization: "Neural-guided program synthesis"
    robustness: "Novelty verification"

10) Human Overview — Implementation Pillars 

Neurobiology → Engineering Pragmatics
To sustain whole-brain scale and continuous learning, the platform must be modular (mirroring segregated but integrated brain networks), scalable (distributed compute akin to parallel cortical columns), safe (multi-layer control like neuromodulatory gating), and measurable (as in neurophysiology, with rich telemetry). Predictive coding/GNW suggest explicit verification and broadcast layers for safe, auditable operation. 


Machine-Readable Spec

implementation_systems:
  scalability:
    implementation: "Distributed microservices + Ray"
    optimization: "Auto-scaling; load-aware sharding"
    robustness: "Fault-tolerant orchestration"
  modularity:
    implementation: "Service mesh; typed contracts"
    optimization: "gRPC + schema evolution"
    robustness: "Circuit breakers; backpressure"
  safety:
    implementation: "Policy sandbox + monitors"
    optimization: "Runtime guards; spec-based tests"
    robustness: "Fail-safe fallbacks; kill-switches"
  evaluation:
    implementation: "Comprehensive bench + probes"
    optimization: "Parallelized eval; cheap canaries"
    robustness: "Cross-validation; drift alarms"


☁️ Cloud & Cost (Machine-Readable)
laptop_environment:
  neural_simulation: "PyTorch + Brian2 + NEURON"
  data_management: "SQLite + HDF5"
  registry: "MLflow local"
  resources: { max_memory: "16GB", max_cpu_cores: 8, gpu: "RTX optional" }
  capabilities: ["≤1M neurons", "STDP", "Neuromod systems", "Visual validation"]

cloud_burst:
  compute: "Ray + Kubernetes"
  training: "Distributed PyTorch"
  orchestration: "Apache Airflow"
  serving: "Ray Serve"
  cost: "Spot instances"
  capabilities: ["≤100M neurons", "Large-scale training", "Bio validation"]

production:
  cluster: "Kubernetes (EKS/GKE/AKS) + Ray"
  workflows: "Kubeflow + MLflow"
  data: "Spark + Dask"
  monitoring: "Prometheus + Grafana"
  costs: { reserved: "30-60% savings", autoscale: true }
  capabilities: ["≤100B neurons", "Full brain sim", "Continuous workloads"]


Estimated Monthly — Dev: $0–50; Test: $50–200; Prod: $200–1000 (spot/reserved, autoscaling).

🧪 Testing Framework & Coverage

Current: 35.9% (14/39) → Target: 85%+ (33/39); Projected: 92.3% (36/39)

Categories (24 new tests): Neuromod (4), Hierarchical (3), Connectomics (3), Functional Networks (4), Developmental (3), Multi-Scale (2), Whole-Brain (2), Cloud (3)

Quality Gates

Visual validation for all components

Bio plausibility checks (laminar IO, neuromodulatory signatures, oscillations)

Performance (real-time loop stability; latency budgets)

Integration (network switching; replay integrity)

Cloud deployment (idempotent infra, autoscaling tests)

📊 Metrics
agi_metrics:
  memory_systems: "≥95% recall across modalities"
  learning_efficiency: "≥10x sample efficiency vs baseline"
  reasoning_accuracy: "≥99% on formal logic suites"
  problem_solving: "≥95% success on novel sets"
integration_metrics:
  transfer: "≥90% cross-domain transfer efficiency"
  multimodal: "≥95% integrated task accuracy"
  adaptation: "≥85% success in novel envs"
  social_intel: "≥90% on ToM-style tasks"
robustness_metrics:
  adversarial: "≥99% resistance to known attacks"
  calibration: "≥95% CI coverage"
  continual: "≥95% retention; no catastrophic forgetting"
  safety: "100% policy compliance"
optimization_metrics:
  compute: "≥90% device utilization"
  memory: "≥80% footprint efficiency"
  energy: "≥50% lower vs naive baselines"
  latency: "≤100 ms typical inference"
scalability_metrics:
  horizontal: "Linear to 1000+ nodes"
  balancing: "≥95% load distribution"
  recovery: "≤1s failover"
  cost: "≥60% savings vs on-demand"

🧬 Developmental & Biological Fidelity Targets
biological_accuracy:
  stdp: "≥95% match to STDP phenomenology"
  neuromod: "≥92% qualitative match to DA/NE/5-HT/ACh roles"
  cortical_arch: "≥88% laminar IO and motif fidelity"
  connectivity: "≥90% small-world + hub metrics"
validation_framework:
  neuroscience_benchmarks: "Required per module"
  continuous_bio_testing: true
  real_time_monitoring: "Oscillation bands; evoked responses"
  cost_tracking: "Budget adherence per stage"


Notes on Evidence Mapping

DA/RPE (Schultz/updates) → actor–critic loops. 


LC-NE adaptive gain → exploration/exploitation. 


ACh encoding vs consolidation gates. 


5-HT reversal learning/flexibility (OFC/DRN). 


Sleep/SWS consolidation; pattern separation stability. 


DAN/VAN/SN/DMN roles and switching. 


Small-world integration cost-efficiency. 


Critical periods / pruning developmental controls. 


Predictive coding/GNW normative/architectural frames. 



🧱 Code & Files (Naming, Entry Points)

Entry: [brain_simulator.py](../../../brain/architecture/brain_simulator.py) or [embodied_brain_simulation.py](../../../embodied_brain_simulation.py)

Configs: [agi_enhanced_connectome.yaml](../../configurations/project/agi_enhanced_connectome.yaml)

Packages:

/brain/cortex/* (laminar, columns, attention)

/brain/bg/* (actor–critic loops)

/brain/hippocampus/* (episodic, replay)

/brain/cerebellum/* (forward models)

/brain/networks/* (DMN/SN/DAN/VAN)

/ml/* (RAG/KG, planners, verifiers)

/infra/* (Ray/K8s/Airflow/Kubeflow/MLflow/Spark/Dask)

📅 Overall Timeline & Status

Overall Progress: ~25%

Current Pillar: Pillar 2 (Neuromodulatory Systems)

Next Milestone: Finish neuromod → start hierarchical processing (Pillar 3)

8-Week Path: Complete pillars incl. cloud; 20-Week Path: Full AGI capability integration & productionization

📚 Key References (selected, high-impact)

Neocortex laminae/microcircuits: Harris & Shepherd 2015; Miyashita 2022. 


DA reward prediction error & BG actor–critic: Schultz 1998; Joel & Dayan 2002; Glimcher 2011. 


LC-NE adaptive gain: Aston-Jones & Cohen 2005. 


ACh in encoding/attention: Hasselmo 2004/2006. 


5-HT flexibility/reversal: Hyun et al. 2023. 


Sleep & consolidation: Born & Wilhelm 2012; Rasch & Born 2013. 


DMN: Raichle 2015; Menon 2023. 


DAN/VAN: Corbetta & Shulman 2002; Fox et al. 2006. 


Salience network: Seeley et al. 2007; Menon 2010. 


Small-world connectomics: Bassett & Bullmore 2016/2017; Bullmore & Sporns 

Critical periods & pruning: Hensch 2005; Spear 2013. 


Predictive coding & Free-Energy; GNW: Friston 2009/2010; Dehaene & Changeux 2011; Mashour et al. 2020. 


Appendix A — Efficiency & Robustness (Design Patterns)
efficiency_measures:
  computational: ["Sparse reps", "Selective attention", "Hierarchical abstraction", "Compute caching"]
  memory: ["Checkpointing", "Memory-mapped arrays", "Compression", "Lazy eval"]
  communication: ["Batching", "Async pipelines", "Priority routing", "Efficient codecs"]

robustness_measures:
  fault_tolerance: ["Redundant paths", "Graceful degradation", "Auto-recovery", "Self-healing"]
  adaptability: ["Online learning", "Dynamic reconfiguration", "Context switching", "Continuous improvement"]
  validation: ["Real-time consistency", "Cross-module checks", "Bio constraints", "Perf dashboards"]

Appendix B — Safety, Alignment, and Auditing

Runtime guards for tool calls (RBAC + affordance checks)

Policy compiler: declarative constraints → runtime monitors

Provenance: every artifact linked to inputs, prompts, code rev, data snapshot

Human-in-the-loop review for high-impact decisions; sandbox for self-mods

Incident response: anomaly triage, rollback, root-cause notebook

Appendix C — Benchmarks & Probes

Cognition: working memory spans, attentional blink, task-switching, reversal learning

Memory: episodic recall under lures (pattern separation probes), consolidation under sleep/wake protocols 


Reasoning: formal logic, math word problems, program induction

Planning: hierarchical puzzles; long-horizon tool use

Social: ToM stories/images; norm conflicts; moral dilemmas (explainable traces) 


Robustness: OOD suites, adversarial stress, calibration/abstention

Networks: DMN/SN/DAN/VAN switching latency; hub centrality; efficiency under load 


📌 One-Page Program Charter

Goal: A biologically-informed AGI architecture that is efficient, robust, explainable, and deployable at production scale.

Success: All 10 domains integrated; ≥90% bio-fidelity in target assays; ≥90% utilization and ≤100 ms latency; ≥99% reliability; scalable to 1000+ nodes.

Deliverable: A validated AGI brain simulation with cloud pipelines, safety and auditability, complete test suites, and documented neurobiological mapping.

## 📑 Appendix — Reference Catalog Samples

### Ion Channel Catalogue (Sample)
| Channel Family | Subtypes | Primary Location | Kinetics Ref |
|---------------|----------|------------------|--------------|
| Nav (Na+) | Nav1.1–1.9 | Axon initial segment, nodes | ion_channel_library.yaml ➜ `nav_family` |
| Kv (K+) | Kv1–Kv12 | Soma, dendrites | ion_channel_library.yaml ➜ `kv_family` |
| Cav (Ca2+) | Cav1–Cav3 | Presynaptic terminals | ion_channel_library.yaml ➜ `cav_family` |
| HCN | HCN1–4 | Dendrites | ion_channel_library.yaml ➜ `hcn_family` |

### Gene-Expression Marker Table (Sample)
| Marker | Cell Type | Functional Note | Source |
|--------|----------|-----------------|--------|
| NeuN | Mature neurons | Neuronal nuclei marker | gene_markers.csv |
| GFAP | Astrocytes | Intermediate filament protein | gene_markers.csv |
| OLIG2 | Oligodendrocyte lineage | Myelination driver | gene_markers.csv |
| IBA1 | Microglia | Immune surveillance | gene_markers.csv |

> Full tables reside in the referenced YAML/CSV files for programmatic loading.

## Appendix A — Deployment Profiles & Cost Model

```yaml
laptop_environment:
  neural_simulation: "PyTorch + Brian2 + NEURON"
  data_management: "SQLite + HDF5"
  registry: "MLflow local"
  resources: { max_memory: "16GB", max_cpu_cores: 8, gpu: "RTX optional" }
  capabilities: ["≤1M neurons", "STDP", "Neuromod systems", "Visual validation"]

cloud_burst:
  compute: "Ray + Kubernetes"
  training: "Distributed PyTorch"
  orchestration: "Apache Airflow"
  serving: "Ray Serve"
  cost: "Spot instances"
  capabilities: ["≤100M neurons", "Large-scale training", "Bio validation"]

production:
  cluster: "Kubernetes (EKS/GKE/AKS) + Ray"
  workflows: "Kubeflow + MLflow"
  data: "Spark + Dask"
  monitoring: "Prometheus + Grafana"
  costs: { reserved: "30-60% savings", autoscale: true }
  capabilities: ["≤100B neurons", "Full brain sim", "Continuous workloads"]

estimated_monthly:
  dev: "$0–50"
  test: "$50–200"
  prod: "$200–1000"  # spot/reserved, autoscaling
```

## Appendix B — System-Design Reference
Context Management: Versioned knowledge graph + vector store; RAG with symbolic nodes (e.g., LangGraph + Neo4j).
Memory Service: Episodic (event segmentation, infinite context), semantic (facts), consolidation policies; benchmark with memory-centric suites.
Workflow Engine: Deterministic orchestration of probabilistic modules; transactional rollbacks.
Symbolic Reasoning: Hybrid neural-symbolic; solvers (SAT/SMT/CAS); lifted regression planning.
Planning & Goal Management: Subgoal graphs, dependencies; HyperTree planning; classical planner interfaces.
Cross-Modal: Unified embeddings (text/vision/audio/video); early-exit on device; cached embeddings.
Security & Sandboxing: Tool sandbox; PII redaction; inversion/membership-inference defenses.
Observability: Full decision logs, context snapshots, rollbacks; provenance chains.
Self-Evolution: Tool discovery; fine-tune experts; governance reviews.
Hybrid Cloud & Edge: S3-backed snapshots; EC2/Spot/On-Demand balancing; optional quantum acceleration for combinatorial cases.