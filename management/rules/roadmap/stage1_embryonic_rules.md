*Stage 1 - Embryonic* 

**Version 2.1 – 2025-09-01**

> **Canonical** – This file supersedes all earlier Stage 1 roadmaps.  

Embryonic - Weeks 3–8: neural tube patterning seeds brain-region primordia.

Early vertebrate morphogenesis establishes neural axes and ventricular scaffold.

**Roadmap Status:** 📋 In Progress 

Weeks 3–8: neural-tube folding plus SHH/BMP/WNT/FGF gradients establish fore-, mid-, hindbrain primordia; ventricles and meninges emerge, scaffolding future neurogenesis.

**Engineering Milestones — implementation tasks driving Stage 1 goals**

* [foundation-layer] Establish foundation-layer morphogen solver aligning with SHH/BMP/WNT/FGF gradients.
* [developmental-biology] Generate lineage-tagged neuroepithelial cells for downstream proliferation.
* [foundation-layer] Excavate ventricular cavities (lateral, third, fourth, aqueduct) in voxel map.
* [foundation-layer] Lay meninges scaffold (dura, arachnoid, pia) surrounding neural tube.
* [brainstem] Segment brainstem subdivisions (midbrain, pons, medulla) with dedicated sensorimotor/autonomic labels.
* [cerebellum] Model cerebellar vermis, hemispheres, and deep nuclei microzones for future motor-control loops.
* [cortex-mapping] Generate cerebral lobe mapping table (frontal, parietal, temporal, occipital, limbic, insular) linked to functional subsystems.


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

**Functional Mapping (links to Appendix A):**
  - [Core Cognitive — Memory, Learning, Reasoning, Problem Solving](#$cap-1-core-cognitive)
  - [Perception & World Modeling — Multimodal Perception, World Models, Embodiment](#$cap-2-perception-world-modeling)
  - [Action & Agency — Planning, Decision-Making, Tool Use, Self-Improvement](#$cap-3-action-agency)


---
→ Continue to: [Stage 2 – Fetal](stage2_fetal_rules.md)
