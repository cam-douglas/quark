*Stage 1 - Embryonic* 

**Version 2.1 – 2025-09-01**

> **Canonical** – This file supersedes all earlier Stage 1 roadmaps.  

Embryonic - Weeks 3–8: neural tube patterning seeds brain-region primordia.

Early vertebrate morphogenesis establishes neural axes and ventricular scaffold.

**Roadmap Status:** 🚨 **FOUNDATION LAYER ✅ COMPLETE** - 6 non-foundation tasks 🚨 PENDING 

Weeks 3–8: neural-tube folding plus SHH/BMP/WNT/FGF gradients establish fore-, mid-, hindbrain primordia; ventricles and meninges emerge, scaffolding future neurogenesis.

**Engineering Milestones — implementation tasks driving Stage 1 goals**

* [foundation-layer] ✅ **DONE** - Establish foundation-layer morphogen solver aligning with SHH/BMP/WNT/FGF gradients.
* [developmental-biology] ✅ **DONE** - Generate lineage-tagged neuroepithelial cells with human-compliant longitudinal validation.
* [foundation-layer] ✅ **DONE** - Excavate ventricular cavities (lateral, third, fourth, aqueduct) in voxel map. → [Detailed sub-tasks](../../state/tasks/roadmap_tasks/foundation_layer_detailed_tasks.md#11-ventricular-system-construction)
* [foundation-layer] ✅ **DONE** - Lay meninges scaffold (dura, arachnoid, pia) surrounding neural tube. → [Detailed sub-tasks](../../state/tasks/roadmap_tasks/foundation_layer_detailed_tasks.md#12-meninges-scaffold-construction)
* [brainstem] ✅ **DONE** - Segment brainstem subdivisions (midbrain, pons, medulla) with labels; deployed with Prometheus/Grafana monitoring. → Retrospective: `docs/reports/brainstem_segmentation_phase4_postmortem.md`, Dashboard: `management/configurations/project/grafana_dashboards/brainstem_segmentation.json`
* [cerebellum] 🚨 **PENDING** - Model cerebellar vermis, hemispheres, and deep nuclei microzones for future motor-control loops.
* [cortex-mapping] 🚨 **PENDING** - Generate cerebral lobe mapping table (frontal, parietal, temporal, occipital, limbic, insular) linked to functional subsystems.


**Biological Goals — desired biological outcomes for Stage 1**

* [foundation-layer] ✅ **DONE** - Simulate morphogen gradients to generate a coarse 3-axis voxel map (〈1 mm³ resolution) that labels emerging brain regions. **(KPI: segmentation_dice ✅ 0.267 baseline; experimental_accuracy ✅ 0.705 ACCEPTABLE)**
* [developmental-biology] ✅ **DONE** - Instantiate proto-cell populations with lineage tags for later differentiation. **(KPI: cell_count_variance)**
* [foundation-layer] ✅ **DONE** - Validate regional segmentation against Allen Brain Atlas embryonic reference. **(KPI: segmentation_dice ✅ 1.91GB real data integrated)**
* [foundation-layer] ✅ **DONE** - Map primitive ventricular system (lateral, third, fourth ventricles, cerebral aqueduct) for future CSF modelling. **(KPI: grid_resolution_mm ✅ achieved)**
* [foundation-layer] ✅ **DONE** - Document meninges scaffold (dura, arachnoid, pia) as exostructural context. **(KPI: meninges_mesh_integrity ✅ validated)**


**SOTA ML Practices (2025) — recommended methods**

* [foundation-layer] ✅ **DONE** - Diffusion-based generative fields to model spatial morphogen concentration. → [Detailed sub-tasks](../../state/tasks/roadmap_tasks/foundation_layer_detailed_tasks.md#22-advanced-ml-integration)
* [foundation-layer] ✅ **DONE** - Transformer-based graph neural nets (GNN-ViT hybrid) for 3-D segmentation with limited labels. → [Detailed sub-tasks](../../state/tasks/roadmap_tasks/foundation_layer_detailed_tasks.md#23-3d-segmentation-system)
* [curriculum-learning] 🚨 **PENDING** - Curriculum learning: train on simplified in-silico embryos → full morphology.
* [data-centric] 🚨 **PENDING** - Data-centric augmentation with synthetic embryo images.

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
