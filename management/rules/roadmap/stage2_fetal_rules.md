*Stage 2 - Fetal* 

**Version 2.1 – 2025-09-01**

> **Canonical** – This file supersedes all earlier Stage 2 roadmaps.  

## Fetal - Coordinates neurogenesis and radial migration building cortical laminae

Weeks 9–38: explosive neurogenesis and radial migration assemble cortical layers.

**Roadmap Status:** 🚨 **FOUNDATION LAYER ✅ COMPLETE** - Other tasks 🚨 PENDING

Fetal period (wks 9–38) drives massive neurogenesis and radial migration, building six-layer cortex under Reelin/Notch guidance.

**Engineering Milestones — implementation tasks driving Stage 2 goals**

* [hierarchical-processing] 🚨 **PENDING** - Advance foundation-layer to P2 Core Modules: instantiate six-layer cortical template and thalamic relay stubs.
* [hierarchical-processing] 🚨 **PENDING** - Activate hierarchical-processing P0 laminar scaffold validation harness.
* [developmental-biology] 🚨 **PENDING** - Populate brainstem (midbrain, pons, medulla) voxel regions with progenitor cell pools.
* [developmental-biology] 🚨 **PENDING** - Carve cerebellar vermis / hemispheres and embed deep nuclei placeholders.
* [developmental-biology] 🚨 **PENDING** - Compile initial ion-channel library (Nav/Kv/Cav/HCN) and attach to neurogenesis engine.
* [glia-integration] 🚨 **PENDING** - Expand `cell_type_library` to include glial classes (astrocyte, oligodendrocyte, microglia, ependymal) with basic interaction models.
* [electrophysiology] 🚨 **PENDING** - Parameterise Purkinje, granule, and motor neuron electrophysiology in `ion_channel_catalog`; attach Hodgkin-Huxley / AdEx presets.
* [synapse-schema] 🚨 **PENDING** - Extend connectivity schema to specify synapse types (excitatory AMPA/NMDA, inhibitory GABA/Gly, electrical gap junctions) including receptor subtype metadata.

**Biological Goals — desired biological outcomes for Stage 2**

* [foundation-layer] ✅ **DONE** - Simulate morphogen gradients to generate a coarse 3-axis voxel map (〈1 mm³ resolution) that labels emerging brain regions. **(KPI: segmentation_dice ✅ 0.267 baseline established)**
* [developmental-biology] 🚨 **PENDING** - Instantiate proto-cell populations with lineage tags for later differentiation. **(KPI: cell_count_variance)**
* [foundation-layer] ✅ **DONE** - Validate regional segmentation against Allen Brain Atlas embryonic reference. **(KPI: segmentation_dice ✅ 1.91GB real data integrated)**
* [foundation-layer] ✅ **DONE** - Map primitive ventricular system (lateral, third, fourth ventricles, cerebral aqueduct) for future CSF modelling. **(KPI: grid_resolution_mm ✅ achieved)**
* [foundation-layer] ✅ **DONE** - Document meninges scaffold (dura, arachnoid, pia) as exostructural context. **(KPI: meninges_mesh_integrity ✅ validated)**
* [neurogenesis] 🚨 **PENDING** - Generate 1e9 neurons through stochastic birth-process simulation. **(KPI: neuron_count_error_pct <=5%)**
* [radial-migration] 🚨 **PENDING** - Implement RL-based radial migration with PPO + curiosity bonus. **(KPI: laminar_accuracy >=0.80)**
* [cortical-lamination] 🚨 **PENDING** - Build six-layer cortical template with proper laminar organization. **(KPI: laminar_thickness validation)**

**SOTA ML Practices (2025)**

* [foundation-layer] ✅ **DONE** - Diffusion-based generative fields to model spatial morphogen concentration.
* [hierarchical-processing] 🚨 **PENDING** - Reinforcement-learning cellular automata (policy: migration vector; reward: laminar ordering).
* [foundation-layer] ✅ **DONE** - Transformer-based graph neural nets (GNN-ViT hybrid) for 3-D segmentation with limited labels.
* [developmental-biology] 🚨 **PENDING** - Use WandB sweeps for hyper-parameter exploration of proliferation rates.
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
→ Continue to: [Stage 3 – Early Post-natal](stage3_early_post-natal_rules.md)