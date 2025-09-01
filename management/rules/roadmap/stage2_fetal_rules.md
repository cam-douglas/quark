*Stage 2 - Fetal* 

**Version 2.1 – 2025-09-01**

> **Canonical** – This file supersedes all earlier Stage 2 roadmaps.  

## Fetal - Coordinates neurogenesis and radial migration building cortical laminae

Weeks 9–38: explosive neurogenesis and radial migration assemble cortical layers.

**Roadmap Status:** 📋 Planned 

Fetal period (wks 9–38) drives massive neurogenesis and radial migration, building six-layer cortex under Reelin/Notch guidance.

**Engineering Milestones — implementation tasks driving Stage 2 goals**
* [foundation-layer] Advance foundation-layer to P2 Core Modules: instantiate six-layer cortical template and thalamic relay stubs.
* [hierarchical-processing] Activate hierarchical-processing P0 laminar scaffold validation harness.
* [developmental-biology] Populate brainstem (midbrain, pons, medulla) voxel regions with progenitor cell pools.
* [developmental-biology] Carve cerebellar vermis / hemispheres and embed deep nuclei placeholders.
* [developmental-biology] Compile initial ion-channel library (Nav/Kv/Cav/HCN) and attach to neurogenesis engine.
* [glia-integration] Expand `cell_type_library` to include glial classes (astrocyte, oligodendrocyte, microglia, ependymal) with basic interaction models.
* [electrophysiology] Parameterise Purkinje, granule, and motor neuron electrophysiology in `ion_channel_catalog`; attach Hodgkin-Huxley / AdEx presets.
* [synapse-schema] Extend connectivity schema to specify synapse types (excitatory AMPA/NMDA, inhibitory GABA/Gly, electrical gap junctions) including receptor subtype metadata.

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
**Functional Mapping (links to Appendix Part 1):**
- [Core Cognitive — Memory, Learning, Reasoning, Problem Solving](#$cap-1-core-cognitive)
- [Perception & World Modeling — Multimodal Perception, World Models, Embodiment](#$cap-2-perception-world-modeling)
- [Action & Agency — Planning, Decision-Making, Tool Use, Self-Improvement](#$cap-3-action-agency)

---
→ Continue to: [Stage 3 – Early Post-natal](stage3_early_post-natal_rules.md)

