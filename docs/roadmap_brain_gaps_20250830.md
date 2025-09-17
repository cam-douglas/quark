# Brain Roadmap Gap Analysis — 2025-08-30

This document enumerates brain-related topics **present in the Human Brain Wikipedia article** but **missing or insufficiently covered** in the current `master_roadmap.md`.  Each bullet lists the missing topic(s) followed by concrete suggestions for integration.

## 1. Macro-Level Structure
* **Brainstem subdivisions (midbrain, pons, medulla)** — add explicit modelling goals and validation tests for each sub-region’s sensorimotor and autonomic roles.
* **Cerebellar sub-regions (vermis, hemispheres, deep nuclei)** — extend Pillar 1 description to include cerebellar microzones and cerebro-cerebellar loops.
* **Cerebral lobes & cortical areas (frontal, parietal, temporal, occipital, limbic, insular)** — insert a mapping table linking lobes to functional subsystems.

## 2. Microanatomy & Support Cells
* **Neuron classes beyond pyramidal/interneurons** (e.g., Purkinje, granule, motor neurons) — introduce cell-type library with electrophysiological parameters.
* **Glial cells** (astrocytes, oligodendrocytes, microglia, ependymal) — add glia-neuron interaction layer (ion buffering, myelination, immune response).
* **Synapse diversity** (excitatory, inhibitory, electrical gap junctions) — specify synaptic types in machine-readable connectivity schema.
* **Meninges & ventricular system** — briefly document anatomical context; not simulated but relevant for CSF dynamics.

## 3. Developmental Biology
* **Neural tube patterning (forebrain, midbrain, hindbrain)** — embed milestones before Pillar 7.
* **Neurogenesis & radial migration** — add tasks for developmental timeline simulation.
* **Programmed cell death (apoptosis)** & **synaptic pruning** — expand existing “critical periods & pruning” section with quantitative targets.
* **Myelination timeline** — include age/stage-dependent myelin growth impacting conduction speed.

## 4. Physiology & Neurotransmission
* **Primary fast neurotransmitters:**
  * Glutamate (AMPA, NMDA, kainate) — excitatory baseline
  * GABA & glycine — inhibitory control
  * Glycine (spinal/brainstem) — reflex arcs
* **Secondary modulators:** Histamine, opioid peptides, endocannabinoids — add to neuromodulatory checklist.
* **Ion channel dynamics** — reference Hodgkin-Huxley / adaptive exponential models for spiking variety.
* **Cerebrospinal fluid (CSF) flow & glymphatic clearance** — outline non-neuronal homeostasis subsystem.
* **Energy metabolism (glucose, ketone, astrocyte-neuron lactate shuttle)** — add metabolic constraints to robustness metrics.

## 5. Gene & Protein Expression
* **Transcriptomic gradients & cell-type markers** (NeuN, GFAP, OLIG2, IBA1) — propose integration with `agi_enhanced_connectome.yaml` for cell identity tagging.
* **Proteomics of synaptic proteins (SNARE, PSD-95, synaptophysin)** — annotate synapse specs for plasticity models.

## 6. Vascular & Barrier Systems
* **Blood supply (internal carotid, vertebral, Circle of Willis)** — incorporate perfusion constraints for energy models.
* **Blood–brain barrier & pericyte regulation** — add safety layer for molecular transport simulation (placeholder for future toxin / drug modules).

## 7. Ion Channels & Electrophysiology
* **Voltage-gated channels:** Na<sup>+</sup> (Nav1.1–1.9), K<sup>+</sup> (Kv1–Kv12), Ca<sup>2+</sup> (Cav1–Cav3), HCN (I<sub>h</sub>), and Cl<sup>−</sup> channels — parameterize firing phenotypes across neuron classes.
* **Ligand-gated channels:** AMPA, NMDA, kainate (glutamatergic); GABA<sub>A</sub>, glycine receptors (inhibitory); nicotinic ACh, 5-HT<sub>3</sub>, P2X (ATP) — include receptor subtype distribution maps.
* **Metabotropic receptors & second-messenger pathways:** mGluR (group I–III), GABA<sub>B</sub>, muscarinic ACh (M1–M5), dopamine (D1–D5), adrenergic (α, β), serotonergic (5-HT1–7), histamine (H1–H4), cannabinoid (CB1/2), opioid (μ, κ, δ) — expand neuromodulatory modeling scope.

## 8. Expanded Structural Items
* **Ventricular System:** detailed node list (lateral, third, fourth ventricles, cerebral aqueduct); consider CSF flow modelling for waste clearance.
* **Protective Layers:** dura, arachnoid, pia; integrate as exostructural context for mechanical / vascular constraints.
* **Cranial Nerves (I–XII):** add interface stubs for sensorimotor I/O; map to corresponding brainstem nuclei.
* **Named Tracts & Commissures:** corpus callosum, anterior/posterior commissures, internal capsule, corticospinal tract — include connectivity templates.
* **Gyri/Sulci & Brodmann Areas:** propose level-of-detail toggle for high-resolution cortical mapping.

---
### Immediate Roadmap Edits
1. **Pillar 1 (Foundation Layer):** embed cerebellar and brainstem sub-region details; add glial interaction placeholder.
2. **Pillar 2 (Neuromodulatory Systems):** extend transmitter list to include glutamate, GABA, glycine, histamine, peptides.
3. **Pillar 3 (Hierarchical Processing):** insert lobe/area mapping table; label microanatomical cell-type diversity.
4. **Pillar 7 (Developmental Biology):** include neural tube patterning, myelination timeline, apoptosis metrics.
5. **Robustness Metrics:** append metabolic and vascular fidelity KPIs.
6. **Machine-Readable Specs:** update `memory_architecture`, `learning_systems`, etc., with new cell-type and synapse keys.

> **Note:** These additions maintain alignment with the existing 10 AGI capability domains while closing biological fidelity gaps.

---
**Additional Immediate Roadmap Edits**
7. **Pillar 2:** incorporate explicit ion-channel catalogue with default kinetics libraries.
8. **Pillar 4 (Connectomics):** add tractography targets for major white-matter bundles.
9. **Testing Framework:** new electrophysiology benchmarks (stimulus-response curves, refractory periods).
