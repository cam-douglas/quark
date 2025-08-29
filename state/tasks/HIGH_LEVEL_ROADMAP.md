# Project Quark: High-Level AGI Brain Simulation Roadmap

This document outlines the strategic roadmap for Project Quark, detailing the major phases of development from foundational neural simulation to advanced AGI capabilities. This roadmap is a living document and will be updated as the project evolves.

It is designed to be used in conjunction with the detailed `management/rules/ml_workflow.md`, which provides the technical implementation framework for each step.

---

## Phase 1: Foundational Scaffolding (Developmental Stage: Fetal)

**Goal:** Implement the basic neural dynamics and minimal structural scaffold for core brain regions, simulating the earliest stages of brain development.

**Key Milestones:**

- **1.1: Core Infrastructure & Data Strategy:**
  - Define and implement data schemas for neural activity, connectivity, and molecular data.
  - Set up experiment tracking and versioning infrastructure (e.g., DVC, MLflow).
  - Establish the baseline simulation environment (e.g., Brian2, NEST, or custom framework).

- **1.2: Thalamic Relay System:**
  - Model the thalamus as a basic information relay and gating system.
  - Implement simple routing of sensory information proxies.

- **1.3: Hippocampal Formation (Early Memory):**
  - Model basic Hebbian learning (STDP) for pattern encoding.
  - Simulate the initial formation of memory traces.

- **1.4: Basal Ganglia (Action Primitives):**
  - Implement primitive action selection through simple gating mechanisms.
  - No complex reinforcement learning at this stage.

- **1.5: Proto-Cortex Simulation:**
  - Simulate a basic multi-layered cortical sheet with local connectivity.
  - Implement homeostatic plasticity to ensure stable dynamics.

**Phase 1 Deliverable:** A running simulation of a fetal-stage brain demonstrating stable neural dynamics, basic information flow, and primitive synaptic plasticity.

---

## Phase 2: Emergence of Core Functions (Developmental Stage: Neonate N0)

**Goal:** Introduce more complex dynamics, including sleep cycles, neuromodulation, and the emergence of early cognitive functions.

**Key Milestones:**

- **2.1: Sleep Cycle & Memory Consolidation:**
  - Implement simulated sleep/wake cycles (e.g., slow-wave and REM-like states).
  - Model sleep-dependent memory consolidation in the hippocampus, strengthening patterns learned during "wakefulness".

- **2.2: Salience & Attention Networks:**
  - Develop a basic salience network to detect novel or important stimuli proxies.
  - Implement a simple attention mechanism to modulate information flow from the thalamus.

- **2.3: Default Mode Network (Proto-DMN):**
  - Implement an early-stage Default Mode Network for internal simulation and replay of learned patterns.

- **2.4: Reinforcement Learning in Basal Ganglia:**
  - Enhance the Basal Ganglia model with actor-critic reinforcement learning algorithms.
  - Introduce a dopamine-like reward signal to guide action selection.

**Phase 2 Deliverable:** A simulation that can learn simple tasks via reinforcement learning, demonstrates memory consolidation during sleep, and exhibits rudimentary attentional focus.

---

## Phase 3: Higher-Order Cognition (Developmental Stage: Early Postnatal N1)

**Goal:** Implement executive functions, expanded working memory, and integrated, goal-directed behavior.

**Key Milestones:**

- **3.1: Prefrontal Cortex (Executive Control):**
  - Implement a working memory module capable of holding and manipulating information over short periods.
  - Develop basic planning capabilities and goal-setting mechanisms.

- **3.2: Global Workspace Architecture (Consciousness Integration):**
  - Develop an "Architecture Agent" that implements a global workspace.
  - This system will integrate and broadcast information between different brain modules, forming a basis for simulated consciousness.

- **3.3: Cerebellar Modulation:**
  - Add a cerebellum model to fine-tune and coordinate activity from motor and cognitive loops.

- **3.4: Cross-modal Sensory Integration:**
  - Implement pathways for combining and processing information from multiple simulated sensory streams.

**Phase 3 Deliverable:** A simulation capable of solving multi-step problems, demonstrating flexible goal-directed behavior, and showing evidence of integrated information processing via the global workspace.

---

## Phase 4: AGI Capabilities & Full Validation

**Goal:** Validate the simulated brain against a comprehensive suite of cognitive and AGI benchmarks, and manage its long-term, continuous development.

**Key Milestones:**

- **4.1: Cognitive Benchmark Suite:**
  - Develop and run a suite of tests based on established cognitive science paradigms (e.g., working memory capacity tests, decision-making tasks, attentional blink).
  - Measure performance and compare against biological benchmarks.

- **4.2: Robustness & Adaptivity Testing:**
  - Introduce novel, unseen tasks and environments to measure the system's generalization and adaptation capabilities.
  - Test resilience to noise and partial information.

- **4.3: Metacognition & Self-Modeling:**
  - Implement and test the system's ability to monitor its own cognitive states (e.g., uncertainty, confidence).

- **4.4: Long-term Lifecycle Management:**
  - Demonstrate stable, continuous learning over extended simulation runs without catastrophic forgetting.
  - Implement protocols for updating and evolving the model while preserving core identity and memories.

**Phase 4 Deliverable:** A comprehensive evaluation report detailing the AGI's performance on cognitive, adaptive, and metacognitive tasks, along with a framework for its ongoing lifecycle management.
