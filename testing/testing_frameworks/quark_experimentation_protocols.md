# 🧪 quark experiment evaluation & testing protocols (slm + llm hybrid edition)

**Priority Status**: 🔴 HIGH PRIORITY - Core Infrastructure Rule
**Integration Level**: Main Rules Section - Primary Testing & Evaluation Framework
**Dependencies**: Brain Architecture, ML Infrastructure, Development Roadmaps
**Last Updated**: August 21, 2025

## Project Context: QUARK – Brain Simulation ML Framework

QUARK is a brain-inspired ML framework integrating biologically-mapped architectures, complexity evolution agents, and cognitive simulation. This evaluation protocol is updated to reflect hybrid modular systems that combine Small Language Models (SLMs) and Large Language Models (LLMs), based on the research paper *"Small Language Models are the Future of Agentic AI" (arXiv:2506.02153)*.

This approach allows efficient task decomposition, performance scaling, and formatting consistency aligned with QUARK's biologically modular architecture.

---

## ✅ 1. General ML Evaluation Best Practices

### 🎯 Dataset Splitting
- Strict `train/val/test` splits with 5-fold cross-validation.
- Preserve biologically-inspired temporal consistency.

### 📈 Metrics
- **Classification**: Accuracy, F1-score, AUC
- **Regression**: MSE, MAE, R²
- **Language/Generation**: Perplexity, BLEU, ROUGE, Hallucination Rate
- Add confidence intervals and bootstrap error bounds.

### 🔍 Robustness & Ablation
- Inject noise to test sensory robustness.
- Ablate modules (e.g., simulated hippocampus, PFC) to analyze causal behavior.

### 🧠 Biological Plausibility
- Use `Brain-Score`, `NeuralBench`, `Algonauts` to validate simulation alignment with brain activity.

---

## 🧠 2. AGI & Brain Simulation Testing (Task + Architecture-Specific)

### 🧪 Task Hierarchy for Cognitive Evaluation

| Level | Domain | Example Task | Purpose |
|-------|--------|--------------|---------|
| L1 | Sensory Encoding | Visual/audio classification | Sensory cortex benchmarking |
| L2 | Motor Control | Maze navigation | Basal ganglia modeling |
| L3 | Working Memory | N-back, sequence recall | PFC alignment |
| L4 | Reasoning | Raven's matrices | General intelligence & abstraction |
| L5 | Language | Dialogue, QA, analogies | Language cognition |
| L6 | Meta-cognition | Self-reflection, error correction | Consciousness testing |

### 🧬 Simulation Benchmarks
- Use open environments: **PsychLab**, **BabyAI**, **Crafter**, **Procgen**, **Meta-World**.
- Track **zero-shot**, **continual learning**, **catastrophic forgetting**.

### 🧠 Neurocomputational Fidelity
- Match activity to fMRI, EEG, and neural spike datasets.
- Measure network modularity, graph centrality, and synchrony.

### 📊 Consciousness Evaluation
- Score models on:
  - Global Workspace Theory tasks
  - Attention filtering
  - Volitional timing (Libet clock tasks)

---

## 🤖 3. Hybrid SLM + LLM Evaluation

### Modular Assignment Strategy

| Component | Role | Performance Focus |
|----------|------|-------------------|
| LLM | Planner / Orchestrator | Reasoning, abstraction, routing accuracy |
| SLM | Sub-agent / Toolcaller | Format compliance, speed, token efficiency |
| Combined | Integrated execution | Switching accuracy, cost efficiency |

### Specialized Metrics

- **SLM**: Format fidelity, latency, hallucination rate, repeatability
- **LLM**: Planning coherence, misrouting rate, abstraction depth
- **Hybrid**: Module switching accuracy, throughput, compute cost

### Evaluation Patterns
- Use **uncertainty-based routing**: delegate low-uncertainty tasks to SLM, escalate to LLM when confidence drops.
- Track format deviation, fallback rates, and cost savings from SLM-first execution.

---

## 🔬 4. Experimental Design Template (Hybrid-Aware)

**Experiment Title**: Hybrid Task Execution with SLM–LLM Routing  
**Objective**: Test cost/performance trade-off using modular inference.  
**Setup**:
- Task: Tool-based QA + reasoning
- Routing: Uncertainty threshold triggers LLM fallback
**Control**: LLM-only baseline  
**Metrics**: Task accuracy, hallucination rate, cost per task, switching accuracy  
**Neuroalignment**: Evaluate biological plausibility across modules

---

## 📋 5. Performance Tracker Template

| Module | Task | Accuracy | Latency | Format Compliance | Neuroaligned? |
|--------|------|----------|---------|-------------------|----------------|
| Hippocampus | Seq Recall | 86% | 0.32s | N/A | ✓ |
| PFC (SLM) | N-back | 91% | 0.18s | 100% | ✓ |
| LLM Planner | Abstract QA | 88% | 0.78s | N/A | ✗ |
| Hybrid (SLM+LLM) | Tool use + plan | 92% | 0.48s | 98% | ✓ |

---

## ✅ 6. Final Notes

- **Use Wolfram Brain** for symbolic verification, task validation, and alignment visualization.
- Optimize **SLM fine-tuning** to reduce LLM reliance in format-heavy subtasks.
- Maintain log of task delegation outcomes to improve routing policies.
- Use your existing notebook infrastructure (339+ experiments) to track, compare, and version performance.

---

**Maintained by**: QUARK Development Team  
**Last updated**: August 21, 2025

---

## 🔗 Integration & Roadmap Connections

### 📍 Main Rules Integration
This protocol is integrated into the **Main Rules Section** as a **HIGH PRIORITY** framework rule that governs all experimentation and testing activities across QUARK.

### 🗺️ Roadmap Dependencies
- **Phase 1**: Core Testing Infrastructure (Current)
- **Phase 2**: Hybrid SLM+LLM Integration (Q4 2024)
- **Phase 3**: Advanced Neuroalignment Testing (Q1 2025)
- **Phase 4**: Production Deployment & Scaling (Q2 2025)

### 🔄 Cross-Framework Integration
- **Brain Architecture**: Direct integration with neural core modules
- **ML Infrastructure**: Unified evaluation pipeline
- **Development Workflows**: Automated testing and validation
- **Performance Tracking**: Real-time metrics and optimization

### 📋 Related Rules & Protocols
- **RULE-001**: Brain Architecture Development Standards
- **RULE-002**: ML Model Validation Protocols  
- **RULE-003**: Performance Benchmarking Standards
- **RULE-004**: Neuroalignment Validation Framework
- **RULE-005**: Hybrid System Integration Protocols

### 🎯 Implementation Checklist
- [x] **Integrate with existing testing frameworks** - Core experiment framework implemented and integrated
- [x] **Connect to performance tracking systems** - Performance metrics and tracking system implemented
- [x] **Link with development roadmaps** - Integrated with main rules and roadmap connections established
- [🔄] **Establish automated validation pipelines** - Framework ready, automation deployment in progress
- [🔄] **Deploy monitoring and alerting systems** - Framework ready, system deployment in progress

**Implementation Status**: 3/5 Complete (60%) - Core infrastructure deployed and operational
**Next Milestone**: Complete automation and monitoring deployment (Target: September 2024)
