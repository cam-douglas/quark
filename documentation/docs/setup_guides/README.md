# 🧠 Quark Brain Simulation Framework

A newborn cognitive brain simulation framework with biologically-grounded neural architectures.

## 🚀 Quick Start

### Command Line
```bash
# Install the package
pip install quark-brain-simulation

# Run a basic simulation
quark-brain --connectome src/config/connectome_v3.yaml --steps 10 --stage F

# Or use the alternative command
quark-sim --connectome src/config/connectome_v3.yaml --steps 10 --stage F
```

### Python API
```python
from brain_modules.conscious_agent.main.brain_launcher_v3 import Brain

# Create and run a brain simulation
brain = Brain(
    connectome_path="project_management/configurations/config/connectome_v3.yaml",
    stage="F",
    steps=10
)
brain.run()
```

### DeepSeek-R1 Knowledge Oracle (New!)
```python
# Observe natural emergence with knowledge support
from src.core.natural_emergence_integration import create_natural_emergence_monitor

# Create knowledge observer (zero simulation influence)
monitor = create_natural_emergence_monitor("quark_brain")

# Document natural development at pillar milestones
observation = monitor.observe_brain_state(
    brain_state=brain.get_state(),
    pillar_stage="PILLAR_2_NEUROMODULATION",
    natural_progression=True
)

# Get scientific insights without affecting simulation
insights = observation.get('knowledge_insights', '')
```

## 🧬 Features

### Core Brain Modules
- **Prefrontal Cortex (PFC)**: Executive control, planning, reasoning
- **Basal Ganglia (BG)**: Action selection, reinforcement learning
- **Thalamus**: Information relay, attentional modulation
- **Working Memory**: Short-term memory buffers
- **Hippocampus**: Episodic memory, pattern completion
- **Default Mode Network (DMN)**: Internal simulation, self-reflection
- **Salience Networks**: Attention allocation, novelty detection

### DeepSeek-R1 Knowledge Oracle Integration 🆕
- **Natural Emergence Observer**: Document and interpret natural brain development
- **Read-Only Knowledge Resource**: Scientific insights without simulation interference
- **Research Documentation**: Export emergence patterns for publication
- **Zero Influence Guarantee**: Preserves natural developmental progression

### Enhanced Developmental Framework
- **Biological Validation**: 8 developmental markers with biological accuracy
- **Multi-Scale Integration**: Molecular → Cellular → Circuit → System interactions
- **Sleep-Consolidation**: Sophisticated NREM/REM cycles with memory consolidation
- **Progressive Capacity**: 7 cognitive capacities with stage-specific constraints

### Developmental Stages
- **F (Fetal)**: Basic neural dynamics, minimal scaffold (8-20 weeks gestation)
- **N0 (Neonate)**: Sleep cycles, consolidation, salience switching (20-40 weeks gestation)
- **N1 (Early Postnatal)**: Expanded working memory, cerebellar modulation (0-12 weeks postnatal)

### Advanced Features
- **Sleep Cycles**: NREM/REM alternation with consolidation and replay
- **Neuromodulators**: DA/NE/ACh/5‑HT global signals
- **Telemetry**: Real-time monitoring of brain state
- **Connectome Configurations**: Multiple brain connectivity patterns
- **Emergent Properties**: Consciousness, learning, development, intelligence detection
- **Validation Metrics**: Biological accuracy, developmental progression, multi-scale integration

## 📊 Example Simulations

### Quick Start (Fetal Stage)
```bash
quark-brain --connectome src/config/connectome_v3.yaml --steps 5 --stage F
```

### Enhanced Brain Simulation (v4)
```bash
python src/core/brain_launcher_v4.py --connectome src/config/connectome_v3.yaml --steps 100 --stage F --log_csv enhanced_metrics.csv --summary
```

### Sleep Cycle Simulation (Neonate)
```bash
quark-brain --connectome src/config/connectome_v3.yaml --steps 20 --stage N0
```

### Memory Expansion (Early Postnatal)
```bash
quark-brain --connectome src/config/connectome_v3.yaml --steps 15 --stage N1
```

## 🏗️ Architecture

The framework implements a biologically-grounded architecture with:

- **8 Expert Domains**: Computational Neuroscience, Cognitive Science, Machine Learning, Systems Architecture, Developmental Neurobiology, Philosophy of Mind, Data Engineering, Ethics & Safety
- **Strict Hierarchy**: Architecture Agent → Core Modules → Specialized Subsystems
- **Stage Constraints**: Preserved hierarchy across F → N0 → N1 stages
- **Ethical Guardrails**: Safety protocols, control boundaries, auditability

## 📦 Installation

### From PyPI
```bash
pip install quark-brain-simulation
```

### From Source
```bash
git clone https://github.com/cam-douglas/quark-brain-simulation.git
cd quark-brain-simulation
pip install -e .
```

## 🔧 Configuration

### Connectome Files
The framework uses YAML configuration files to define brain connectivity:

- `src/config/connectome_v3.yaml`: Latest configuration with all modules
- `src/config/connectome_v2.yaml`: Previous version
- `src/config/connectome.yaml`: Basic configuration

### Custom Parameters
You can customize simulation parameters through YAML files or command-line arguments.

## 📚 Documentation

- **Core Scientific Foundation**: `docs/01_CORE_SCIENTIFIC_FOUNDATION.md`
- **Technical Architecture**: `docs/02_TECHNICAL_ARCHITECTURE.md`
- **Research Applications**: `docs/03_RESEARCH_APPLICATIONS.md`
- **Implementation Guide**: `docs/04_IMPLEMENTATION_GUIDE.md`
- **Enhanced Framework**: `docs/05_ENHANCED_FRAMEWORK.md`

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **GitHub Repository**: https://github.com/cam-douglas/quark-brain-simulation
- **Hugging Face Space**: https://huggingface.co/spaces/cam-douglas/quark-brain-simulation
- **PyPI Package**: https://pypi.org/project/quark-brain-simulation/

## 🙏 Acknowledgments

This framework builds upon decades of neuroscience research and computational modeling. We thank the research community for their foundational work in understanding brain development and cognition.

---

**Quark Brain Simulation** - Simulating the emergence of consciousness, one neuron at a time. 🧠✨
