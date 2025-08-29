# 🧠 SmallMind Simulation Collection

This directory contains a comprehensive collection of brain development and neural simulation tools, frameworks, and research implementations.

## 🚀 Core Simulation Files

### Brain Physics Simulators
- **`optimized_brain_physics.py`** - Ultra-optimized brain physics simulator using PyTorch 2.x, FlashAttention-2, and advanced optimization techniques
- **`dual_mode_simulator.py`** - Hybrid simulator supporting MuJoCo (physical), NEST (neural), and combined modes

### Specialized Simulators
- **`fetal_anatomical_simulation.py`** - Fetal brain anatomical development simulation with MRI generation
- **`morphogen_physics.py`** - Morphogen gradient and signaling pathway simulation
- **`tissue_mechanics.py`** - Tissue growth, deformation, and mechanical properties simulation
- **`neural_simulator.py`** - Neural network dynamics and plasticity simulation
- **`neural_visualization.py`** - Advanced neural network visualization tools

### Framework Interfaces
- **`mujoco_interface.py`** - MuJoCo physics engine interface for brain development
- **`nest_interface.py`** - NEST neural simulation interface
- **`visit_interface.py`** - Visit visualization framework interface
- **`worm_browser_integration.py`** - Worm browser integration for simulation visualization

### Data and Resources
- **`enhanced_data_resources.py`** - Comprehensive neuroscience data integration (DANDI, OpenNeuro, Allen Brain Atlas, etc.)
- **`simulation_runner.py`** - Main simulation orchestration and management system

### Configuration
- **`brain_development.xml`** - MuJoCo brain development model configuration
- **`__init__.py`** - Package initialization

## 🏗️ Simulation Frameworks

### CompuCell3D
Complete CompuCell3D framework for cellular automata and multi-scale modeling:
- Cellular automata simulation
- Multi-scale modeling capabilities
- Tissue development simulation
- Biochemical network integration

### Visit Visualization
Complete Visit visualization framework:
- Scientific visualization tools
- 3D rendering capabilities
- Data analysis and plotting
- Real-time visualization

### Additional Frameworks
- **`cc3d/`** - CompuCell3D core components
- **`visit_env/`** - Visit environment configuration
- **`conda-recipes/`** - Conda package recipes for dependencies
- **`conda-recipes-compucell3d/`** - CompuCell3D-specific conda recipes

## 🔬 Research Applications

### Brain Development Research
- Fetal brain anatomical development
- Neural migration and axon guidance
- Cortical folding simulation
- White matter development

### Neural Network Research
- Spiking neural networks
- Synaptic plasticity
- Network connectivity analysis
- Learning and memory simulation

### Physics-Based Modeling
- Tissue mechanics and growth
- Morphogen gradient dynamics
- Biomechanical interactions
- Multi-scale physics integration

## 🚀 Performance Features

### Optimized Brain Physics
- FlashAttention-2 for efficient attention computation
- PyTorch 2.0 torch.compile optimization
- Mixed precision training
- Distributed training support
- Memory optimization techniques

### Multi-Modal Simulation
- Physical simulation (MuJoCo)
- Neural simulation (NEST)
- Hybrid physical-neural simulation
- Real-time visualization

## 📊 Data Integration

### Neuroscience Data Sources
- DANDI Archive (NWB 2.0+)
- OpenNeuro (BIDS 2.0+)
- Allen Brain Atlas (API v3+)
- MICrONS/BossDB
- Human Connectome Project
- NeuroMorpho
- CRCNS datasets

### Fetal Brain Simulation Tools
- FaBiAN: Synthetic fetal brain MRI phantoms
- 4D Embryonic Brain Atlas
- ReWaRD: Prenatal visual signal simulation
- Conditional Fetal Brain Atlas

## 🛠️ Usage

### Basic Simulation
```python
from simulation_runner import BrainDevelopmentSimulation

# Create simulation
sim = BrainDevelopmentSimulation()
results = sim.run_simulation()
```

### Optimized Brain Physics
```python
from optimized_brain_physics import OptimizedBrainPhysicsSimulator, SimulationConfig

# Configure simulation
config = SimulationConfig(
    hidden_dim=2048,
    num_layers=24,
    use_flash_attention=True
)

# Run simulation
simulator = OptimizedBrainPhysicsSimulator(config)
```

### Dual-Mode Simulation
```python
from dual_mode_simulator import DualModeBrainSimulator

# Run hybrid simulation
simulator = DualModeBrainSimulator(simulation_mode="hybrid")
```

## 🔧 Dependencies

### Core Dependencies
- PyTorch 2.x
- NumPy
- Matplotlib
- Pandas

### Optional Dependencies
- MuJoCo (for physical simulation)
- NEST (for neural simulation)
- Visit (for visualization)
- FlashAttention-2 (for optimization)
- vLLM (for inference optimization)

## 📚 References

- FaBiAN: Synthetic fetal brain MRI phantoms
- 4D Embryonic Brain Atlas: Deep learning atlas
- ReWaRD: Prenatal visual signal simulation
- CompuCell3D: Multi-scale modeling
- NEST: Neural simulation
- MuJoCo: Physics simulation

## 🎯 Research Potential

This collection provides a comprehensive toolkit for:
- Brain development research
- Neural network modeling
- Multi-scale simulation
- Physics-based modeling
- Data-driven neuroscience
- Educational applications
- Clinical research applications
