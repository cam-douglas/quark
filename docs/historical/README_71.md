# Quark Brain Simulation Notebooks

This directory contains Jupyter notebooks for interactive brain simulation development, training, and analysis.

## 📁 Directory Structure

```
notebooks/
├── consciousness_integration/    # Connection utilities
│   └── consciousness_bridge.py   # Bridge to main consciousness agent
├── training/                     # Model training notebooks
│   └── interactive_model_training.ipynb
├── simulation/                   # Real-time simulation notebooks
│   └── brain_component_simulation.ipynb
├── analysis/                     # Data analysis notebooks
│   └── data_exploration_analysis.ipynb
├── prototyping/                  # Experimental notebooks
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

1. **Activate your virtual environment:**
   ```bash
   source venv/bin/activate  # From project root
   ```

2. **Install Jupyter (if not already done):**
   ```bash
   pip install notebook jupyterlab matplotlib seaborn pandas scikit-learn
   ```

3. **Start Jupyter:**
   ```bash
   # Option 1: Classic Notebook
   jupyter notebook
   
   # Option 2: JupyterLab (recommended)
   jupyter lab
   ```

4. **Navigate to the notebooks directory** and start exploring!

## 📚 Notebook Guide

### 🧠 Consciousness Bridge (`consciousness_integration/consciousness_bridge.py`)

**Purpose:** Connect notebooks to your main consciousness agent as a tool.

**Key Functions:**
- `quick_connect(session_id)` - Fast connection setup
- `bridge.get_brain_state()` - Get current brain state
- `bridge.run_simulation_step()` - Execute simulation steps
- `bridge.train_component()` - Train specific components
- `bridge.visualize_brain_activity()` - Create activity visualizations

**Example Usage:**
```python
from consciousness_bridge import quick_connect
bridge = quick_connect('my_session')
state = bridge.get_brain_state()
bridge.visualize_brain_activity()
```

### 🎯 Interactive Model Training (`training/interactive_model_training.ipynb`)

**Purpose:** Train and experiment with brain simulation components interactively.

**Features:**
- Connect to consciousness agent for training
- Generate synthetic training data for brain components
- Monitor training progress with real-time visualizations
- Compare performance across different components
- Export trained models and training histories

**Use Cases:**
- Experiment with different learning rates and architectures
- Train specific brain components (PFC, working memory, etc.)
- Validate training performance with custom metrics
- Debug training issues interactively

**Workflow:**
1. Connect to consciousness agent
2. Configure training parameters
3. Generate or load training data
4. Run interactive training loops
5. Visualize results and save models

### 🔬 Brain Component Simulation (`simulation/brain_component_simulation.ipynb`)

**Purpose:** Test and simulate individual brain components and their interactions.

**Features:**
- Real-time brain simulation steps
- Component-level testing and debugging
- Sleep/wake cycle simulation
- Neural activity visualization
- Network connectivity analysis

**Use Cases:**
- Test individual brain components
- Debug component interactions
- Simulate sleep/wake cycles
- Monitor neural activity patterns
- Validate biological plausibility

**Workflow:**
1. Initialize simulation environment
2. Configure simulation parameters
3. Run real-time simulation steps
4. Visualize activity patterns
5. Analyze component interactions

### 📊 Data Exploration & Analysis (`analysis/data_exploration_analysis.ipynb`)

**Purpose:** Comprehensive analysis of brain simulation data and training results.

**Features:**
- Load data from multiple sessions
- Statistical analysis and hypothesis testing
- Interactive data filtering and visualization
- Compare training results across configurations
- Export insights and reports

**Use Cases:**
- Analyze training performance trends
- Explore consciousness emergence patterns
- Compare different simulation configurations
- Generate research insights and reports
- Validate experimental hypotheses

**Workflow:**
1. Load data from various sources
2. Explore data structure and quality
3. Perform statistical analyses
4. Create comprehensive visualizations
5. Export findings and reports

## 💡 Best Practices

### Session Management
- Use descriptive session IDs (e.g., `training_working_memory_20240119`)
- Save important sessions for future reference
- Document experimental parameters and findings

### Data Organization
- Store training results in `analysis/training_sessions/`
- Keep simulation outputs organized by date/session
- Use consistent naming conventions

### Visualization
- Always include titles, labels, and legends
- Use consistent color schemes across notebooks
- Save important plots for presentations/papers

### Performance
- Monitor memory usage during long simulations
- Use smaller datasets for prototyping
- Leverage consciousness bridge for real brain data when available

## 🔧 Integration with Main System

### Consciousness Agent Connection
All notebooks can connect to your main consciousness agent through the `consciousness_bridge.py` utility. This allows:

- **Real-time data:** Get actual brain state data from running simulations
- **Component training:** Train components and integrate back to main system
- **State synchronization:** Keep notebook experiments in sync with main brain
- **Resource sharing:** Share computational resources with main simulation

### File System Integration
Notebooks automatically integrate with your project structure:
- Load configurations from `src/config/`
- Access brain components from `src/core/`
- Read/write data to `database/` and `data/`
- Generate outputs compatible with main testing framework

### Version Control
Notebook outputs are included in `.gitignore` by default. To save important results:
1. Export key visualizations as PNG/PDF
2. Save analysis results as JSON/CSV
3. Document findings in markdown cells
4. Use notebook checkpoints for important states

## 📈 Advanced Usage

### Custom Analysis
Create your own notebooks in the `prototyping/` directory for:
- Novel experimental setups
- Custom visualizations
- Algorithm development
- Research-specific analyses

### Batch Processing
Use the consciousness bridge to run batch experiments:
```python
for config in experiment_configs:
    bridge = quick_connect(f'batch_{config["name"]}')
    results = run_experiment(bridge, config)
    save_results(results, config["name"])
    bridge.disconnect()
```

### Integration Testing
Notebooks can serve as integration test environments:
- Test new components before adding to main codebase
- Validate API changes
- Performance benchmark different approaches
- Debug complex interactions

## 🆘 Troubleshooting

### Connection Issues
If consciousness bridge fails to connect:
1. Check that main consciousness agent is running
2. Verify virtual environment is activated
3. Ensure all dependencies are installed
4. Check database connectivity

### Memory Issues
For large simulations:
1. Reduce simulation duration or time steps
2. Process data in chunks
3. Clear variables between experiments
4. Monitor system resources

### Visualization Problems
If plots don't render:
1. Ensure `%matplotlib inline` is set
2. Try restarting Jupyter kernel
3. Check matplotlib backend configuration
4. Update plotting libraries

## 📖 Further Reading

- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)

---

## 🎯 Quick Start Commands

```bash
# Start from project root
cd notebooks

# Activate environment
source ../venv/bin/activate

# Launch Jupyter Lab
jupyter lab

# Open training notebook
# -> Navigate to training/interactive_model_training.ipynb

# Connect to consciousness agent
# -> Run first few cells to establish connection

# Start experimenting!
```

**Happy brain simulation! 🧠✨**
