# QUARK - Brain Simulation ML Framework

## 🧠 Project Overview
QUARK is a modular brain-simulation and machine-learning framework that spans molecular genetics → tissue and morphogenesis → circuits → cognition. It combines biologically inspired components (hippocampus, basal ganglia, cortex, thalamus) with modern ML tooling to study emergent cognition, learning, and control in a reproducible way.

At a glance:
- Multi-scale components: gene regulatory logic, morphogenesis utilities, microcircuits, cognitive controllers
- Learning and plasticity: working memory, episodic memory, salience and attention, executive control
- Reproducibility-first: pinned deps, deterministic testing, CI-ready structure
- Rich testing harness: focused and comprehensive benchmarks to validate capabilities and regressions

## 📁 Repository Structure

### Root Level Organization
- **`root/`** - Contains all important project files and documentation
  - `README.md` - This file (project overview)
  - `LICENSE` - Project license
  - `.cursorrules` - Cursor IDE configuration and project rules
  - `requirements.txt` - Python dependencies
  - `requirements_gdrive.txt` - Google Drive integration dependencies
  - `package.json` & `package-lock.json` - Node.js dependencies
  - `GIT_SETUP_INSTRUCTIONS.md` - Git configuration guide

### Core Architecture Directories
- **`brain_architecture/`** - Neural core and brain hierarchy systems
- **`ml_architecture/`** - Training systems and expert domains
- **`data_knowledge/`** - Research, data repository, and models
- **`testing/`** - Testing frameworks and results
- **`tools_utilities/`** - Scripts and utilities (including `aws_s3_sync.py`)
- **`integration/`** - Applications and architecture
- **`management/`** - Configurations and project management
- **`documentation/`** - Comprehensive documentation

### Configuration Files
- **`.gitignore`** - Located in `management/configurations/project/`
- **`.cursorrules`** - Located in `root/` (Cursor IDE configuration)

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   cd root
   pip install -r requirements.txt
   npm install  # if using Node.js components
   ```

2. **Setup Git:**
   ```bash
   cd root
   # Follow instructions in GIT_SETUP_INSTRUCTIONS.md
   ```

3. **Run Utilities:**
   ```bash
   cd tools_utilities/scripts
   python aws_s3_sync.py  # AWS S3 synchronization
   ```

## 🔧 Development

- **Cursor IDE**: Use `.cursorrules` in `root/` for consistent development
- **Testing**: Located in `testing/` directory
- **Documentation**: Comprehensive docs in `documentation/` directory

## 📚 Documentation

- **Architecture**: `documentation/architecture_docs/`
- **Implementation**: `documentation/implementation_docs/`
- **Setup Guides**: `documentation/setup_guides/`

## 🧪 Testing

- **Frameworks**: Located in `testing/testing_frameworks/`
- **Results**: Outputs stored in `testing/results_outputs/`

### Benchmark Capabilities
The repository provides runnable benchmarks to evaluate core cognitive functions and system health.

- Executive control: `testing/testing_frameworks/01_Executive_Control_Benchmark.py`
- Working memory: `testing/testing_frameworks/02_Working_Memory_Benchmark.py`
- Episodic memory: `testing/testing_frameworks/03_Episodic_Memory_Benchmark.py`
- Consciousness proxy metrics: `testing/testing_frameworks/04_Consciousness_Benchmark.py`
- Comprehensive integration and audit suites in `testing/testing_frameworks/tests/`

Run examples:
```bash
python testing/testing_frameworks/01_Executive_Control_Benchmark.py
python testing/testing_frameworks/02_Working_Memory_Benchmark.py
python testing/testing_frameworks/03_Episodic_Memory_Benchmark.py
```

These scripts report accuracy/latency metrics and write artifacts under `testing/testing_frameworks/*/generated_test_files/` when applicable.

## 🔒 License

See `LICENSE` file in `root/` directory.

---

*This repository follows a clean, organized structure where all important project files are contained in the `root/` directory, while maintaining the established brain simulation architecture in dedicated subdirectories.*
