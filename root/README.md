# QUARK - Brain Simulation ML Framework

## 🧠 Project Overview
QUARK is a comprehensive brain simulation ML framework that integrates complexity evolution agents, neural architectures, and advanced cognitive processing systems.

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

## 🔒 License

See `LICENSE` file in `root/` directory.

---

*This repository follows a clean, organized structure where all important project files are contained in the `root/` directory, while maintaining the established brain simulation architecture in dedicated subdirectories.*
