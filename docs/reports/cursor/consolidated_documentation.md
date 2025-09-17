# Cursor Configuration - Consolidated Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Consolidation Summary](#consolidation-summary)
3. [Terminal Rules System](#terminal-rules-system)
4. [Multi-Agent Framework](#multi-agent-framework)
5. [Resource Optimization](#resource-optimization)
6. [Environment Management](#environment-management)
7. [Usage Guide](#usage-guide)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

### What This System Provides
- **Unified Configuration**: Single, clean configuration files instead of scattered settings
- **Multi-Agent Framework**: Integrated OmniNode (blockchain) and Digital Brain Project (neuroscience) systems
- **Environment Management**: Automatic Python virtual environment handling
- **Resource Optimization**: Maximum CPU/GPU allocation for Cursor IDE
- **Terminal Integration**: Enhanced shell experience with agent controls

### Key Files Created
- `cursor_config.zsh` - Consolidated shell configuration
- `ml_agent_config.yaml` - Multi-agent system configuration
- `consolidated_documentation.md` - This comprehensive guide

---

## Consolidation Summary

### What Was Consolidated

#### Before: Scattered Configuration
- Multiple bootstrap files with overlapping functionality
- Redundant prompt configurations and virtual environment management
- Hardcoded paths scattered across multiple subdirectories
- Overlapping ML and agentic system definitions
- Multiple summary and documentation files

#### After: Clean, Unified System
- **Single shell configuration** (`cursor_config.zsh`) with all essential functionality
- **Unified multi-agent framework** (`ml_agent_config.yaml`) combining blockchain and neuroscience systems
- **Portable paths** using `$HOME` instead of hardcoded user paths
- **Comprehensive documentation** in one place

### Benefits of Consolidation
1. **✅ No More Confusion**: One configuration to rule them all
2. **✅ Consistent Dependencies**: All packages are compatible
3. **✅ Easy Maintenance**: Update one file, not many
4. **✅ Portable**: Works across different user environments
5. **✅ Resource Efficient**: No duplicate functionality
6. **✅ Professional Setup**: Clean, research-grade configuration

---

## Terminal Rules System

### Overview
The Terminal Rules system provides automated environment management, agent controls, and Cursor IDE optimization. It automatically manages Python virtual environments, provides unified agent control commands, and enhances the terminal experience within Cursor.

### Quick Start

#### Enable Terminal Rules
```bash
# Navigate to project root
cd /Users/camdouglas/small-mind

# Source the consolidated configuration
source cursor/cursor_config.zsh

# Or enable persistent rules
touch ~/.terminal-rules-enabled
```

#### Verify Installation
```bash
# Check if environment is active
echo $VIRTUAL_ENV

# Test helper commands
sm-env-complete
```

### Environment Configuration

#### Project Paths
The system automatically configures these environment variables:
```bash
TR_PROJECT_ROOT="/Users/camdouglas/small-mind"
TR_VENV_PATH="$TR_PROJECT_ROOT/env"
TR_PROJECT_PYTHON="$TR_VENV_PATH/bin/python"
TR_SYSTEM_PYTHON="/opt/homebrew/bin/python3"
```

#### Virtual Environment Management
- **Auto-activation**: Automatically activates virtual environment when in project directory
- **Prompt customization**: Shows active venv name in **cyan** with **full directory path** in **yellow**
- **Path normalization**: Prioritizes project binaries and Python paths
- **Full path display**: Always shows complete directory path for unambiguous navigation

#### Cursor Integration
- **Working directory**: Automatically navigates to project root in Cursor terminals
- **Environment detection**: Detects Cursor/VS Code terminals via `TERM_PROGRAM`
- **Auto-scoping**: Enables environment rules when navigating into project tree
- **Universal compatibility**: Works in Cursor, VS Code, native Terminal, iTerm2, and all macOS terminals

### Agent Control System

#### `agentctl` Command
Universal agent backend selector with multiple modes:

```bash
# Auto-selection (prefers open-interpreter → crewai → none)
agentctl auto

# Specific agent selection
agentctl open-interpreter    # or 'oi'
agentctl crewai             # or 'crew'
agentctl none               # disable agents
```

#### `agent-run` Command
Parallel multi-agent execution using tmux:

```bash
# Run multiple agents in parallel panes
agent-run \
  "oi --eval 'ls -la && python -V'" \
  "crew run ./agents/planner.yaml"
```

#### `smctl` Integration
Natural language agent control with Cursor parity:

```bash
# Ask questions with tool access
smctl ask "Set up a new venv and install torch; verify GPU; save a report" \
  --tools shell,python --allow-shell

# Plan multi-step workflows
smctl plan "Benchmark 7B vs 13B local models on MPS with 1K token prompts"

# Run specific models
smctl run --model sm.reasoner -- "Reason about tradeoffs for RAG chunking."
smctl run --model hf.qwen2.5 -- "Summarize this repo: https://…"

# Parallel execution
smctl parallel "smctl run --model oi.default -- 'audit my pip env'" \
               "smctl run --model sm.rag -- 'index /docs then answer: …'"
```

### Helper Commands
```bash
sm-env              # Basic environment info
sm-python-info      # Python version and path info
sm-deps             # All installed packages
sm-env-complete     # Complete environment context
sm-path             # Current working directory
sm-env-path         # Virtual environment path
```

---

## Multi-Agent Framework

### System Overview
The integrated multi-agent framework combines:
- **OmniNode**: Multi-agent blockchain development system
- **Digital Brain Project**: Computational neuroscience simulation system
- **Automatic project detection** and role selection

### Project Detection
The system automatically detects your project type based on your prompts:

#### Blockchain/Web3 Projects
**Indicators**: blockchain, web3, smart contract, ethereum, solidity, defi, nft, cryptocurrency, decentralized, dapp

**Activates**: Full OmniNode system with roles:
- Front End Expert (UI/UX)
- Back End Expert (Server-side logic)
- Blockchain Expert (Smart contracts)
- ML Expert (AI integration)
- Security Expert (Security across all components)
- Network Expert (P2P networking)
- Designer Expert (Visual/audio design)
- Supervisor Expert (Process management)
- OmniTect Expert (Lead developer/architect)

#### Neuroscience/Brain Simulation
**Indicators**: brain, neural, neuroscience, simulation, morphogenesis, neurulation, neurogenesis, circuit, synapse, cognitive, developmental, molecular, genetic, tissue, compucell3d, sbml, grn, morphogen

**Activates**: Digital Brain Project roles:
- Program Orchestrator (Lead Systems Architect)
- Developmental Neurobiologist (Neurulation → Neurogenesis)
- Molecular Geneticist (Gene Regulatory Networks)
- Computational Biologist (GRN → Phenotype)
- Tissue & Morphogenesis Engineer (Neural Tube & Vesicles)
- Connectomics Engineer (Wiring Plan)
- Circuit Builder (Microcircuit & Columns)
- Neuroplasticity & Learning Scientist
- Self-Organization Engineer (Proto-Cortex)
- Multimodal Sensory Engineer
- Cognitive Architect (High-level Functions)
- Neuromorphic/Systems Engineer (Runtime & Accel)
- Data & Pipelines Engineer
- Evaluation & Behavior Scientist
- Safety & Ethics Officer
- Product & HCI (Observability + Interfaces)
- QA & Reproducibility Engineer

#### ML/AI Projects
**Indicators**: machine learning, ml, ai, neural network, deep learning, training, model, prediction, classification, regression, tensorflow, pytorch, scikit-learn, jupyter, data science

**Activates**: ML-focused roles from both systems as appropriate

### Safety and Execution Policies

#### Safety Rules
The system includes comprehensive safety policies that require user approval for:
- Destructive file operations (`rm -rf`)
- System permission changes (`chown`, `chmod` with `-R`)
- Disk operations (`diskutil erase/repartition`)
- Unsafe curl commands (piping to shell)
- Insecure SSH connections
- System authentication changes

#### Repair Hints
Automatic suggestions for common issues:
- Missing Python modules → Install requirements
- Missing package managers → Install poetry/pipenv
- Python not found → Install Homebrew python or pyenv

---

## Resource Optimization

### Overview
The resource optimization system provides maximum CPU and GPU allocation for Cursor on macOS, ensuring optimal performance for AI operations, code analysis, and development workflows.

### What the Optimization Does

#### CPU Optimization
- ✅ Uses all available CPU cores and threads
- ✅ Disables CPU throttling and power management
- ✅ Sets real-time priority scheduling
- ✅ Maximizes process limits

#### GPU Optimization
- ✅ Allocates maximum GPU memory
- ✅ Disables GPU power saving modes
- ✅ Enables all GPU compute units
- ✅ Sets highest GPU process priority

#### Memory Optimization
- ✅ Prioritizes RAM allocation for Cursor
- ✅ Minimizes swap usage
- ✅ Pins critical processes in memory
- ✅ Optimizes file I/O operations

#### Process Management
- ✅ Sets Cursor to highest process priority
- ✅ Optimizes I/O and network operations
- ✅ Manages background process priorities
- ✅ Monitors resource usage continuously

### Quick Start

#### Option 1: Automated Optimization (Recommended)
```bash
# Make the script executable
chmod +x .cursor/rules/apply-resource-optimization.sh

# Run the optimization script
sudo .cursor/rules/apply-resource-optimization.sh
```

#### Option 2: Install as Startup Daemon (Permanent)
```bash
# Install the startup daemon (runs automatically at boot with root access)
sudo .cursor/rules/install-startup-daemon.sh

# The optimization will now run automatically:
# - At system startup
# - Every 5 minutes
# - With full root privileges
```

### Monitoring and Maintenance

#### Real-time Monitoring
```bash
# Use the provided monitoring script
~/CursorConfig/cursor_monitor.sh

# Or monitor manually
top -pid $(pgrep -f "Cursor")
```

#### Performance Indicators
- **CPU Usage**: Should utilize all available cores
- **Memory Usage**: High RAM allocation for Cursor
- **Response Time**: Faster AI suggestions and code analysis
- **Thermal**: Monitor system temperature

### Safety Considerations
⚠️ **Important Warnings**:
- These optimizations may reduce battery life on laptops
- Monitor system temperature during heavy usage
- Some settings may affect other applications
- Settings may reset after system reboot

---

## Environment Management

### Virtual Environment Consolidation

#### Before: Multiple Scattered Environments
Your system had **multiple virtual environments** scattered across different locations:
- `/Users/camdouglas/small-mind/.venv` (old project environment)
- `/Users/camdouglas/.smallmind/venv` (old smallmind environment)
- `/Users/camdouglas/.venv` (global user environment)
- `/Users/camdouglas/DRIVE/HITSYNCLUB/DEV/omninode/venv` (old omninode environment)
- `/Users/camdouglas/DRIVE/HITSYNCLUB/DEV/max4node/max_device/venv` (old max4node environment)
- Various other project-specific environments

#### After: Single, Clean Environment
Now you have **one consolidated environment**:
- **Location**: `/Users/camdouglas/small-mind/env`
- **Python Version**: Python 3.13.6
- **Pip Version**: pip 25.2
- **All Essential Dependencies**: torch, numpy, pandas, transformers, openai, anthropic, etc.

### Current Clean Setup

#### Single Project Environment
```bash
/Users/camdouglas/small-mind/env/
├── bin/
│   ├── python          # Python 3.13.6
│   ├── pip             # pip 25.2
│   └── activate        # Activation script
├── lib/
│   └── python3.13/
│       └── site-packages/
│           ├── torch/          # PyTorch 2.8.0
│           ├── numpy/          # NumPy 2.3.2
│           ├── pandas/         # Pandas 2.3.1
│           ├── transformers/   # Transformers 4.55.2
│           ├── openai/         # OpenAI 1.99.9
│           ├── anthropic/      # Anthropic 0.64.0
│           └── [other packages]
└── pyvenv.cfg
```

### Environment Requirements

#### Mandatory Information
When working in the project, **ALL AGENTS MUST** provide complete, unambiguous information:
- **Current working directory** (absolute path)
- **Active virtual environment** (absolute path)
- **Python version and environment**
- **Key dependencies and versions**
- **File paths** (absolute)
- **Directory references** (absolute)

#### Example Format
```bash
# ✅ CORRECT
Current working directory: /Users/camdouglas/small-mind/src/smallmind/models
Active virtual environment: /Users/camdouglas/small-mind/env
Python version: Python 3.11.5
Python path: /Users/camdouglas/small-mind/env/bin/python

# ❌ INCORRECT
Current working directory: models
Active virtual environment: env
Python: 3.11
```

---

## Usage Guide

### Basic Environment Management

#### Automatic Activation (Recommended)
```bash
# Navigate to any project directory
cd /Users/camdouglas/small-mind/src/smallmind
# Environment automatically activates!

# Check status
sm-env-complete
```

#### Manual Activation (if needed)
```bash
# Activate manually
source /Users/camdouglas/small-mind/env/bin/activate

# Deactivate
deactivate
```

### Agent Management

#### Select Agent Backend
```bash
# Select agent backend
agentctl auto                    # Auto-select best available
agentctl open-interpreter        # Use open-interpreter
agentctl crewai                  # Use CrewAI
agentctl none                    # Disable agents
```

#### Run Parallel Agents
```bash
# Run parallel agents
agent-run \
  "agentctl oi --task 'analyze code'" \
  "agentctl crew --task 'design API'"
```

### Natural Language Control
```bash
# Ask questions with tool access
smctl ask "Set up a new venv and install torch" \
  --tools shell,python --allow-shell

# Plan workflows
smctl plan "Benchmark local models on MPS"

# Run specific models
smctl run --model sm.reasoner -- "Analyze this codebase"
```

### Maintenance

#### Updating Dependencies
```bash
# Activate environment (auto-activates in project dirs)
cd /Users/camdouglas/small-mind

# Update all packages
pip install --upgrade pip
pip install --upgrade torch numpy pandas transformers openai anthropic

# Or update specific packages
pip install --upgrade torch
```

#### Adding New Packages
```bash
# Install new packages
pip install package_name

# Install with specific version
pip install package_name==1.2.3
```

#### Checking Environment Health
```bash
# Complete environment status
sm-env-complete

# Check for outdated packages
pip list --outdated

# Verify key dependencies
pip show torch numpy transformers
```

---

## Troubleshooting

### Common Issues

#### Virtual Environment Not Found
```bash
# Check if env directory exists
ls -la /Users/camdouglas/small-mind/env

# Create if missing
/opt/homebrew/bin/python3 -m venv env
source env/bin/activate
pip install -U pip
```

#### Environment Doesn't Auto-Activate
```bash
# Check terminal rules status
ls -la ~/.terminal-rules-enabled

# Re-enable if needed
touch ~/.terminal-rules-enabled

# Source rules
source ~/small-mind/cursor/cursor_config.zsh
```

#### Agent Not Installed
```bash
# Install open-interpreter
pip install open-interpreter

# Install crewai
pip install crewai
```

#### tmux Not Available
```bash
# Install tmux
brew install tmux
```

#### Packages Are Missing
```bash
# Reinstall essential packages
pip install torch numpy pandas transformers openai anthropic

# Or restore from requirements
pip install -r requirements.txt  # if you have one
```

### Debug Commands
```bash
# Check environment status
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "PYTHONPATH: $PYTHONPATH"
echo "PROJECT_ROOT: $TR_PROJECT_ROOT"

# Test agent availability
agentctl auto --help

# Check terminal rules status
ls -la ~/.terminal-rules-enabled

# Complete environment context
sm-env-complete
```

### Performance Issues
If performance degrades:
1. Restart Cursor
2. Check system resources with Activity Monitor
3. Run the optimization script again
4. Monitor for resource conflicts

### Getting Help
- Check system logs for errors
- Use Activity Monitor for resource analysis
- Review the optimization script output
- Consult macOS system documentation

---

## Advanced Configuration

### Customizing Project Paths
Edit variables in `cursor_config.zsh`:
```bash
export TR_PROJECT_ROOT="/path/to/your/project"
export TR_VENV_PATH="$TR_PROJECT_ROOT/your_venv_name"
```

### Adding New Agent Types
Extend the `agentctl` function in `cursor_config.zsh`:
```bash
    your_agent|ya)
      if command -v your_agent >/dev/null 2>&1; then
        your_agent "$@"
      else
        echo "[agentctl] your_agent not found. Install with: pip install your_agent"
      fi
      ;;
```

### Persistent Configuration
Add to your `~/.zshrc`:
```bash
# Auto-source terminal rules
if [[ -f "$HOME/.terminal-rules-enabled" ]]; then
  source ~/small-mind/cursor/cursor_config.zsh
fi
```

### Custom Resource Limits
```bash
# Set custom CPU limits
sudo launchctl limit maxproc 100000

# Set custom file limits
sudo launchctl limit maxfiles 100000

# Custom process priority
sudo renice -n -15 -p $(pgrep -f "Cursor")
```

---

## Integration with OmniNode

The Terminal Rules system integrates seamlessly with the OmniNode multi-agent framework:

1. **Environment Consistency**: Provides consistent Python environments across all agent roles
2. **Agent Routing**: Unified interface for selecting and running different AI agents
3. **Development Workflow**: Streamlines the development process with automated environment management
4. **Cursor Optimization**: Enhanced terminal experience within the Cursor IDE

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify the terminal rules are enabled: `ls -la ~/.terminal-rules-enabled`
3. Check syntax: `zsh -n cursor/cursor_config.zsh`
4. Test sourcing: `source cursor/cursor_config.zsh`

---

## Summary

🎉 **Your system now has a unified, clean, research-grade configuration!**

- **One configuration** instead of many scattered files
- **Integrated multi-agent framework** for blockchain and neuroscience projects
- **Automatic environment management** with helper commands
- **Resource optimization** for maximum Cursor performance
- **Comprehensive documentation** in one place
- **Professional setup** that matches the sophistication of your research

This consolidation eliminates confusion, prevents conflicts, and provides a professional development environment that scales with your projects! 🚀
