# Small-Mind Multi-Agent Terminal Hub

A sophisticated multi-agent system that makes your terminal behave like Cursor with intelligent routing, safety controls, and comprehensive monitoring.

## 🚀 Features

### **Isolation & Reproducibility**
- **Per-run directories** under `runs/` capture stdout/stderr, wall-clock time, and JSONL traces
- **Deterministic seeds** via `utils.seed_everything()` for reproducible results
- **Model registry** (`models.yaml`) as the source of truth with declarative configuration
- **System information capture** for full reproducibility across environments

### **Parallelism & Resource Control**
- **Lightweight tmux panes** for agent multiplexing (`agent-run`)
- **Resource limits** via `ulimit` (macOS) and `cgroups` (Linux)
- **GPU control** with `CUDA_VISIBLE_DEVICES` and `MPS_DEVICE` pinning
- **Memory and CPU limits** to prevent resource exhaustion
- **Concurrency controls** per model to avoid conflicts

### **Intelligent Routing Policy**
- **Sophisticated needs inference** using pattern matching and capability analysis
- **Load balancing** with round-robin and capacity-aware selection
- **Complexity-based routing** (low/medium/high) for appropriate model selection
- **Fallback strategies** when primary models are unavailable

### **Safety & Security**
- **Destructive capabilities opt-in** (`--allow-shell`, `--sudo-ok`)
- **Comprehensive logging** of all actions and decisions
- **Dangerous operation detection** in outputs
- **Resource usage monitoring** and limits enforcement
- **Sandboxed execution** with proper isolation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Layer     │    │  Planning Layer │    │  Routing Layer  │
│                 │    │                 │    │                 │
│ • smctl         │───▶│ • infer_needs   │───▶│ • choose_model  │
│ • agent_hub.cli │    │ • validate_needs│    │ • load_balancing│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Registry Layer │    │  Execution Layer│
                       │                 │    │                 │
                       │ • ModelRegistry │    │ • run_model     │
                       │ • models.yaml   │    │ • adapters      │
                       └─────────────────┘    └─────────────────┘
```

## 📁 File Structure

```
models/agent_hub/
├── __init__.py                 # Package initialization
├── cli.py                     # Command-line interface
├── planner.py                  # Intelligent needs inference
├── router.py                   # Model selection and routing
├── runner.py                   # Execution orchestration
├── registry.py                 # Model registry management
├── utils.py                    # Utilities and resource control
├── adapters/                   # Model-specific adapters
│   ├── adapter_open_interpreter.py
│   ├── adapter_crewai.py
│   ├── adapter_llamacpp.py
│   ├── adapter_transformers.py
│   └── adapter_smallmind.py
└── README.md                   # This file
```

## 🔧 Configuration

### Models Registry (`models.yaml`)

```yaml
# Model definitions with capabilities and constraints
smallmind:
  - id: sm.reasoner
    type: smallmind
    entry: "agents/reasoner.py:main"
    capabilities: [reasoning, planning, shell, fs, python]
    concurrency: 1
    complexity: medium
    resource_limits:
      memory_gb: 2
      cpu_limit: 4

transformers:
  - id: hf.qwen2.5
    type: transformers
    model_id: "Qwen/Qwen2.5-7B-Instruct"
    device: "mps"
    dtype: "float16"
    capabilities: [chat, code]
    complexity: low
    timeout: 300

# Routing rules with conditions
routing:
  - if: {need: planning, complexity: high} then: sm.reasoner
  - if: {need: shell} then: oi.default
  - if: {need: chat, complexity: low} then: hf.qwen2.5
  - default: llama.local.7b
```

### Environment Variables

```bash
# Resource limits
export SM_CPU_LIMIT=4
export SM_MEMORY_LIMIT_GB=8
export SM_GPU_LIMIT=0

# Safety controls
export SM_ALLOW_SHELL=false
export SM_SUDO_OK=false

# Logging
export SM_LOG_LEVEL=INFO
export SM_LOG_FILE=/path/to/logs
```

## 🚀 Usage

### Basic Commands

```bash
# List available models
smctl list

# Ask a question (auto-routed)
smctl ask "How do I install Python packages?"

# Ask with specific tools
smctl ask "Install numpy using pip" --tools shell,python

# Plan a complex task
smctl plan "Build a web application with authentication"

# Run a specific model
smctl run --model hf.qwen2.5 "Explain quantum computing"

# Parallel execution
smctl parallel "task1" "task2" "task3"

# Interactive shell
smctl shell
```

### Advanced Usage

```bash
# Allow shell access (use with caution)
smctl ask "Install and configure nginx" --allow-shell

# Allow sudo operations (use with extreme caution)
smctl ask "Update system packages" --allow-shell --sudo-ok

# Show run directory for inspection
smctl ask "Analyze this data" --show-run-dir

# Custom resource limits
export SM_RESOURCE_LIMITS='{"cpu_limit": 2, "memory_limit_gb": 4}'
smctl ask "Train a machine learning model"
```

## 🔒 Safety Features

### **Permission Controls**
- **Shell access**: Must be explicitly enabled with `--allow-shell`
- **Sudo operations**: Must be explicitly enabled with `--sudo-ok`
- **File system access**: Limited to run directories and specified paths
- **Network access**: Controlled via environment variables

### **Resource Limits**
- **CPU limits**: Configurable per model and globally
- **Memory limits**: Hard limits to prevent system exhaustion
- **GPU limits**: Device pinning and memory management
- **Timeout controls**: Prevents runaway processes

### **Content Safety**
- **Dangerous operation detection**: Scans outputs for risky commands
- **Output truncation**: Prevents memory issues from large outputs
- **Pattern matching**: Identifies potentially harmful content
- **Audit logging**: Complete trace of all operations

## 📊 Monitoring & Debugging

### **Run Information**

Each execution creates an isolated run directory with:

```
runs/ask-1234567890-abc123/
├── .running                    # Lock file (deleted when complete)
├── trace.jsonl                # Complete execution trace
├── metadata.json              # Run metadata and configuration
├── stdout/                    # Standard output files
│   └── output.txt
├── stderr/                    # Standard error files
│   └── errors.txt
├── artifacts/                 # Generated artifacts
├── checkpoints/               # Model checkpoints (if applicable)
└── model/                     # Model-specific information
    └── model_info.json
```

### **Logging**

Comprehensive logging at multiple levels:

```bash
# View real-time logs
tail -f logs/agent_hub.log

# Filter by model
grep "model_id.*hf.qwen2.5" logs/agent_hub.log

# Filter by operation
grep "event.*invoke" logs/agent_hub.log
```

### **Debugging**

```bash
# Enable debug logging
export SM_LOG_LEVEL=DEBUG

# Show detailed routing decisions
smctl ask "test" --show-run-dir

# Inspect run artifacts
ls -la $(smctl ask "test" --show-run-dir 2>&1 | grep run_dir | cut -d' ' -f2)

# Check model status
smctl describe hf.qwen2.5
```

## 🔧 Development

### **Adding New Models**

1. **Create adapter** in `adapters/` directory
2. **Add configuration** to `models.yaml`
3. **Update routing rules** as needed
4. **Test integration** with the system

### **Extending Capabilities**

1. **Update capability patterns** in `planner.py`
2. **Add routing conditions** in `router.py`
3. **Implement safety checks** in adapters
4. **Update documentation** and tests

### **Testing**

```bash
# Run unit tests
python -m pytest tests/

# Test specific adapter
python -c "from adapters.adapter_transformers import TransformersAdapter; print('OK')"

# Test routing logic
python -c "from router import Router; print('OK')"
```

## 🚨 Troubleshooting

### **Common Issues**

1. **Model not found**: Check `models.yaml` and file paths
2. **Permission denied**: Verify `--allow-shell` and `--sudo-ok` flags
3. **Resource exhaustion**: Check resource limits and system resources
4. **Timeout errors**: Adjust timeout values in configuration

### **Performance Tuning**

1. **Adjust concurrency**: Modify `concurrency` values in models
2. **Resource limits**: Set appropriate CPU/memory limits
3. **Model selection**: Use complexity-based routing for efficiency
4. **Caching**: Enable model caching where appropriate

### **Security Considerations**

1. **Review permissions**: Regularly audit allowed capabilities
2. **Monitor logs**: Check for suspicious operations
3. **Update models**: Keep models and dependencies current
4. **Network isolation**: Limit network access when possible

## 📚 API Reference

### **Core Classes**

- **`ModelRegistry`**: Manages model configurations and metadata
- **`Router`**: Intelligent model selection and load balancing
- **`RunTracker`**: Execution monitoring and cleanup
- **`BaseAdapter`**: Interface for all model adapters

### **Key Functions**

- **`infer_needs(prompt, tools)`**: Analyze prompt and determine capabilities needed
- **`choose_model(needs, routing, registry)`**: Select best model for the task
- **`run_model(model_cfg, prompt, ...)`**: Execute model with safety controls
- **`apply_resource_limits(env, limits)`**: Apply resource constraints

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Implement changes** with proper tests and documentation
4. **Submit pull request** with detailed description

### **Development Guidelines**

- **Safety first**: All changes must maintain security controls
- **Test coverage**: Include tests for new functionality
- **Documentation**: Update README and docstrings
- **Backward compatibility**: Maintain existing API contracts

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CompuCell3D** for simulation infrastructure
- **Hugging Face** for transformer models
- **CrewAI** for multi-agent workflows
- **Open Interpreter** for shell integration

---

**⚠️ Warning**: This system provides powerful capabilities. Use responsibly and always review outputs before executing commands, especially with elevated privileges.
