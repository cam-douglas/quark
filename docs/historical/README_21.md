# SmallMind Command System

A sophisticated command discovery, organization, and execution system that provides intelligent natural language processing and secure command execution for the SmallMind project.

## 🚀 Features

### 🗂️ Hierarchical Command Organization
- **Numbered hierarchy** (1.1.1, 1.1.2, etc.) for easy navigation
- **Categories and subcategories** for logical grouping
- **Automatic discovery** of commands from codebase
- **SQLite database** for fast searching and persistence

### 🧠 Natural Language Processing
- **Intent recognition** - understands what you want to do
- **Parameter extraction** - automatically pulls out values from requests
- **Command disambiguation** - helps when multiple commands match
- **Semantic similarity** - finds related commands even with different wording

### 🔒 Secure Execution
- **Safety checks** - prevents dangerous operations
- **Resource monitoring** - tracks memory and CPU usage
- **Timeout protection** - prevents runaway processes
- **Dry run mode** - preview what will be executed
- **Confirmation prompts** - for potentially destructive operations

### 🔗 Neuro Agent Integration
- **Smart discovery** - uses neuro agents to find project-specific commands
- **Connectome analysis** - identifies important executable files
- **Automatic categorization** - organizes discovered commands intelligently

## 📦 Installation

```bash
# Basic installation (no optional NLP dependencies)
cd src/smallmind/commands
pip install -r requirements.txt

# For enhanced NLP features (optional)
pip install spacy sentence-transformers scikit-learn
python -m spacy download en_core_web_sm
```

## 🎮 Usage

### Command Line Interface

```bash
# Natural language commands
python -m smallmind.commands "train a neural network with 100 epochs"
python -m smallmind.commands "show me brain simulation tools"
python -m smallmind.commands "deploy to AWS using GPU"

# Direct command execution by number
python -m smallmind.commands 1.3.1

# Interactive mode
python -m smallmind.commands --interactive

# Help and discovery
python -m smallmind.commands --help-category "Brain Development"
python -m smallmind.commands --list-commands
python -m smallmind.commands --search "neural"
python -m smallmind.commands --stats

# Safe execution options
python -m smallmind.commands "delete old files" --dry-run
python -m smallmind.commands "install packages" --no-safe-mode
```

### Interactive Mode

```bash
🧠 SmallMind Interactive Command System
Type 'help' for assistance, 'quit' to exit
==================================================

🤖 smallmind> train a neural network
✅ Found command: advanced neural
   Description: Run neural network optimization
   Execute? [y/N]: y

🤖 smallmind> help brain
📚 Brain Development Commands:
   1.1.1 neuroscience list - List available neuroscience experts
   1.1.2 neuroscience execute - Execute neuroscience simulation tasks
   ...

🤖 smallmind> 1.3.1
✅ Executing: neuro scan
   Scanning files and building connectivity analysis...
```

### Programmatic Usage

```python
from smallmind.commands import CommandDatabase, CommandExecutor

# Initialize system
db = CommandDatabase()
executor = CommandExecutor(db)

# Execute natural language command
result = executor.execute_natural_language("show brain simulation tools")
print(f"Success: {result.success}")
print(f"Output: {result.stdout}")

# Search for commands
commands = db.search_commands("neural simulation")
for cmd in commands:
    print(f"{cmd.number} - {cmd.name}: {cmd.description}")

# Get help
help_content = executor.generate_help_content("neural")
print(help_content)
```

## 🗂️ Command Categories

The system organizes commands into a hierarchical structure:

### 1. Brain Development
- **1.1 Simulation** - Brain physics and development simulation
- **1.2 Morphogenesis** - Tissue and cellular morphogenesis  
- **1.3 Connectome** - Neural connectivity and network analysis
- **1.4 Plasticity** - Learning and synaptic plasticity

### 2. AI Models
- **2.1 Training** - Model training and fine-tuning
- **2.2 Inference** - Model inference and generation
- **2.3 Management** - Model downloading, storage, and versioning
- **2.4 Routing** - Intelligent model selection and routing

### 3. Data Processing
- **3.1 Analysis** - Data analysis and exploration
- **3.2 Transformation** - Data cleaning and transformation
- **3.3 Pipeline** - Data pipeline management
- **3.4 Validation** - Data validation and quality checks

### 4. Cloud Computing
- **4.1 AWS** - Amazon Web Services commands
- **4.2 Deployment** - Application deployment commands
- **4.3 Optimization** - Performance optimization commands
- **4.4 Monitoring** - System monitoring and logging

### 5. Development Tools
- **5.1 Testing** - Testing and validation tools
- **5.2 Debugging** - Debugging and troubleshooting tools
- **5.3 Setup** - Environment setup and configuration
- **5.4 Integration** - System integration and CI/CD

## 🧠 Natural Language Examples

The system understands various ways of expressing commands:

| Input | Intent | Matched Commands |
|-------|--------|------------------|
| "train a neural network" | execute | advanced neural, neuroscience execute |
| "show me brain tools" | query | neuroscience list, neuro scan |
| "deploy to AWS" | execute | aws deploy, aws config |
| "help with optimization" | help | Help for optimization commands |
| "what models are available?" | query | moe list, smctl list |

## 🔒 Safety Features

### Safety Checks
- **Dangerous pattern detection** - Blocks commands with destructive patterns
- **Permission validation** - Checks file system permissions
- **Resource limits** - Prevents excessive resource usage
- **Executable verification** - Ensures commands exist before execution

### Confirmation System
Commands requiring elevated privileges or potentially destructive operations will prompt for confirmation:

```
⚠️  SAFETY WARNING ⚠️

Command: aws deploy
Description: Deploy SmallMind to AWS
Warning: Command requires shell access

This command requires confirmation because:
• Requires shell access: True
• Requires sudo: False
• Safe mode: True

To proceed, add --confirm-dangerous to your command or set safe_mode=False
```

## 📊 Database Schema

The system uses SQLite with the following tables:

### Categories
- `number` - Hierarchical number (1.1, 1.2, etc.)
- `name` - Category name
- `description` - Category description  
- `parent` - Parent category (for hierarchy)

### Commands
- `id` - Unique command identifier
- `number` - Hierarchical number (1.1.1, 1.1.2, etc.)
- `name` - Command name
- `description` - Command description
- `category` - Category number
- `executable` - Executable name
- `args` - Command arguments (JSON)
- `flags` - Available flags (JSON)
- `examples` - Usage examples (JSON)
- `keywords` - Search keywords (JSON)
- `requires_shell` - Boolean flag
- `requires_sudo` - Boolean flag
- `safe_mode` - Boolean flag
- `complexity` - low/medium/high
- `source_file` - Origin file

### Command Usage (Analytics)
- `command_id` - Command identifier
- `timestamp` - Execution time
- `success` - Boolean result
- `user_input` - Original user input
- `execution_time` - Duration in seconds

## 🔗 Neuro Agent Integration

The system integrates with neuro agents for enhanced functionality:

### Smart Discovery
- **File analysis** - Uses neuro scanners to identify executable files
- **Connectome analysis** - Identifies high-centrality files as potential commands
- **Pattern recognition** - Discovers common command patterns

### Dynamic Commands
- **Project-specific** - Discovers commands unique to your project
- **Context-aware** - Adapts to your working environment
- **Learning** - Improves suggestions based on usage

## 🧪 Testing and Demo

Run the comprehensive demo to see all features:

```bash
cd src/smallmind/commands
python demo.py
```

This will demonstrate:
- Database initialization and statistics
- Natural language parsing examples
- Command execution with safety checks
- Neuro agent integration
- Interactive features

## 🔧 Development

### Adding New Commands

Commands are automatically discovered from the codebase, but you can also add them manually:

```python
from smallmind.commands import CommandDatabase, Command

db = CommandDatabase()
new_command = Command(
    id="my_custom_command",
    number="6.1.1",
    name="my command",
    description="Does something useful",
    category="6.1",
    subcategory="Custom",
    executable="python",
    args=["my_script.py"],
    flags={"--verbose": "Enable verbose output"},
    examples=["my command --verbose"],
    keywords=["custom", "utility"],
    complexity="low"
)
db.store_command(new_command)
```

### Extending NLP

To improve natural language understanding, you can:

1. **Add intent patterns** in `natural_language_parser.py`
2. **Extend entity extraction** for domain-specific terms
3. **Improve parameter extraction** with new regex patterns
4. **Add semantic similarity** with custom embeddings

### Custom Execution Context

```python
from smallmind.commands import ExecutionContext

context = ExecutionContext(
    working_directory="/custom/path",
    environment_vars={"CUSTOM_VAR": "value"},
    timeout=600.0,
    dry_run=False,
    safe_mode=True,
    interactive=True,
    resource_limits={
        "max_memory_mb": 8192,
        "max_cpu_percent": 90,
        "max_execution_time": 1200
    }
)
```

## 📈 Performance

The system is designed for performance:

- **SQLite database** - Fast querying and indexing
- **Cached embeddings** - Semantic similarity caching
- **Lazy loading** - NLP models loaded on demand
- **Resource monitoring** - Prevents system overload
- **Parallel discovery** - Multi-threaded command discovery

## 🔍 Troubleshooting

### Common Issues

**Command not found**
```bash
# Refresh the database
python -c "from smallmind.commands import CommandDatabase; db = CommandDatabase(); db.load_commands()"
```

**NLP features not working**
```bash
# Install optional dependencies
pip install spacy sentence-transformers scikit-learn
python -m spacy download en_core_web_sm
```

**Neuro integration failing**
```bash
# Check neuro agent availability
python -c "from smallmind.commands.neuro_integration import NeuroAgentConnector; print(NeuroAgentConnector().get_status())"
```

**Permission errors**
```bash
# Check database permissions
ls -la ~/.smallmind/commands.db

# Reset database
rm ~/.smallmind/commands.db
python -m smallmind.commands --stats  # This will recreate it
```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. **Add new command sources** in `command_database.py`
2. **Improve NLP patterns** in `natural_language_parser.py`  
3. **Enhance safety checks** in `command_executor.py`
4. **Extend neuro integration** in `neuro_integration.py`
5. **Add new CLI features** in `cli.py`

## 📜 License

This project is part of the SmallMind computational neuroscience framework.

---

**Happy commanding! 🧠✨**
