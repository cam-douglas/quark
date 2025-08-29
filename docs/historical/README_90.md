# Cursor Integration System

## Overview

This directory contains a comprehensive integration system for Cursor IDE that enables:

- **Programmatic Settings Management**: Automated configuration of Cursor settings, keybindings, and preferences
- **CLI Terminal Integration**: Complete setup of Cursor CLI for Mac terminal usage
- **Documentation Integration**: Local integration of Cursor documentation into project rules
- **Automated Testing**: Comprehensive test suite with simulation capabilities
- **Brain Simulation Optimization**: Specialized configuration for neural simulation projects

## 🚀 Quick Start

### 1. Run Complete Setup
```bash
cd /Users/camdouglas/quark/.cursor/rules
python setup_cursor_integration.py
```

### 2. Verify Installation
```bash
# Check CLI availability
cursor-agent --version

# Test opening current directory
cursor .

# Validate settings
python cursor_settings_manager.py validate
```

### 3. Run Tests
```bash
cd /Users/camdouglas/quark
python tests/cursor_integration_tests.py
```

## 📁 File Structure

```
.cursor/rules/
├── README.md                          # This file
├── cursor_documentation_integration.md # Cursor docs integration
├── cursor_settings_manager.py          # Settings management system
├── cursor_cli_setup.py                # CLI installation & setup
├── setup_cursor_integration.py        # Main orchestrator script
└── integration_setup_report.json      # Generated setup report

tests/
└── cursor_integration_tests.py        # Comprehensive test suite
```

## 🔧 Components

### 1. Documentation Integration (`cursor_documentation_integration.md`)
- **Purpose**: Local integration of Cursor documentation
- **Features**: 
  - Complete reference to all Cursor features
  - Direct links to official documentation
  - Integration status tracking
- **Usage**: Reference file for Cursor capabilities and configuration options

### 2. Settings Manager (`cursor_settings_manager.py`)
- **Purpose**: Programmatic management of Cursor IDE settings
- **Features**:
  - Automated settings configuration
  - Backup and restore functionality
  - AI model selection
  - Rules system configuration
  - Keybindings management
- **Usage**:
  ```bash
  # Initialize default settings
  python cursor_settings_manager.py init
  
  # Set AI model
  python cursor_settings_manager.py set-model claude-3.5-sonnet
  
  # Backup current settings
  python cursor_settings_manager.py backup
  
  # Validate configuration
  python cursor_settings_manager.py validate
  ```

### 3. CLI Setup (`cursor_cli_setup.py`)
- **Purpose**: Automated Cursor CLI installation for Mac terminal
- **Features**:
  - CLI installation and verification
  - PATH configuration (bash/zsh)
  - Shell command integration
  - Comprehensive testing
- **Usage**:
  ```bash
  # Run complete CLI setup
  python cursor_cli_setup.py
  
  # Create test script only
  python cursor_cli_setup.py --test
  ```

### 4. Integration Orchestrator (`setup_cursor_integration.py`)
- **Purpose**: Main coordinator for complete Cursor integration
- **Features**:
  - Step-by-step setup process
  - Validation and verification
  - Comprehensive reporting
  - Error handling and recovery
- **Usage**:
  ```bash
  # Run complete integration setup
  python setup_cursor_integration.py
  ```

### 5. Test Suite (`../tests/cursor_integration_tests.py`)
- **Purpose**: Comprehensive testing of all integration components
- **Features**:
  - Unit tests for all components
  - Integration testing
  - Simulation-based testing
  - Detailed reporting
- **Usage**:
  ```bash
  # Run all tests
  python tests/cursor_integration_tests.py
  
  # Run specific test class
  python -m unittest tests.cursor_integration_tests.TestCursorSettingsManager
  ```

## ⚙️ Configuration

### Default Settings Applied

The integration system applies the following optimized settings for brain simulation projects:

```json
{
  "cursor.aiModel": "claude-3.5-sonnet",
  "cursor.rules": {
    "enabled": true,
    "autoLoad": true,
    "directory": ".cursor/rules"
  },
  "cursor.memories": {
    "enabled": true,
    "autoSave": true
  },
  "cursor.agent": {
    "backgroundEnabled": true,
    "webMobileEnabled": true
  },
  "cursor.codebase": {
    "indexingEnabled": true,
    "ignorePatterns": [
      "venv/**",
      "__pycache__/**",
      "cache/**",
      "logs/**",
      "backups/**"
    ]
  }
}
```

### Keybindings

Default keybindings optimized for development workflow:

- `Cmd+K`: Ask AI
- `Cmd+Shift+K`: Start Agent
- `Cmd+I`: Inline Edit

## 🧪 Testing

### Test Categories

1. **Settings Management Tests**
   - Configuration file handling
   - Backup and restore functionality
   - Settings validation
   - Deep merge operations

2. **CLI Setup Tests**
   - Installation verification
   - PATH configuration
   - Shell integration
   - Command availability

3. **Documentation Integration Tests**
   - File existence and structure
   - Content validation
   - Link verification

4. **Simulation Tests**
   - Mock command execution
   - File operation simulation
   - Error condition testing

### Running Tests

```bash
# Run all tests with verbose output
python tests/cursor_integration_tests.py

# Run specific test category
python -m unittest tests.cursor_integration_tests.TestCursorSettingsManager -v

# Generate test report
python -c "
from tests.cursor_integration_tests import generate_test_report
import json
print(json.dumps(generate_test_report(), indent=2))
"
```

## 📊 Monitoring & Validation

### Setup Validation

The system provides comprehensive validation:

```bash
# Validate complete setup
python setup_cursor_integration.py

# Check specific components
python cursor_settings_manager.py validate
python cursor_cli_setup.py --verify
```

### Generated Reports

- **Setup Report**: `integration_setup_report.json`
- **CLI Report**: `cli_setup_report.json`
- **Settings Backup**: `.cursor/backups/settings_backup_*.json`

## 🔧 Troubleshooting

### Common Issues

1. **"cursor-agent not found"**
   ```bash
   # Check PATH
   echo $PATH
   
   # Add to shell profile
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

2. **Settings not loading**
   ```bash
   # Check settings file
   cat .cursor/settings.json
   
   # Validate JSON syntax
   python -m json.tool .cursor/settings.json
   ```

3. **Permission errors**
   ```bash
   # Fix CLI permissions
   chmod +x ~/.local/bin/cursor-agent
   chmod +x ~/.local/bin/cursor
   ```

4. **Import errors in tests**
   ```bash
   # Check Python path
   python -c "import sys; print('\\n'.join(sys.path))"
   
   # Run from project root
   cd /Users/camdouglas/quark
   python tests/cursor_integration_tests.py
   ```

### Debug Mode

Run components in debug mode for detailed output:

```bash
# Settings manager debug
python cursor_settings_manager.py validate 2>&1 | tee debug_settings.log

# CLI setup debug  
python cursor_cli_setup.py 2>&1 | tee debug_cli.log
```

## 🚀 Advanced Usage

### Custom Settings Configuration

```python
from cursor_settings_manager import CursorSettingsManager

manager = CursorSettingsManager()

# Custom configuration
custom_settings = {
    "cursor.aiModel": "gpt-4",
    "cursor.customFeature": True
}

manager.update_settings(custom_settings)
```

### Programmatic CLI Management

```python
from cursor_cli_setup import CursorCLISetup

setup = CursorCLISetup()
report = setup.generate_setup_report()
print(f"CLI Status: {report['installation_status']}")
```

### Integration with Brain Simulation

The system is optimized for brain simulation projects with:

- Specialized ignore patterns for simulation data
- Memory management for large datasets
- AI model configuration for neuroscience tasks
- Integration with multi-agent systems

## 📚 Documentation References

- **Cursor Official Docs**: [https://docs.cursor.com](https://docs.cursor.com)
- **CLI Documentation**: [https://docs.cursor.com/en/cli](https://docs.cursor.com/en/cli)
- **Rules System**: [https://docs.cursor.com/en/rules](https://docs.cursor.com/en/rules)
- **Configuration**: [https://docs.cursor.com/en/configuration](https://docs.cursor.com/en/configuration)

## 🤝 Contributing

When modifying the integration system:

1. **Update tests** for any new functionality
2. **Run validation** before committing changes
3. **Update documentation** to reflect changes
4. **Test in clean environment** to ensure reproducibility

## 📄 License

This integration system is part of the Quark brain simulation project and follows the same licensing terms.

---

**Last Updated**: Generated automatically by Cursor Integration System
**Version**: 1.0.0
**Compatibility**: Cursor IDE v1.0+, macOS 10.15+
