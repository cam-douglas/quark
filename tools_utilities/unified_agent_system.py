#!/usr/bin/env python3
"""
Unified Agent System Entry Point

Modular wrapper for the unified agent system. All functionality is
implemented in the unified_agent_system package.

This system provides a single entry point for:
- General agent delegation (compliance, testing, documentation, cline)
- Specialized engineering role orchestration (17 roles from engineering-roles.mdc)

Author: Quark AI
Date: 2025-01-27
"""

import sys
from pathlib import Path

# Add the tools_utilities directory to the path
tools_utilities_path = Path(__file__).parent
unified_system_path = tools_utilities_path / "unified_agent_system"
sys.path.insert(0, str(tools_utilities_path))
sys.path.insert(0, str(unified_system_path))

# Import and run the main CLI interface
try:
    from unified_agent_system.cli_interface import main
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
