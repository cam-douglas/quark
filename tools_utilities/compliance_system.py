#!/usr/bin/env python3
"""
Quark Compliance System Entry Point

This is the main entry point for the Quark compliance system.
All functionality is implemented in the compliance_system package.
"""

import sys
import os
from pathlib import Path

# Add the compliance_system package to the path
compliance_system_path = Path(__file__).parent / "compliance_system"
sys.path.insert(0, str(compliance_system_path))

# Import and run the main CLI interface
try:
    from cli_interface import main
except ImportError:
    # Fallback: run the CLI directly
    import subprocess
    cli_path = compliance_system_path / "cli_interface.py"
    sys.argv[0] = str(cli_path)
    subprocess.run([sys.executable, str(cli_path)] + sys.argv[1:])
    sys.exit(0)

if __name__ == "__main__":
    main()