"""
Cursor AI Integration

Set up Cursor AI integration for compliance checking.

Author: Quark AI
Date: 2025-01-27
"""

import os
from pathlib import Path


def setup_cursor_integration(workspace_root: str = "/Users/camdouglas/quark"):
    """Set up Cursor AI integration for compliance checking"""
    workspace_path = Path(workspace_root)
    
    # Create Cursor integration script
    cursor_integration_script = workspace_path / "tools_utilities" / "cursor_compliance_integration.py"
    
    integration_code = '''#!/usr/bin/env python3
"""
Cursor AI Compliance Integration

Automatically runs compliance checks before, during, and after
Cursor AI operations.

Author: Quark AI
Date: 2025-01-27
"""

import sys
import os
from pathlib import Path

# Add tools_utilities to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "tools_utilities"))

try:
    from compliance_system.compliance_launcher import ComplianceSystemLauncher
    
    # Initialize compliance system
    compliance_system = ComplianceSystemLauncher(str(workspace_root))
    
    def before_cursor_operation(operation_name: str, target_files: list = None):
        """Run before Cursor AI operation"""
        print(f"🔍 Pre-operation compliance check for: {operation_name}")
        return compliance_system.run_operation_with_compliance(operation_name, target_files)
    
    def after_cursor_operation(operation_name: str, target_files: list = None):
        """Run after Cursor AI operation"""
        print(f"✅ Post-operation compliance verification for: {operation_name}")
        return compliance_system.check_compliance_now(target_files)
    
    # Make functions available globally
    globals()['before_cursor_operation'] = before_cursor_operation
    globals()['after_cursor_operation'] = after_cursor_operation
    
    print("✅ Cursor compliance integration loaded")
    
except ImportError as e:
    print(f"❌ Failed to load compliance integration: {e}")
'''
    
    with open(cursor_integration_script, 'w') as f:
        f.write(integration_code)
    
    # Make executable
    os.chmod(cursor_integration_script, 0o755)
    
    print(f"✅ Cursor integration script created: {cursor_integration_script}")
