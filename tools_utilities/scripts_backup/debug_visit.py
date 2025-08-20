#!/usr/bin/env python3
"""
Debug script to check VisIt module status
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("🔍 Debugging VisIt Module...")

# Check if visit module exists
try:
    import visit
    print(f"✅ visit module imported: {type(visit)}")
    print(f"   visit module location: {visit.__file__ if hasattr(visit, '__file__') else 'No __file__'}")
    
    # Check available methods
    methods = [attr for attr in dir(visit) if not attr.startswith('_')]
    print(f"   Available methods: {methods}")
    
    # Check specific methods
    test_methods = ['OpenDatabase', 'SaveWindowAttributes', 'CloseComputeEngine']
    for method in test_methods:
        if hasattr(visit, method):
            print(f"   ✅ {method} found")
        else:
            print(f"   ❌ {method} NOT found")
    
except ImportError as e:
    print(f"❌ Could not import visit: {e}")

# Check the interface
try:
    from physics_simulation.visit_interface import VisItInterface, VISIT_AVAILABLE
    print(f"\n✅ VisIt interface imported: {VISIT_AVAILABLE}")
    
    # Create interface
    visit_interface = VisItInterface()
    print("✅ VisIt interface created")
    
    # Check what visit module it's using
    print(f"   Interface visit module: {type(visit_interface.visit_interface)}")
    
except Exception as e:
    print(f"❌ Interface test failed: {e}")
    import traceback
    traceback.print_exc()
