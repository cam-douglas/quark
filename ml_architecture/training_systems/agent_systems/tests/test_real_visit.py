#!/usr/bin/env python3
"""
Test script to directly import the real VisIt module
"""

import sys
import os

print("🔍 Testing Real VisIt Import...")

# Add VisIt Python path
visit_python_path = "/Applications/VisIt.app/Contents/Resources/3.4.2/darwin-arm64/lib/site-packages"
if visit_python_path not in sys.path:
    sys.path.insert(0, visit_python_path)

print(f"✅ Added VisIt path: {visit_python_path}")
print(f"   Python path now includes: {visit_python_path in sys.path}")

# Check if visit is already in sys.modules
if 'visit' in sys.modules:
    print(f"⚠️  visit module already in sys.modules: {type(sys.modules['visit'])}")
    # Remove it to force fresh import
    del sys.modules['visit']
    print("✅ Removed existing visit module")

# Try to import visit
try:
    import visit
    print(f"✅ visit imported successfully: {type(visit)}")
    print(f"   Module name: {visit.__name__ if hasattr(visit, '__name__') else 'No __name__'}")
    print(f"   Module file: {visit.__file__ if hasattr(visit, '__file__') else 'No __file__'}")
    
    # Check available methods
    methods = [attr for attr in dir(visit) if not attr.startswith('_')]
    print(f"   Available methods: {methods[:10]}...")  # Show first 10
    
    # Check if it's the real VisIt
    if hasattr(visit, 'OpenDatabase'):
        print("✅ Real VisIt detected - OpenDatabase method found")
    else:
        print("❌ Real VisIt not detected - OpenDatabase method missing")
        
except ImportError as e:
    print(f"❌ visit import failed: {e}")
    import traceback
    traceback.print_exc()
