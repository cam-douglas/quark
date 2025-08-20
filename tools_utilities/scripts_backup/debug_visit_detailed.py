#!/usr/bin/env python3
"""
Detailed debug script to understand what visit module is being imported
"""

import sys
import os

print("🔍 Detailed VisIt Module Debug...")

# Check all modules that contain 'visit'
print("📋 All modules containing 'visit':")
for module_name in sorted(sys.modules.keys()):
    if 'visit' in module_name.lower():
        module = sys.modules[module_name]
        print(f"   {module_name}: {type(module)}")

# Check Python path
print(f"\n📁 Python path:")
for i, path in enumerate(sys.path):
    print(f"   {i}: {path}")

# Try to import visit and see what happens
print(f"\n🔍 Trying to import visit...")
try:
    import visit
    print(f"✅ visit imported: {type(visit)}")
    print(f"   Module name: {visit.__name__ if hasattr(visit, '__name__') else 'No __name__'}")
    print(f"   Module file: {visit.__file__ if hasattr(visit, '__file__') else 'No __file__'}")
    print(f"   Module dict: {list(visit.__dict__.keys()) if hasattr(visit, '__dict__') else 'No __dict__'}")
    
    # Check if it's our mock
    if hasattr(visit, 'LaunchNowin'):
        print("   ✅ This appears to be our mock module")
    else:
        print("   ❌ This is NOT our mock module")
        
except ImportError as e:
    print(f"❌ visit import failed: {e}")

# Check if our mock was created
print(f"\n🔍 Checking if our mock was created...")
if 'visit' in sys.modules:
    visit_module = sys.modules['visit']
    print(f"✅ visit module in sys.modules: {type(visit_module)}")
    if hasattr(visit_module, 'LaunchNowin'):
        print("   ✅ Our mock module is in sys.modules")
    else:
        print("   ❌ Different visit module in sys.modules")
else:
    print("❌ No visit module in sys.modules")

# Try to force our mock
print(f"\n🔍 Trying to force our mock...")
try:
    # Remove any existing visit module
    if 'visit' in sys.modules:
        del sys.modules['visit']
    
    # Create our mock
    class MockVisit:
        def LaunchNowin(self): return True
        def OpenDatabase(self, filename): return True
        def SaveWindowAttributes(self): 
            class Attrs: 
                def __init__(self): 
                    self.family = 0; self.format = "PNG"; self.width = 1024; self.height = 768; self.fileName = "output.png"
            return Attrs()
        def SetSaveWindowAttributes(self, attrs): return True
        def SaveWindow(self): return True
        def CloseComputeEngine(self): return True
    
    our_mock = MockVisit()
    sys.modules['visit'] = our_mock
    
    # Now try to import
    import visit
    print(f"✅ Forced import: {type(visit)}")
    if hasattr(visit, 'LaunchNowin'):
        print("   ✅ Our mock is now working!")
    else:
        print("   ❌ Still not working")
        
except Exception as e:
    print(f"❌ Force mock failed: {e}")
    import traceback
    traceback.print_exc()
