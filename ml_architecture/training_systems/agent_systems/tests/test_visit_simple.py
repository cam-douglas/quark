#!/usr/bin/env python3
"""
Simple test for VisIt mock module
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("🔍 Testing Simple VisIt Mock...")

# Try to import the interface
try:
    from physics_simulation.visit_interface import VisItInterface, VISIT_AVAILABLE
    print(f"✅ VisIt interface imported: {VISIT_AVAILABLE}")
    
    # Check if visit module exists
    try:
        import visit
        print(f"✅ visit module imported: {type(visit)}")
        
        # Check specific methods
        methods_to_check = ['OpenDatabase', 'SaveWindowAttributes', 'CloseComputeEngine']
        for method in methods_to_check:
            if hasattr(visit, method):
                print(f"   ✅ {method} found")
            else:
                print(f"   ❌ {method} NOT found")
        
        # Try to create interface
        visit_interface = VisItInterface()
        print("✅ VisIt interface created successfully")
        
        # Test basic functionality
        test_data = {
            "regions": {"test": {"position": [0, 0, 0], "size": 100}},
            "neurons": [{"position": [0.1, 0, 0], "type": 0, "activity": 0.5}]
        }
        
        # Test visualization
        success = visit_interface.create_brain_visualization(test_data, "3D")
        print(f"✅ Visualization creation: {success}")
        
        # Test export
        export_success = visit_interface.export_visualization("test.png")
        print(f"✅ Export: {export_success}")
        
        # Test analysis
        analysis = visit_interface.analyze_brain_data(test_data, "statistics")
        print(f"✅ Analysis: {len(analysis)} metrics")
        
        # Cleanup
        visit_interface.close()
        print("✅ Interface closed successfully")
        
    except Exception as e:
        print(f"❌ Error with visit module: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ Error importing interface: {e}")
    import traceback
    traceback.print_exc()
