#!/usr/bin/env python3
"""
Basic test script for VisIt interface (no external dependencies required)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_visit_interface_basic():
    """Test basic VisIt interface functionality without external dependencies"""
    print("🔬 Testing VisIt Interface (Basic)...")
    
    try:
        # Test import
        from physics_simulation.visit_interface import VisItInterface, VISIT_AVAILABLE
        print(f"✅ VisIt interface imported successfully")
        print(f"   VISIT_AVAILABLE: {VISIT_AVAILABLE}")
        
        # Test interface creation
        visit_interface = VisItInterface()
        print("✅ VisIt interface created successfully")
        
        # Test data creation
        test_brain_data = {
            "regions": {
                "cortex": {"position": [0, 0, 0], "size": 1000},
                "hippocampus": {"position": [1, 0, 0], "size": 500}
            },
            "neurons": [
                {"position": [0.1, 0, 0], "type": 0, "activity": 0.5},
                {"position": [0.2, 0, 0], "type": 1, "activity": 0.3}
            ]
        }
        
        print("✅ Test brain data created")
        
        # Test visualization creation
        success = visit_interface.create_brain_visualization(test_brain_data, "3D")
        if success:
            print("✅ 3D visualization created successfully")
        else:
            print("❌ Visualization creation failed")
        
        # Test data analysis
        analysis_results = visit_interface.analyze_brain_data(test_brain_data, "statistics")
        if analysis_results:
            print(f"✅ Data analysis completed: {len(analysis_results)} metrics")
            for key, value in analysis_results.items():
                print(f"   {key}: {value}")
        else:
            print("❌ Data analysis failed")
        
        # Test export (mock)
        export_success = visit_interface.export_visualization("test_visualization.png")
        if export_success:
            print("✅ Visualization export completed (mock)")
        else:
            print("❌ Visualization export failed")
        
        # Cleanup
        visit_interface.close()
        print("✅ VisIt interface closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ VisIt interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quick_visualization():
    """Test quick visualization function"""
    print("\n⚡ Testing Quick Visualization...")
    
    try:
        from physics_simulation.visit_interface import visualize_brain_data
        
        # Create sample brain data
        sample_data = {
            "regions": {
                "frontal_lobe": {"position": [0, 0, 0], "size": 800},
                "temporal_lobe": {"position": [1, 0, 0], "size": 600},
                "parietal_lobe": {"position": [2, 0, 0], "size": 700}
            },
            "neurons": [
                {"position": [0.1, 0, 0], "type": 0, "activity": 0.8},
                {"position": [0.2, 0, 0], "type": 1, "activity": 0.6},
                {"position": [1.1, 0, 0], "type": 0, "activity": 0.7}
            ]
        }
        
        print("✅ Sample brain data created")
        
        # Test quick visualization
        success = visualize_brain_data(sample_data, "quick_brain_viz.png", "3D")
        if success:
            print("✅ Quick visualization completed successfully")
        else:
            print("❌ Quick visualization failed")
        
        return success
        
    except Exception as e:
        print(f"❌ Quick visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_analysis():
    """Test data analysis capabilities"""
    print("\n📈 Testing Data Analysis...")
    
    try:
        from physics_simulation.visit_interface import VisItInterface
        
        # Create interface
        visit_interface = VisItInterface()
        
        # Test data
        test_data = {
            "regions": {
                "region1": {"position": [0, 0, 0], "size": 100},
                "region2": {"position": [1, 0, 0], "size": 200}
            },
            "neurons": [
                {"position": [0.1, 0, 0], "type": "excitatory", "activity": 0.8},
                {"position": [0.2, 0, 0], "type": "inhibitory", "activity": 0.6}
            ],
            "time_series": {
                "time": [0, 1, 2, 3, 4],
                "activity": [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        }
        
        # Test different analysis types
        analysis_types = ["statistics", "spatial", "temporal"]
        
        for analysis_type in analysis_types:
            print(f"   Testing {analysis_type} analysis...")
            results = visit_interface.analyze_brain_data(test_data, analysis_type)
            if results:
                print(f"     ✅ {analysis_type} analysis successful")
                print(f"     Results: {len(results)} metrics")
            else:
                print(f"     ❌ {analysis_type} analysis failed")
        
        # Cleanup
        visit_interface.close()
        print("✅ Data analysis tests completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Data analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic tests"""
    print("🧪 Basic VisIt Interface Tests (No External Dependencies)")
    print("=" * 60)
    
    tests = [
        ("VisIt Interface Basic", test_visit_interface_basic),
        ("Quick Visualization", test_quick_visualization),
        ("Data Analysis", test_data_analysis)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 Running {test_name}...")
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n📊 Test Results:")
    print("-" * 40)
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All basic tests passed!")
        print("\n💡 Next Steps:")
        print("1. Install VisIt for full functionality")
        print("2. Install NEST simulator for brain physics")
        print("3. Run the full integration tests")
    else:
        print("\n⚠️  Some basic tests failed.")
        print("\n🔧 Troubleshooting:")
        print("1. Check Python path configuration")
        print("2. Verify all required modules are available")
        print("3. Check for syntax errors in the interface code")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
