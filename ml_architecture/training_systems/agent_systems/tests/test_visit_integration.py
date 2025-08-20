#!/usr/bin/env python3
"""
Test script for VisIt integration with brain physics simulator
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from physics_simulation.brain_physics import BrainPhysicsSimulator
from physics_simulation.visit_interface import VisItInterface, visualize_brain_data, VISIT_AVAILABLE

def test_visit_interface():
    """Test basic VisIt interface functionality"""
    print("🔬 Testing VisIt Interface...")
    
    try:
        # Test if VisIt is available (including mock)
        if not VISIT_AVAILABLE:
            print("⚠️  VisIt not available, skipping interface tests")
            return False
        
        # Create VisIt interface
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
        
        # Test visualization creation
        success = visit_interface.create_brain_visualization(test_brain_data, "3D")
        if success:
            print("✅ 3D visualization created successfully")
            
            # Test export
            export_success = visit_interface.export_visualization("test_visualization.png")
            if export_success:
                print("✅ Visualization exported successfully")
            else:
                print("❌ Visualization export failed")
        else:
            print("❌ Visualization creation failed")
        
        # Test data analysis
        analysis_results = visit_interface.analyze_brain_data(test_brain_data, "statistics")
        if analysis_results:
            print(f"✅ Data analysis completed: {len(analysis_results)} metrics")
        else:
            print("❌ Data analysis failed")
        
        # Cleanup
        visit_interface.close()
        print("✅ VisIt interface closed successfully")
        
        return success
        
    except Exception as e:
        print(f"❌ VisIt interface test failed: {e}")
        return False

def test_brain_physics_visit_integration():
    """Test VisIt integration with brain physics simulator"""
    print("🧠 Testing Brain Physics + VisIt Integration...")
    
    try:
        # Create brain physics simulator
        brain_sim = BrainPhysicsSimulator(simulation_time=100.0, time_step=0.1)
        
        # Setup brain development model
        brain_regions = ["cortex", "hippocampus"]
        cell_types = ["excitatory", "inhibitory"]
        region_sizes = {"cortex": 100, "hippocampus": 50}
        
        success = brain_sim.setup_brain_development_model(brain_regions, cell_types, region_sizes)
        if not success:
            print("❌ Failed to setup brain development model")
            return False
        
        print("✅ Brain development model setup successfully")
        
        # Test VisIt visualization
        if brain_sim.visit_interface:
            print("✅ VisIt interface available in brain simulator")
            
            # Test visualization
            vis_success = brain_sim.visualize_brain_development("3D", "brain_dev_3d.png")
            if vis_success:
                print("✅ Brain development visualization created")
            else:
                print("⚠️  Brain development visualization failed")
            
            # Test data analysis
            analysis_results = brain_sim.analyze_brain_data_with_visit("statistics")
            if analysis_results:
                print(f"✅ Brain data analysis completed: {analysis_results}")
            else:
                print("⚠️  Brain data analysis failed")
        else:
            print("⚠️  VisIt interface not available in brain simulator")
        
        # Cleanup
        brain_sim.cleanup()
        print("✅ Brain simulator cleaned up successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Brain physics + VisIt integration test failed: {e}")
        return False

def test_quick_visualization():
    """Test quick visualization function"""
    print("⚡ Testing Quick Visualization...")
    
    try:
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
        
        # Test quick visualization
        success = visualize_brain_data(sample_data, "quick_brain_viz.png", "3D")
        if success:
            print("✅ Quick visualization completed successfully")
        else:
            print("❌ Quick visualization failed")
        
        return success
        
    except Exception as e:
        print(f"❌ Quick visualization test failed: {e}")
        return False

def main():
    """Run all VisIt integration tests"""
    print("🧪 Testing VisIt Integration with Brain Physics Simulator")
    print("=" * 60)
    
    tests = [
        ("VisIt Interface", test_visit_interface),
        ("Brain Physics + VisIt Integration", test_brain_physics_visit_integration),
        ("Quick Visualization", test_quick_visualization)
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
        print("\n🎉 All VisIt integration tests passed!")
    else:
        print("\n⚠️  Some VisIt integration tests failed.")
    
    print("\n💡 Next Steps:")
    print("- Install VisIt if not already installed")
    print("- Use brain_sim.visualize_brain_development() for 3D brain visualization")
    print("- Use brain_sim.analyze_brain_data_with_visit() for data analysis")
    print("- Use visualize_brain_data() for quick visualization")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
