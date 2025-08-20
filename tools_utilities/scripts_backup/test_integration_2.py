"""
Integration Test Script
Purpose: Test the integration between enhanced consciousness and main conscious agent
Inputs: All consciousness components
Outputs: Integration test results and validation
Seeds: Test patterns and validation checks
Dependencies: integrated_main_consciousness, enhanced_consciousness_simulator, brain_integration
"""

import os, sys
import time
import json
from datetime import datetime

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_component_imports():
    """Test if all required components can be imported"""
    print("🧪 Testing Component Imports...")
    
    test_results = {}
    
    # Test main consciousness agent
    try:
        from unified_consciousness_agent import UnifiedConsciousnessAgent
        test_results['main_agent'] = True
        print("✅ Main consciousness agent import successful")
    except ImportError as e:
        test_results['main_agent'] = False
        print(f"❌ Main consciousness agent import failed: {e}")
    
    # Test enhanced consciousness simulator
    try:
        from enhanced_consciousness_simulator import EnhancedConsciousnessSimulator
        test_results['enhanced_consciousness'] = True
        print("✅ Enhanced consciousness simulator import successful")
    except ImportError as e:
        test_results['enhanced_consciousness'] = False
        print(f"❌ Enhanced consciousness simulator import failed: {e}")
    
    # Test brain integration
    try:
        from brain_integration import BrainConsciousnessBridge
        test_results['brain_integration'] = True
        print("✅ Brain integration import successful")
    except ImportError as e:
        test_results['brain_integration'] = False
        print(f"❌ Brain integration import failed: {e}")
    
    # Test integrated main consciousness
    try:
        from integrated_main_consciousness import IntegratedMainConsciousness
        test_results['integrated_main'] = True
        print("✅ Integrated main consciousness import successful")
    except ImportError as e:
        test_results['integrated_main'] = False
        print(f"❌ Integrated main consciousness import failed: {e}")
    
    return test_results

def test_enhanced_consciousness():
    """Test enhanced consciousness simulator functionality"""
    print("\n🧪 Testing Enhanced Consciousness Simulator...")
    
    try:
        from enhanced_consciousness_simulator import EnhancedConsciousnessSimulator
        
        # Create simulator
        simulator = EnhancedConsciousnessSimulator()
        
        # Test basic functionality
        print("  Testing basic functionality...")
        simulator.start_simulation()
        time.sleep(1)
        
        # Test speech
        print("  Testing speech...")
        simulator.speak_thought("Testing speech functionality")
        time.sleep(2)
        
        # Test consciousness report
        print("  Testing consciousness report...")
        report = simulator.get_consciousness_report()
        print(f"    Consciousness level: {report['neural_state']['consciousness_level']:.2f}")
        
        # Clean up
        simulator.cleanup()
        
        print("✅ Enhanced consciousness simulator test passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced consciousness simulator test failed: {e}")
        return False

def test_brain_integration():
    """Test brain integration functionality"""
    print("\n🧪 Testing Brain Integration...")
    
    try:
        from brain_integration import BrainConsciousnessBridge
        
        # Create brain bridge
        bridge = BrainConsciousnessBridge()
        
        # Test with simulated brain state
        simulated_brain_state = {
            'pfc_firing_rate': 25.0,
            'bg_firing_rate': 45.0,
            'thalamus_firing_rate': 65.0,
            'loop_stability': 0.7,
            'feedback_strength': 0.8,
            'synchrony': 0.6,
            'oscillation_power': 0.5,
            'biological_realism': True
        }
        
        # Update brain state
        bridge.brain_state = simulated_brain_state
        
        # Map to consciousness state
        consciousness_state = bridge.map_to_consciousness_state()
        
        print(f"  PFC Firing Rate: {simulated_brain_state['pfc_firing_rate']:.1f} Hz")
        print(f"  → Consciousness Level: {consciousness_state['consciousness_level']:.2f}")
        print(f"  → Phase: {consciousness_state['phase']}")
        print(f"  → Stability: {consciousness_state['stability']}")
        
        print("✅ Brain integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Brain integration test failed: {e}")
        return False

def test_integrated_main_consciousness():
    """Test integrated main consciousness functionality"""
    print("\n🧪 Testing Integrated Main Consciousness...")
    
    try:
        from integrated_main_consciousness import IntegratedMainConsciousness
        
        # Create integrated consciousness
        integrated = IntegratedMainConsciousness()
        
        # Check component status
        print("  Component Status:")
        for component, status in integrated.integration_state.items():
            status_icon = "✅" if status else "❌"
            print(f"    {component}: {status_icon}")
        
        # Test integration start
        print("  Testing integration start...")
        if integrated.start_integration():
            print("    ✅ Integration started successfully")
            
            # Let it run for a moment
            time.sleep(2)
            
            # Test integrated report
            print("  Testing integrated report...")
            report = integrated.get_integrated_report()
            print(f"    Integration active: {report['integration_state']['integration_active']}")
            print(f"    Consciousness level: {report['unified_state']['consciousness_level']:.2f}")
            
            # Test speech
            print("  Testing unified speech...")
            integrated.speak_unified_thought("Testing unified consciousness speech")
            time.sleep(2)
            
            # Stop integration
            integrated.stop_integration()
            print("    ✅ Integration stopped successfully")
            
        else:
            print("    ❌ Integration start failed")
            return False
        
        print("✅ Integrated main consciousness test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integrated main consciousness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_integration():
    """Test full integration workflow"""
    print("\n🧪 Testing Full Integration Workflow...")
    
    try:
        from integrated_main_consciousness import IntegratedMainConsciousness
        
        # Create integrated consciousness
        integrated = IntegratedMainConsciousness()
        
        # Start integration
        if not integrated.start_integration():
            print("❌ Cannot test full integration - start failed")
            return False
        
        print("  Integration started, running workflow test...")
        
        # Simulate workflow
        workflow_steps = [
            ("Initialization", 1),
            ("State Synchronization", 2),
            ("Consciousness Generation", 2),
            ("Speech Synthesis", 2),
            ("Integration Monitoring", 2)
        ]
        
        for step_name, duration in workflow_steps:
            print(f"    Running: {step_name}")
            time.sleep(duration)
            
            # Check status
            report = integrated.get_integrated_report()
            consciousness_level = report['unified_state']['consciousness_level']
            print(f"      Consciousness level: {consciousness_level:.2f}")
        
        # Final status check
        final_report = integrated.get_integrated_report()
        print(f"  Final Status:")
        print(f"    Integration Active: {final_report['integration_state']['integration_active']}")
        print(f"    Consciousness Level: {final_report['unified_state']['consciousness_level']:.2f}")
        print(f"    Emotional State: {final_report['unified_state'].get('emotional_state', 'unknown')}")
        
        # Stop integration
        integrated.stop_integration()
        
        print("✅ Full integration workflow test passed")
        return True
        
    except Exception as e:
        print(f"❌ Full integration workflow test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive integration test suite"""
    print("🧠🔗 Comprehensive Integration Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Component imports
    test_results['component_imports'] = test_component_imports()
    
    # Test 2: Enhanced consciousness
    test_results['enhanced_consciousness'] = test_enhanced_consciousness()
    
    # Test 3: Brain integration
    test_results['brain_integration'] = test_brain_integration()
    
    # Test 4: Integrated main consciousness
    test_results['integrated_main'] = test_integrated_main_consciousness()
    
    # Test 5: Full integration workflow
    test_results['full_workflow'] = test_full_integration()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 40)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your integration is working perfectly!")
        print("\n🚀 Next steps:")
        print("1. Run the integrated main consciousness: python integrated_main_consciousness.py")
        print("2. Connect to brain simulation when ready")
        print("3. Monitor integration performance")
        print("4. Customize consciousness mapping")
        print("5. Scale up for research simulations")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("1. Verify all dependencies are installed")
        print("2. Check import paths and file structure")
        print("3. Ensure main consciousness agent is accessible")
        print("4. Verify speech and display libraries are working")
    
    return passed == total

def main():
    """Main test function"""
    print("Integration Test Suite")
    print("1. Run comprehensive test suite")
    print("2. Test individual components")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        run_comprehensive_test()
    elif choice == '2':
        print("Individual component tests:")
        print("1. Component imports")
        print("2. Enhanced consciousness")
        print("3. Brain integration")
        print("4. Integrated main consciousness")
        print("5. Full workflow")
        
        test_choice = input("Select test (1-5): ").strip()
        
        if test_choice == '1':
            test_component_imports()
        elif test_choice == '2':
            test_enhanced_consciousness()
        elif test_choice == '3':
            test_brain_integration()
        elif test_choice == '4':
            test_integrated_main_consciousness()
        elif test_choice == '5':
            test_full_integration()
        else:
            print("Invalid choice")
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice. Running comprehensive test...")
        run_comprehensive_test()

if __name__ == "__main__":
    main()
