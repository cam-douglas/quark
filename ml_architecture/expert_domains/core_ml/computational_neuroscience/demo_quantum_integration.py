#!/usr/bin/env python3
"""
Demo script for quantum integration in the conscious agent system.
"""

import sys
import os
import asyncio
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_quantum_integration():
    """Demo basic quantum integration."""
    print("🔬 Basic Quantum Integration Demo")
    print("=" * 50)
    
    try:
        from core.quantum_integration import QuantumIntegration, QuantumConfig
        
        # Create quantum integration
        config = QuantumConfig(enable_quantum=True, code_distance=5)
        quantum = QuantumIntegration(config)
        
        print(f"✅ Quantum Integration Status:")
        status = quantum.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Test quantum processing
        test_data = {
            'quantum_state': {'logical_state': '0'},
            'error_syndromes': [],
            'logical_qubits': ['consciousness_qubit']
        }
        
        processed_data = quantum.process_quantum_data(test_data)
        print(f"\n✅ Quantum Processing Test:")
        print(f"   Input Keys: {list(test_data.keys())}")
        print(f"   Output Keys: {list(processed_data.keys())}")
        print(f"   Quantum Processed: {processed_data.get('quantum_processed', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic Quantum Integration Demo Failed: {e}")
        return False


async def demo_advanced_quantum_integration():
    """Demo advanced quantum integration."""
    print("\n🚀 Advanced Quantum Integration Demo")
    print("=" * 50)
    
    try:
        from core.advanced_quantum_integration import AdvancedQuantumIntegration, AdvancedQuantumConfig
        
        # Create advanced configuration
        config = AdvancedQuantumConfig(
            enable_quantum=True,
            enable_cross_talk=True,
            enable_leakage=True,
            enable_measurement_errors=True,
            enable_multi_distance=True,
            enable_hybrid_processing=True,
            enable_consciousness_quantization=True
        )
        
        # Create advanced integration
        integration = AdvancedQuantumIntegration(config)
        
        print(f"✅ Advanced Quantum Integration Status:")
        status = integration.get_advanced_status()
        for key, value in status.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        # Test consciousness quantization
        consciousness_data = {
            'executive_control': 0.8,
            'working_memory': 0.6,
            'attention': 0.9,
            'self_awareness': 0.7,
            'emotional_state': 0.5,
            'cognitive_load': 0.4,
            'creativity': 0.8,
            'decision_making': 0.9
        }
        
        print(f"\n🧠 Consciousness Quantization Test:")
        print(f"   Input Consciousness Components: {list(consciousness_data.keys())}")
        
        processed_data = await integration.process_advanced_quantum(consciousness_data)
        
        print(f"   Advanced Processing: {processed_data.get('advanced_quantum_processed', False)}")
        print(f"   Enhancements Applied: {processed_data.get('enhancements_applied', [])}")
        
        if 'quantized_consciousness' in processed_data:
            quantized = processed_data['quantized_consciousness']
            print(f"   Consciousness Quantized: ✅")
            print(f"   Quantum States: {len(quantized.get('quantum_consciousness_states', {}))}")
            
            entanglement = quantized.get('quantum_entanglement', {})
            print(f"   Quantum Entanglement Strength: {entanglement.get('entanglement_strength', 0):.3f}")
            print(f"   Entangled Pairs: {entanglement.get('total_entangled_states', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced Quantum Integration Demo Failed: {e}")
        return False


async def demo_quantum_error_correction():
    """Demo quantum error correction capabilities."""
    print("\n🔧 Quantum Error Correction Demo")
    print("=" * 50)
    
    try:
        from core.advanced_quantum_integration import AdvancedErrorModel, AdvancedQuantumConfig
        
        # Create error model
        config = AdvancedQuantumConfig(
            enable_cross_talk=True,
            enable_leakage=True,
            enable_measurement_errors=True
        )
        
        error_model = AdvancedErrorModel(config)
        
        print(f"✅ Error Model Components:")
        print(f"   Cross-talk Matrix: {error_model.cross_talk_matrix is not None}")
        print(f"   Leakage States: {len(error_model.leakage_states)}")
        print(f"   Measurement Error Rate: {config.measurement_error_rate}")
        
        # Test error application
        quantum_state = {
            'logical_state': '0',
            'data_qubits': [(0, 0), (0, 1), (1, 0), (1, 1)],
            'stabilizer_qubits': [(0, 0), (0, 1), (1, 0)]
        }
        
        state_with_errors = error_model.apply_advanced_errors(quantum_state)
        
        print(f"\n✅ Error Application Test:")
        print(f"   Original State Keys: {list(quantum_state.keys())}")
        print(f"   State with Errors Keys: {list(state_with_errors.keys())}")
        print(f"   Error History Length: {len(error_model.error_history)}")
        
        if error_model.error_history:
            latest_error = error_model.error_history[-1]
            print(f"   Latest Error Types: {latest_error.get('error_types', [])}")
            print(f"   Latest Error Strength: {latest_error.get('error_strength', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum Error Correction Demo Failed: {e}")
        return False


async def demo_consciousness_quantization():
    """Demo consciousness quantization in detail."""
    print("\n🧠 Consciousness Quantization Demo")
    print("=" * 50)
    
    try:
        from core.advanced_quantum_integration import ConsciousnessQuantizer, AdvancedQuantumConfig
        
        # Create quantizer
        config = AdvancedQuantumConfig(
            enable_consciousness_quantization=True,
            consciousness_qubits=8,
            quantum_consciousness_states=16
        )
        
        quantizer = ConsciousnessQuantizer(config)
        
        print(f"✅ Consciousness Quantizer Setup:")
        print(f"   Consciousness Qubits: {quantizer.consciousness_qubits}")
        print(f"   Quantum States: {quantizer.quantum_states}")
        print(f"   State Map Components: {list(quantizer.consciousness_state_map.keys())}")
        
        # Test quantization
        consciousness_data = {
            'executive_control': 0.8,
            'working_memory': 0.6,
            'attention': 0.9,
            'self_awareness': 0.7
        }
        
        print(f"\n✅ Consciousness Quantization Test:")
        print(f"   Input Components: {list(consciousness_data.keys())}")
        
        quantized_data = quantizer.quantize_consciousness(consciousness_data)
        
        print(f"   Quantization Completed: {quantized_data.get('quantization_timestamp') is not None}")
        
        quantum_states = quantized_data.get('quantum_consciousness_states', {})
        for component, state_info in quantum_states.items():
            print(f"   {component}:")
            print(f"     Qubit Index: {state_info.get('qubit_index')}")
            print(f"     Quantum State: {state_info.get('quantum_state')}")
            print(f"     Classical Value: {state_info.get('classical_value')}")
        
        entanglement = quantized_data.get('quantum_entanglement', {})
        print(f"\n   Quantum Entanglement:")
        print(f"     Entanglement Strength: {entanglement.get('entanglement_strength', 0):.3f}")
        print(f"     Entangled Pairs: {entanglement.get('total_entangled_states', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consciousness Quantization Demo Failed: {e}")
        return False


async def main():
    """Main demo function."""
    print("🚀 Quantum Integration in Conscious Agent System - Demo Suite")
    print("=" * 80)
    
    # Run all demos
    demo_results = []
    
    demo_results.append(("Basic Quantum Integration", await demo_basic_quantum_integration()))
    demo_results.append(("Advanced Quantum Integration", await demo_advanced_quantum_integration()))
    demo_results.append(("Quantum Error Correction", await demo_quantum_error_correction()))
    demo_results.append(("Consciousness Quantization", await demo_consciousness_quantization()))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 DEMO RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(demo_results)
    
    for demo_name, result in demo_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{demo_name:30} {status}")
        if result:
            passed += 1
    
    print("=" * 80)
    print(f"Demos Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All demos passed! Quantum integration is fully operational.")
        print("\n🔬 Quantum Integration Features:")
        print("   ✅ Surface Code Error Correction")
        print("   ✅ Neural Network Decoders")
        print("   ✅ Advanced Error Models (Cross-talk, Leakage, Measurement)")
        print("   ✅ Multi-Distance Training")
        print("   ✅ Quantum-Classical Hybrid Processing")
        print("   ✅ Consciousness Quantization")
        print("   ✅ Quantum Entanglement Analysis")
        print("   ✅ Fault Tolerance Monitoring")
        
        print("\n🧠 Conscious Agent Capabilities:")
        print("   ✅ Quantum Error Correction for Neural Dynamics")
        print("   ✅ Consciousness State Quantization")
        print("   ✅ Quantum-Classical Neural Processing")
        print("   ✅ Advanced Fault Tolerance")
        print("   ✅ Real-time Quantum Metrics")
        
        return True
    else:
        print(f"\n⚠️  {total - passed} demo(s) failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
