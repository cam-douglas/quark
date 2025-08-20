#!/usr/bin/env python3
"""Test script for advanced quantum integration"""

import sys
import os
import asyncio

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.advanced_quantum_integration import AdvancedQuantumIntegration, AdvancedQuantumConfig


async def test_advanced_integration():
    """Test advanced quantum integration."""
    print("🧪 Testing Advanced Quantum Integration...")
    
    try:
        # Create configuration
        config = AdvancedQuantumConfig(
            enable_quantum=True,
            enable_cross_talk=True,
            enable_leakage=True,
            enable_measurement_errors=True,
            enable_multi_distance=True,
            enable_hybrid_processing=True,
            enable_consciousness_quantization=True
        )
        
        # Create integration
        integration = AdvancedQuantumIntegration(config)
        
        # Test consciousness data
        consciousness_data = {
            'executive_control': 0.8,
            'working_memory': 0.6,
            'attention': 0.9,
            'self_awareness': 0.7
        }
        
        # Process data
        processed_data = await integration.process_advanced_quantum(consciousness_data)
        
        # Get status
        status = integration.get_advanced_status()
        
        print(f"✅ Advanced Quantum Integration Test:")
        print(f"   Quantum Available: {status.get('quantum_available', False)}")
        print(f"   Consciousness Quantization: {status.get('configuration', {}).get('consciousness_quantization', False)}")
        print(f"   Advanced Processing: {processed_data.get('advanced_quantum_processed', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Advanced Quantum Integration Test")
    print("=" * 50)
    
    success = await test_advanced_integration()
    
    if success:
        print("\n🎉 Test passed! Advanced quantum integration is working.")
    else:
        print("\n⚠️  Test failed. Check the output above for details.")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())
