#!/usr/bin/env python3
"""
Test Individual Fetal Brain Simulation Tools

This script tests the individual components of the fetal brain simulation tools
without the full demo dependencies.
"""

import asyncio
import json
from pathlib import Path

async def test_fetal_brain_simulation_tools():
    """Test the fetal brain simulation tools module directly"""
    
    print("🧪 Testing Fetal Brain Simulation Tools Components")
    print("=" * 60)
    
    try:
        # Import the fetal brain simulation tools
        from development.src.neurodata.fetal_brain_simulation import create_fetal_brain_simulation_tools
        
        # Create instance
        fetal_tools = create_fetal_brain_simulation_tools()
        
        print("✅ Successfully created FetalBrainSimulationTools instance")
        print(f"📊 Available tools: {len(fetal_tools.tools)}")
        
        # Test tool information
        print("\n🔧 Tool Information:")
        for tool_id, tool_info in fetal_tools.tools.items():
            print(f"  • {tool_info['name']} ({tool_id})")
            if 'type' in tool_info:
                print(f"    Type: {tool_info['type']}")
            if 'gestational_range' in tool_info:
                print(f"    Gestational Range: {tool_info['gestational_range']}")
            if 'version' in tool_info:
                print(f"    Version: {tool_info['version']}")
            print()
        
        # Test development timeline creation
        print("📅 Testing Development Timeline Creation:")
        timeline = fetal_tools.create_development_timeline(8, 37)
        print(f"  ✅ Timeline created: {timeline['gestational_range']}")
        print(f"  ✅ Stages: {len(timeline['stages'])}")
        print(f"  ✅ Integration points: {len(timeline['integration_points'])}")
        
        # Test integration recommendations
        print("\n🚀 Testing Integration Recommendations:")
        recommendations = await fetal_tools.get_integration_recommendations()
        print(f"  ✅ Recommendations generated: {len(recommendations['recommendations'])} categories")
        print(f"  ✅ Implementation phases: {len(recommendations['implementation_phases'])}")
        
        # Test Docker availability
        print(f"\n🐳 Docker Availability: {'✅ Available' if fetal_tools.docker_available else '❌ Not Available'}")
        
        # Export test results
        test_results = {
            'timestamp': timeline['gestational_range'],
            'tools_count': len(fetal_tools.tools),
            'timeline_stages': len(timeline['stages']),
            'integration_points': len(timeline['integration_points']),
            'docker_available': fetal_tools.docker_available,
            'recommendations_count': len(recommendations['recommendations']),
            'implementation_phases': len(recommendations['implementation_phases'])
        }
        
        output_dir = Path("fetal_brain_simulation_demo_export")
        output_dir.mkdir(exist_ok=True)
        
        test_file = output_dir / f"test_results_{timeline['gestational_range'].replace('-', '_')}.json"
        with open(test_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n📁 Test results exported to: {test_file}")
        print("\n✅ All fetal brain simulation tools tests passed!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

async def test_enhanced_data_resources():
    """Test the enhanced data resources module"""
    
    print("\n🔬 Testing Enhanced Data Resources")
    print("=" * 40)
    
    try:
        # Import enhanced data resources
        from development.src.neurodata.enhanced_data_resources import create_enhanced_data_resources
        
        # Create instance
        enhanced_resources = create_enhanced_data_resources()
        
        print("✅ Successfully created EnhancedDataResources instance")
        
        # Test fetal tools overview
        print("\n📊 Testing Fetal Tools Overview:")
        fetal_tools = await enhanced_resources.get_fetal_brain_simulation_tools()
        print(f"  ✅ Fetal tools retrieved: {len(fetal_tools['tools'])} categories")
        
        # Test pipeline creation
        print("\n🔄 Testing Pipeline Creation:")
        pipeline = await enhanced_resources.create_fetal_brain_development_pipeline()
        print(f"  ✅ Pipeline created: {pipeline['pipeline_name']}")
        print(f"  ✅ Coverage: {pipeline['gestational_coverage']['total_coverage']}")
        print(f"  ✅ Stages: {len(pipeline['pipeline_stages'])}")
        
        print("\n✅ Enhanced data resources tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

async def main():
    """Main test function"""
    
    print("🧪 SmallMind Fetal Brain Simulation Tools - Component Tests")
    print("=" * 80)
    
    # Test individual components
    fetal_tools_success = await test_fetal_brain_simulation_tools()
    enhanced_resources_success = await test_enhanced_data_resources()
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 40)
    print(f"Fetal Brain Simulation Tools: {'✅ PASSED' if fetal_tools_success else '❌ FAILED'}")
    print(f"Enhanced Data Resources: {'✅ PASSED' if enhanced_resources_success else '❌ FAILED'}")
    
    if fetal_tools_success and enhanced_resources_success:
        print("\n🎉 All component tests passed!")
        print("\nNext steps:")
        print("1. Begin Phase 1: Anatomical Structure Simulation")
        print("2. Set up FaBiAN alternatives or wait for public release")
        print("3. Install CompuCell3D and COPASI for cellular modeling")
        print("4. Prepare NEST and Emergent for neural simulation")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
    
    return fetal_tools_success and enhanced_resources_success

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)
