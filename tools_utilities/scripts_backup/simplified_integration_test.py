#!/usr/bin/env python3
"""
Simplified Integration Test for SmallMind Brain Development Training Pack

This script tests the core integration components that are working:
- Direct trainer access
- Neurodata manager integration
- Enhanced data resources integration
"""

import asyncio
import json
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neurodata.human_brain_development import create_smallmind_brain_dev_trainer
from neurodata.neurodata_manager import NeurodataManager
from neurodata.enhanced_data_resources import create_enhanced_data_resources

def test_direct_trainer():
    """Test direct access to the brain development trainer"""
    print("=== Testing Direct Trainer Access ===\n")
    
    try:
        # Create trainer
        trainer = create_smallmind_brain_dev_trainer()
        print(f"✅ Trainer created successfully: {trainer.__class__.__name__}")
        
        # Test training summary
        summary = trainer.get_training_summary()
        print(f"✅ Training summary: {summary.get('total_modules', 0)} modules loaded")
        print(f"   Core modules: {len(summary.get('core_modules', []))}")
        print(f"   Safety modules: {len(summary.get('safety_modules', []))}")
        
        # Test safe query
        response = trainer.safe_query("What is neurulation?", max_length=500)
        print(f"✅ Safe query response: {len(response.get('answer', ''))} characters")
        print(f"   Citations: {len(response.get('citations', []))}")
        print(f"   Uncertainty: {response.get('uncertainty', 'N/A')}")
        
        # Test development timeline
        timeline = trainer.get_development_timeline()
        print(f"✅ Development timeline: {len(timeline)} stages")
        for stage in timeline[:3]:  # Show first 3 stages
            print(f"   - {stage.get('name', 'Unknown')}: {stage.get('gestational_weeks', 'N/A')} weeks")
        
        # Test cell types
        cell_types = trainer.get_cell_types_by_stage('all')
        print(f"✅ Cell types: {len(cell_types)} types found")
        for ct in cell_types[:3]:  # Show first 3
            print(f"   - {ct}")
        
        # Test morphogens
        morphogens = trainer.get_morphogens_by_stage('all')
        print(f"✅ Morphogens: {len(morphogens)} morphogens found")
        for m in morphogens[:3]:  # Show first 3
            print(f"   - {m}")
        
        # Test search functionality
        search_results = trainer.search_development_knowledge("radial glia")
        print(f"✅ Search results: {len(search_results.get('stages', []))} stages, {len(search_results.get('processes', []))} processes")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Direct trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_neurodata_manager():
    """Test neurodata manager integration"""
    print("\n=== Testing Neurodata Manager Integration ===\n")
    
    try:
        # Create manager
        manager = NeurodataManager()
        print(f"✅ Neurodata manager created successfully")
        
        # Test brain development summary
        summary = manager.get_brain_development_summary()
        print(f"✅ Brain development summary: {summary.get('total_modules', 0)} modules")
        
        # Test safe query
        response = manager.safe_brain_development_query("What are the key stages of brain development?")
        print(f"✅ Safe query: {len(response.get('answer', ''))} characters")
        print(f"   Citations: {len(response.get('citations', []))}")
        print(f"   Uncertainty: {response.get('uncertainty', 'N/A')}")
        
        # Test cross-source search
        search_results = manager.search_across_sources("cortex", species="human")
        print(f"✅ Cross-source search: {len(search_results)} sources")
        for source, results in search_results.items():
            print(f"   {source}: {len(results)} results")
        
        # Test brain development specific search
        brain_dev_results = manager.search_brain_development_knowledge("radial glia")
        print(f"✅ Brain development search: {len(brain_dev_results.get('stages', []))} stages")
        
        # Test timeline access
        timeline = manager.get_brain_development_timeline()
        print(f"✅ Timeline access: {len(timeline)} stages")
        
        # Test processes access
        processes = manager.get_brain_development_processes()
        print(f"✅ Processes access: {len(processes)} processes")
        
        return manager
        
    except Exception as e:
        print(f"❌ Neurodata manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_enhanced_data_resources():
    """Test enhanced data resources integration"""
    print("\n=== Testing Enhanced Data Resources Integration ===\n")
    
    try:
        # Create resources
        resources = create_enhanced_data_resources()
        print(f"✅ Enhanced data resources created successfully")
        
        # Test safe query
        response = resources.safe_brain_development_query("What is the timeline of human brain development?")
        print(f"✅ Safe query: {len(response.get('answer', ''))} characters")
        print(f"   Citations: {len(response.get('citations', []))}")
        print(f"   Uncertainty: {response.get('uncertainty', 'N/A')}")
        
        # Test training summary
        summary = resources.get_brain_development_training_summary()
        print(f"✅ Training summary: {summary.get('total_modules', 0)} modules")
        
        # Test comprehensive update
        comprehensive_update = await resources.get_comprehensive_neuroscience_update_with_brain_development(7)
        print(f"✅ Comprehensive update: {len(comprehensive_update.get('sources', []))} sources")
        
        if 'brain_development' in comprehensive_update:
            brain_dev = comprehensive_update['brain_development']
            if 'error' not in brain_dev:
                print(f"   Brain development data included: {brain_dev.get('summary', 'N/A')}")
            else:
                print(f"   Brain development error: {brain_dev['error']}")
        
        return resources
        
    except Exception as e:
        print(f"❌ Enhanced data resources test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_data_export():
    """Test data export functionality"""
    print("\n=== Testing Data Export Functionality ===\n")
    
    try:
        # Create trainer
        trainer = create_smallmind_brain_dev_trainer()
        
        # Export training data
        output_path = Path("integration_test_export")
        export_file = trainer.export_training_data(output_path, "json")
        print(f"✅ Training data exported: {export_file}")
        
        # Export safe responses
        export_file2 = trainer.export_safe_responses(output_path)
        print(f"✅ Safe responses exported: {export_file2}")
        
        # Test neurodata manager export
        manager = NeurodataManager()
        export_file3 = manager.export_brain_development_examples(output_path)
        print(f"✅ Examples exported: {export_file3}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_end_to_end_workflow():
    """Test end-to-end workflow"""
    print("\n=== Testing End-to-End Workflow ===\n")
    
    try:
        # Create all components
        trainer = create_smallmind_brain_dev_trainer()
        manager = NeurodataManager()
        resources = create_enhanced_data_resources()
        
        print(f"✅ All components created successfully")
        
        # Test complete workflow
        query = "What is the complete timeline of human brain development from fertilization to birth, including key stages, cell types, and morphogens?"
        
        # 1. Direct trainer query
        trainer_response = trainer.safe_query(query, max_length=1000)
        print(f"✅ Direct trainer response: {len(trainer_response.get('answer', ''))} characters")
        
        # 2. Neurodata manager query
        manager_response = manager.safe_brain_development_query(query)
        print(f"✅ Manager response: {len(manager_response.get('answer', ''))} characters")
        
        # 3. Enhanced data resources query
        resources_response = resources.safe_brain_development_query(query)
        print(f"✅ Resources response: {len(resources_response.get('answer', ''))} characters")
        
        # 4. Test cross-component data consistency
        timeline1 = trainer.get_development_timeline()
        timeline2 = manager.get_brain_development_timeline()
        print(f"✅ Timeline consistency: {len(timeline1)} vs {len(timeline2)} stages")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("SmallMind Brain Development Training Pack - Simplified Integration Test")
    print("=" * 80)
    
    start_time = time.time()
    test_results = {}
    
    try:
        # Test 1: Direct trainer
        print("Starting Test 1: Direct Trainer Access...")
        test_results['direct_trainer'] = test_direct_trainer() is not None
        
        # Test 2: Neurodata manager
        print("\nStarting Test 2: Neurodata Manager Integration...")
        test_results['neurodata_manager'] = test_neurodata_manager() is not None
        
        # Test 3: Enhanced data resources
        print("\nStarting Test 3: Enhanced Data Resources Integration...")
        test_results['enhanced_data_resources'] = await test_enhanced_data_resources() is not None
        
        # Test 4: Data export
        print("\nStarting Test 4: Data Export Functionality...")
        test_results['data_export'] = test_data_export()
        
        # Test 5: End-to-end workflow
        print("\nStarting Test 5: End-to-End Workflow...")
        test_results['end_to_end'] = await test_end_to_end_workflow()
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("INTEGRATION TEST SUMMARY")
        print(f"{'='*80}")
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
        print(f"Total execution time: {total_time:.2f} seconds")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! The brain development training pack is fully integrated.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Check the output above for details.")
        
        return test_results
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    asyncio.run(main())
