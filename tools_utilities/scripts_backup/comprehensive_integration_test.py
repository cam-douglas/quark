#!/usr/bin/env python3
"""
Comprehensive Integration Test for SmallMind Brain Development Training Pack

This script tests the complete integration of the human brain development training pack
with all components of the OmniNode system:
- Direct trainer access
- Neurodata manager integration
- Enhanced data resources integration
- Neuroscience experts system integration
- MoE (Mixture of Experts) system integration
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
from models.neuroscience_experts import (
    create_neuroscience_expert_manager, 
    NeuroscienceTask, 
    NeuroscienceTaskType
)
from models.moe_manager import create_moe_manager, ExecutionMode

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
        
        # Test safe query
        response = trainer.safe_query("What is neurulation?", max_length=500)
        print(f"✅ Safe query response: {len(response.get('answer', ''))} characters")
        
        # Test development timeline
        timeline = trainer.get_development_timeline()
        print(f"✅ Development timeline: {len(timeline)} stages")
        
        # Test cell types
        cell_types = trainer.get_cell_types_by_stage('all')
        print(f"✅ Cell types: {len(cell_types)} types found")
        
        # Test morphogens
        morphogens = trainer.get_morphogens_by_stage('all')
        print(f"✅ Morphogens: {len(morphogens)} morphogens found")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Direct trainer test failed: {e}")
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
        
        # Test cross-source search
        search_results = manager.search_across_sources("cortex", species="human")
        print(f"✅ Cross-source search: {len(search_results)} sources")
        
        # Test brain development specific search
        brain_dev_results = manager.search_brain_development_knowledge("radial glia")
        print(f"✅ Brain development search: {len(brain_dev_results.get('stages', []))} stages")
        
        return manager
        
    except Exception as e:
        print(f"❌ Neurodata manager test failed: {e}")
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
        
        # Test training summary
        summary = resources.get_brain_development_training_summary()
        print(f"✅ Training summary: {summary.get('total_modules', 0)} modules")
        
        # Test comprehensive update
        comprehensive_update = await resources.get_comprehensive_neuroscience_update_with_brain_development(7)
        print(f"✅ Comprehensive update: {len(comprehensive_update.get('sources', []))} sources")
        
        return resources
        
    except Exception as e:
        print(f"❌ Enhanced data resources test failed: {e}")
        return None

def test_neuroscience_experts():
    """Test neuroscience experts system integration"""
    print("\n=== Testing Neuroscience Experts System Integration ===\n")
    
    try:
        # Create expert manager
        expert_manager = create_neuroscience_expert_manager()
        print(f"✅ Expert manager created successfully")
        
        # Get available experts
        available_experts = expert_manager.get_available_experts()
        print(f"✅ Available experts: {len(available_experts)} experts")
        print(f"   Experts: {', '.join(available_experts)}")
        
        # Test brain development task
        brain_dev_task = NeuroscienceTask(
            task_type=NeuroscienceTaskType.BRAIN_DEVELOPMENT,
            description="What is the timeline of human brain development from fertilization to birth?",
            parameters={"max_length": 1000},
            expected_output="Timeline of brain development stages"
        )
        
        # Execute task
        result = expert_manager.execute_task(brain_dev_task)
        print(f"✅ Brain development task executed: {result.get('success', False)}")
        
        if result.get('success'):
            print(f"   Routed to: {result.get('routed_to_expert', 'Unknown')}")
            print(f"   Confidence: {result.get('routing_confidence', 0.0)}")
        
        # Test another brain development task
        morphogen_task = NeuroscienceTask(
            task_type=NeuroscienceTaskType.BRAIN_DEVELOPMENT,
            description="What are the key morphogens involved in brain patterning?",
            parameters={"max_length": 800},
            expected_output="List of key morphogens and their roles"
        )
        
        result2 = expert_manager.execute_task(morphogen_task)
        print(f"✅ Morphogen task executed: {result2.get('success', False)}")
        
        return expert_manager
        
    except Exception as e:
        print(f"❌ Neuroscience experts test failed: {e}")
        return None

async def test_moe_system():
    """Test MoE (Mixture of Experts) system integration"""
    print("\n=== Testing MoE System Integration ===\n")
    
    try:
        # Create MoE manager
        moe_manager = create_moe_manager(execution_mode=ExecutionMode.SINGLE_EXPERT)
        print(f"✅ MoE manager created successfully")
        
        # Test brain development query
        brain_dev_query = "What are the key stages of cortical development in human brain development?"
        response = await moe_manager.process_query(
            query=brain_dev_query,
            task_type=NeuroscienceTaskType.BRAIN_DEVELOPMENT
        )
        
        print(f"✅ MoE query processed: {response.primary_expert}")
        print(f"   Confidence: {response.confidence}")
        print(f"   Execution time: {response.execution_time:.3f}s")
        
        # Test another query
        timeline_query = "What is the timeline of neurulation in human embryos?"
        response2 = await moe_manager.process_query(
            query=timeline_query,
            task_type=NeuroscienceTaskType.BRAIN_DEVELOPMENT
        )
        
        print(f"✅ Timeline query processed: {response2.primary_expert}")
        print(f"   Confidence: {response2.confidence}")
        print(f"   Execution time: {response2.execution_time:.3f}s")
        
        return moe_manager
        
    except Exception as e:
        print(f"❌ MoE system test failed: {e}")
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
        return False

async def test_end_to_end_workflow():
    """Test end-to-end workflow"""
    print("\n=== Testing End-to-End Workflow ===\n")
    
    try:
        # Create all components
        trainer = create_smallmind_brain_dev_trainer()
        manager = NeurodataManager()
        resources = create_enhanced_data_resources()
        expert_manager = create_neuroscience_expert_manager()
        moe_manager = create_moe_manager()
        
        print(f"✅ All components created successfully")
        
        # Test complete workflow
        query = "What is the complete timeline of human brain development from fertilization to birth, including key stages, cell types, and morphogens?"
        
        # 1. Direct trainer query
        trainer_response = trainer.safe_query(query, max_length=1000)
        print(f"✅ Direct trainer response: {len(trainer_response.get('answer', ''))} characters")
        
        # 2. Neurodata manager query
        manager_response = manager.safe_brain_development_query(query)
        print(f"✅ Manager response: {len(manager_response.get('answer', ''))} characters")
        
        # 3. Expert system query
        expert_task = NeuroscienceTask(
            task_type=NeuroscienceTaskType.BRAIN_DEVELOPMENT,
            description=query,
            parameters={"max_length": 1000},
            expected_output="Complete brain development timeline"
        )
        expert_result = expert_manager.execute_task(expert_task)
        print(f"✅ Expert system response: {expert_result.get('success', False)}")
        
        # 4. MoE system query
        moe_response = await moe_manager.process_query(
            query=query,
            task_type=NeuroscienceTaskType.BRAIN_DEVELOPMENT
        )
        print(f"✅ MoE system response: {moe_response.primary_expert}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("SmallMind Brain Development Training Pack - Comprehensive Integration Test")
    print("=" * 80)
    
    start_time = time.time()
    test_results = {}
    
    try:
        # Test 1: Direct trainer
        test_results['direct_trainer'] = test_direct_trainer() is not None
        
        # Test 2: Neurodata manager
        test_results['neurodata_manager'] = test_neurodata_manager() is not None
        
        # Test 3: Enhanced data resources
        test_results['enhanced_data_resources'] = await test_enhanced_data_resources() is not None
        
        # Test 4: Neuroscience experts
        test_results['neuroscience_experts'] = test_neuroscience_experts() is not None
        
        # Test 5: MoE system
        test_results['moe_system'] = await test_moe_system() is not None
        
        # Test 6: Data export
        test_results['data_export'] = test_data_export()
        
        # Test 7: End-to-end workflow
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
