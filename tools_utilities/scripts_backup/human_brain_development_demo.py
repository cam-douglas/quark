#!/usr/bin/env python3
"""
Human Brain Development Training Pack Integration Demo

This script demonstrates the integration of the human brain development training pack
into the neurodata system. It shows how to access and query the structured knowledge
about prenatal neurodevelopment from fertilization through birth.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neurodata.human_brain_development import create_smallmind_brain_dev_trainer
from neurodata.neurodata_manager import NeurodataManager
from neurodata.enhanced_data_resources import create_enhanced_data_resources

def demo_brain_development_trainer():
    """Demo the brain development trainer directly"""
    print("=== Human Brain Development Trainer Demo ===\n")
    
    # Create trainer
    trainer = create_smallmind_brain_dev_trainer()
    
    # Test safe query functionality
    print("1. Safe Query Test:")
    test_question = "What is neurulation in human brain development?"
    response = trainer.safe_query(test_question, max_length=500)
    print(f"   Question: {test_question}")
    print(f"   Answer: {response.get('answer', 'N/A')[:200]}...")
    print(f"   Citations: {len(response.get('citations', []))}")
    print(f"   Uncertainty: {response.get('uncertainty', 'N/A')}")
    print(f"   Safety Warnings: {response.get('safety_warnings', [])}")
    print()
    
    # Get training summary
    print("2. Training Summary:")
    summary = trainer.get_training_summary()
    print(f"   Summary: {summary}")
    print()
    
    # Test another safe query
    print("3. Another Safe Query Test:")
    test_question2 = "What are the key stages of cortical development?"
    response2 = trainer.safe_query(test_question2, max_length=500)
    print(f"   Question: {test_question2}")
    print(f"   Answer: {response2.get('answer', 'N/A')[:200]}...")
    print(f"   Citations: {len(response2.get('citations', []))}")
    print(f"   Uncertainty: {response2.get('uncertainty', 'N/A')}")
    print()
    
    # Export safe responses
    print("4. Exporting Safe Responses...")
    output_path = Path("brain_development_export")
    export_file = trainer.export_safe_responses(output_path)
    print(f"   Exported to: {export_file}")
    
    return trainer

def demo_neurodata_manager():
    """Demo the neurodata manager with brain development integration"""
    print("\n=== Neurodata Manager with Brain Development Integration Demo ===\n")
    
    # Create neurodata manager
    manager = NeurodataManager()
    
    # Search across all sources including brain development
    print("1. Cross-source search for 'cortex':")
    search_results = manager.search_across_sources("cortex", species="human")
    
    for source, results in search_results.items():
        print(f"   {source}: {len(results)} results")
        if results:
            for result in results[:2]:  # Show first 2 results
                if 'name' in result:
                    print(f"     - {result['name']}")
                elif 'title' in result:
                    print(f"     - {result['title']}")
    
    # Test safe brain development query
    print("\n2. Safe Brain Development Query:")
    response = manager.safe_brain_development_query("What is the timeline of human brain development?")
    print(f"   Answer: {response.get('answer', 'N/A')[:200]}...")
    print(f"   Citations: {len(response.get('citations', []))}")
    print(f"   Uncertainty: {response.get('uncertainty', 'N/A')}")
    
    # Get brain development summary
    print("\n3. Brain Development Summary:")
    summary = manager.get_brain_development_summary()
    print(f"   Summary: {summary}")
    
    # Test another query
    print("\n4. Another Brain Development Query:")
    response2 = manager.safe_brain_development_query("What are the key morphogens in brain patterning?")
    print(f"   Answer: {response2.get('answer', 'N/A')[:200]}...")
    print(f"   Citations: {len(response2.get('citations', []))}")
    print(f"   Uncertainty: {response2.get('uncertainty', 'N/A')}")
    
    # Export examples
    print("\n5. Exporting Brain Development Examples...")
    output_path = Path("brain_development_examples")
    export_file = manager.export_brain_development_examples(output_path)
    print(f"   Exported to: {export_file}")
    
    return manager

async def demo_enhanced_data_resources():
    """Demo the enhanced data resources with brain development"""
    print("\n=== Enhanced Data Resources with Brain Development Demo ===\n")
    
    # Create enhanced data resources
    resources = create_enhanced_data_resources()
    
    # Get brain development data
    print("1. Brain Development Data:")
    brain_dev_data = resources.safe_brain_development_query("What is neurulation?")
    if 'error' not in brain_dev_data:
        print(f"   Answer: {brain_dev_data.get('answer', 'N/A')[:200]}...")
        print(f"   Citations: {len(brain_dev_data.get('citations', []))}")
        print(f"   Uncertainty: {brain_dev_data.get('uncertainty', 'N/A')}")
    else:
        print(f"   Error: {brain_dev_data['error']}")
    
    # Get comprehensive update with brain development
    print("\n2. Comprehensive Neuroscience Update with Brain Development:")
    comprehensive_update = await resources.get_comprehensive_neuroscience_update_with_brain_development(7)
    
    if 'brain_development' in comprehensive_update:
        brain_dev = comprehensive_update['brain_development']
        if 'error' not in brain_dev:
            print(f"   Summary: {brain_dev.get('summary', 'N/A')}")
        else:
            print(f"   Error: {brain_dev['error']}")
    
    # Show summary
    summary = comprehensive_update.get('summary', {})
    print(f"\n   Total sources: {len(summary.get('sources', []))}")
    print(f"   Sources: {', '.join(summary.get('sources', []))}")
    
    return resources

def main():
    """Main demo function"""
    print("Human Brain Development Training Pack Integration Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Direct trainer
        trainer = demo_brain_development_trainer()
        
        # Demo 2: Neurodata manager integration
        manager = demo_neurodata_manager()
        
        # Demo 3: Enhanced data resources integration
        print("\nRunning enhanced data resources demo...")
        asyncio.run(demo_enhanced_data_resources())
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("\nThe human brain development training pack has been integrated into:")
        print("- Direct trainer for standalone use")
        print("- Neurodata manager for cross-source searches")
        print("- Enhanced data resources for comprehensive updates")
        
        print("\nTraining data includes:")
        print("- Development timeline from fertilization to birth")
        print("- Carnegie stages and gestational weeks")
        print("- Cell types and lineages")
        print("- Morphogens and patterning")
        print("- Developmental processes")
        print("- Disorders and critical windows")
        print("- Modeling notes and ethics")
        print("- Comprehensive bibliography")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
