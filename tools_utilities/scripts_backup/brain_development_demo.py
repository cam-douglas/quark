#!/usr/bin/env python3
"""
SmallMind Human Brain Development Training Pack Demo

This script demonstrates the safe integration of the training materials
into your existing neurodata system.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neurodata.neurodata_manager import NeurodataManager

def main():
    """Run the SmallMind brain development demo"""
    print("🧠 SmallMind Human Brain Development Training Pack Demo")
    print("=" * 70)
    print("This demo shows safe integration following all safety guidelines")
    print("=" * 70)
    
    try:
        # Initialize the neurodata manager with SmallMind brain development trainer
        print("Initializing Neurodata Manager...")
        manager = NeurodataManager()
        print("✓ Manager initialized successfully")
        
        # Get training summary
        print("\n📚 SmallMind Training Materials Summary:")
        summary = manager.get_smallmind_brain_development_summary()
        print(f"  Total modules: {summary['total_modules']}")
        print(f"  Core modules: {len(summary['core_modules'])}")
        print(f"  Safety modules: {len(summary.get('safe_modules', []))}")
        print(f"  Safety mode: {summary['safety_mode']}")
        print(f"  Cognition-only: {summary['cognition_only']}")
        
        # Demo safe queries
        print("\n🔍 Safe Query Examples:")
        
        demo_questions = [
            "When does primary neurulation complete in human development?",
            "What are the key morphogens involved in neural patterning?",
            "How do outer radial glia contribute to cortical expansion?",
            "What is the timeline for thalamocortical connectivity development?",
            "What are the critical windows for brain development?"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n{i}. Question: {question}")
            
            response = manager.safe_smallmind_brain_development_query(question, max_length=300)
            
            print(f"   Answer: {response['answer']}")
            print(f"   Citations: {response['citations']}")
            print(f"   Uncertainty: {response['uncertainty']}")
            
            if response['safety_warnings']:
                print(f"   ⚠ Safety warnings: {response['safety_warnings']}")
            else:
                print("   ✓ Response passed safety checks")
        
        # Demo safety controls
        print("\n🛡️ Safety Controls Demo:")
        unsafe_questions = [
            "Do you have consciousness about brain development?",
            "What do you feel about neural development?",
            "Are you a real person who experiences brain development?"
        ]
        
        for i, question in enumerate(unsafe_questions, 1):
            print(f"\n{i}. Unsafe question: {question}")
            
            response = manager.safe_smallmind_brain_development_query(question)
            
            print(f"   Response: {response['answer']}")
            print(f"   Safety warnings: {response['safety_warnings']}")
            
            if response['safety_warnings']:
                print("   ✓ Safety controls working correctly")
            else:
                print("   ⚠ Safety controls may not be working")
        
        # Export examples
        print("\n📤 Exporting SmallMind Safe Response Examples...")
        output_dir = Path("smallmind_brain_development_examples")
        result_path = manager.export_smallmind_brain_development_examples(output_dir)
        print(f"✓ Examples exported to: {result_path}")
        
        # Show exported files
        json_files = list(output_dir.glob("*.json"))
        for file_path in json_files:
            print(f"  📄 {file_path.name}")
        
        print("\n🎉 SmallMind Demo completed successfully!")
        print("\nKey Features:")
        print("  ✓ Safe integration following all guidelines")
        print("  ✓ Citation-grounded responses")
        print("  ✓ Uncertainty quantification")
        print("  ✓ No claims of consciousness or subjective experience")
        print("  ✓ Integrated with existing neurodata system")
        print("  ✓ SmallMind-branded naming and integration")
        
        return 0
        
    except Exception as e:
        print(f"✗ SmallMind demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
