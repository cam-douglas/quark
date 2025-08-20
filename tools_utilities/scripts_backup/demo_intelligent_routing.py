#!/usr/bin/env python3
"""
Demonstration of Intelligent Agent Routing

This script shows how the enhanced agent hub can automatically:
1. Detect user intent without explicit commands
2. Route to the best agent automatically
3. Collect feedback for continuous improvement
4. Train exponentially via cloud-based learning
"""

import sys
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def demo_intelligent_routing():
    """Demonstrate intelligent routing capabilities."""
    print("🚀 Small-Mind Intelligent Agent Routing Demo\n")
    
    try:
        from planner import auto_route_request, detect_intent, infer_needs
        from cloud_training import create_training_manager, EXAMPLE_CONFIG
        from intelligent_feedback import create_feedback_collector
        
        # Example prompts that don't need explicit commands
        example_prompts = [
            "How do I install Python packages using pip?",
            "Create a function to calculate fibonacci numbers",
            "Analyze the performance of this algorithm",
            "Optimize my database queries for better speed",
            "Build a web application with authentication",
            "What's the difference between lists and tuples in Python?",
            "Debug this code that's giving me an error",
            "Plan a machine learning project pipeline"
        ]
        
        print("📝 Example Prompts (No Commands Needed):")
        print("=" * 60)
        
        for i, prompt in enumerate(example_prompts, 1):
            print(f"{i}. {prompt}")
        
        print("\n" + "=" * 60)
        
        # Demo intelligent routing
        print("\n🧠 Intelligent Routing Analysis:")
        print("-" * 60)
        
        for prompt in example_prompts[:3]:  # Show first 3 examples
            print(f"\n🔍 Analyzing: '{prompt}'")
            
            # Auto-detect intent and route
            auto_result = auto_route_request(prompt)
            
            print(f"   Intent: {auto_result['intent']['primary_intent']}")
            print(f"   Action: {auto_result['routing']['action']}")
            print(f"   Model Type: {auto_result['routing']['model_type']}")
            print(f"   Priority: {auto_result['routing']['priority']}")
            print(f"   Response Format: {auto_result['response_config']['format']}")
            print(f"   Auto Command: {auto_result['auto_command']}")
            
            # Show detected needs
            needs = auto_result['needs']
            print(f"   Capabilities Needed: {', '.join(needs['need'])}")
            print(f"   Complexity: {needs['complexity']}")
        
        # Demo cloud training system
        print("\n☁️ Cloud-Based Training System:")
        print("-" * 60)
        
        # Create training manager (with example config)
        training_config = EXAMPLE_CONFIG.copy()
        training_config["cloud_endpoint"] = "https://demo.smallmind.ai/training"
        training_config["api_key"] = "demo_key_123"
        
        training_manager = create_training_manager(training_config)
        
        # Create feedback collector
        feedback_collector = create_feedback_collector(training_manager)
        
        # Show training status
        status = training_manager.get_training_status()
        print(f"📊 Training Status:")
        print(f"   Feedback Collected: {status['feedback_count']}")
        print(f"   Metrics Collected: {status['metrics_count']}")
        print(f"   Ready for Training: {status['ready_for_training']}")
        print(f"   Next Training: {status['next_training_estimate']}")
        
        # Demo feedback collection
        print(f"\n📝 Feedback Collection Demo:")
        
        # Simulate a response
        mock_response = {
            "result": {
                "stdout": "Here's how to install Python packages:\n\n1. Use pip: pip install package_name\n2. Use conda: conda install package_name\n3. Check versions: pip list",
                "stderr": "",
                "rc": 0
            },
            "run_dir": "demo_run_123"
        }
        
        # Collect feedback automatically
        feedback_id = feedback_collector.collect_execution_feedback(
            run_result=mock_response,
            user_prompt="How do I install Python packages?",
            model_id="demo.model",
            execution_metrics={
                "execution_time": 2.5,
                "resource_usage": {"memory_mb": 150, "cpu_percent": 25}
            }
        )
        
        print(f"   Feedback ID: {feedback_id}")
        
        # Show quality report
        quality_report = feedback_collector.get_quality_report(mock_response)
        print(f"   Overall Quality Score: {quality_report['overall_score']:.2f}/1.0")
        print(f"   Estimated User Rating: {quality_report['estimated_rating']}/5")
        
        # Show recommendations
        print(f"   Recommendations:")
        for rec in quality_report['recommendations']:
            print(f"     • {rec}")
        
        print("\n✅ Demo completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_usage_examples():
    """Show practical usage examples."""
    print("\n📚 Practical Usage Examples:")
    print("=" * 60)
    
    examples = [
        {
            "usage": "smctl auto 'How do I install numpy?'",
            "description": "Auto-detect intent and route to best agent",
            "benefit": "No need to remember commands - just ask naturally"
        },
        {
            "usage": "smctl auto 'Create a web scraper in Python'",
            "description": "Automatically routes to creation/development agent",
            "benefit": "Intelligent agent selection based on content"
        },
        {
            "usage": "smctl auto 'Analyze this algorithm complexity'",
            "description": "Routes to analysis/reasoning agent",
            "benefit": "Context-aware routing for optimal results"
        },
        {
            "usage": "smctl auto 'Optimize my database queries'",
            "description": "Routes to optimization/planning agent",
            "benefit": "Specialized agent selection for complex tasks"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['usage']}")
        print(f"   📖 {example['description']}")
        print(f"   ✨ {example['benefit']}")
    
    print("\n🔧 Advanced Features:")
    print("   • Automatic intent detection")
    print("   • Smart agent routing")
    print("   • Quality assessment")
    print("   • Continuous learning")
    print("   • Cloud-based training")

def main():
    """Run the demonstration."""
    print("🎯 Small-Mind Intelligent Agent Hub Demonstration")
    print("=" * 70)
    
    # Run intelligent routing demo
    success = demo_intelligent_routing()
    
    if success:
        # Show usage examples
        demo_usage_examples()
        
        print("\n" + "=" * 70)
        print("🎉 The system is now intelligent enough to:")
        print("   • Understand your intent without explicit commands")
        print("   • Route to the best agent automatically")
        print("   • Learn and improve continuously")
        print("   • Train exponentially via cloud-based learning")
        print("\n🚀 Try: smctl auto 'Your question here'")
    else:
        print("\n❌ Demo failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
