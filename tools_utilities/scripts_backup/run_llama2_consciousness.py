#!/usr/bin/env python3
"""
Llama-2 Consciousness Integration Runner
Automatically connects Llama-2-7B-GGUF with brain simulation and consciousness systems
"""

import os, sys
import json
import time
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "database"))

def run_integrated_consciousness():
    """Run Llama-2 with consciousness integration"""
    print("🧠🦙 Starting Llama-2 Consciousness Integration")
    print("=" * 60)
    
    try:
        # Load configuration
        config_path = project_root / "src" / "config" / "llama2_brain_config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Import components
        from core.llama2_brain_integration import create_llama_brain_integration
        
        # Create Llama integration
        model_path = config["llama2_brain_integration"]["model_path"]
        llama_config = config["llama2_brain_integration"].copy()
        llama_config.pop("model_path", None)  # Remove to avoid duplicate argument
        integration = create_llama_brain_integration(model_path, **llama_config)
        
        if not integration:
            print("❌ Failed to create Llama integration")
            return
        
        # Try to connect to consciousness systems
        try:
            from consciousness_agent.cloud_integrated_consciousness import CloudIntegratedConsciousness
            consciousness = CloudIntegratedConsciousness()
            consciousness.start_integration()
            
            # Connect Llama to consciousness
            integration.connect_consciousness_agent(consciousness)
            
            print("✅ Connected to consciousness system")
            
        except ImportError:
            print("⚠️ Consciousness system not available - running Llama only")
        
        # Start integration
        if integration.start_integration():
            print("🚀 Integration started successfully")
            
            # Interactive mode
            print("\n🎮 Interactive Mode (type 'quit' to exit)")
            print("Commands: chat <message>, status, expression, report, quit")
            
            try:
                while True:
                    command = input("\n> ").strip()
                    
                    if command.lower() == 'quit':
                        break
                    elif command.lower() == 'status':
                        report = integration.get_performance_report()
                        print(f"Status: {report['model_status']}")
                        print(f"Generations: {report['performance_metrics']['total_generations']}")
                    elif command.lower() == 'expression':
                        integration._generate_consciousness_expression()
                    elif command.lower() == 'report':
                        report = integration.get_performance_report()
                        print(json.dumps(report, indent=2))
                    elif command.startswith('chat '):
                        message = command[5:]
                        response = integration.chat_with_brain_context(message)
                        print(f"\n🦙: {response}")
                    else:
                        print("Unknown command. Use: chat <message>, status, expression, report, quit")
            
            except KeyboardInterrupt:
                pass
        
        else:
            print("❌ Failed to start integration")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'integration' in locals():
            integration.stop_integration()
        if 'consciousness' in locals():
            consciousness.stop_integration()
        print("\n🔌 Integration stopped")

if __name__ == "__main__":
    run_integrated_consciousness()
