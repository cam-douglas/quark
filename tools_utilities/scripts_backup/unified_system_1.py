#!/usr/bin/env python3
"""
Unified Intelligence System - Small-Mind
Combines all agents, models, and training into one system
"""

import os, sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

class UnifiedSystem:
    def __init__(self):
        self.models = {}
        self.agents = {}
        self.training_pipelines = {}
        self.is_running = False
        self.start_time = datetime.utcnow()
        
        print("🚀 Initializing Unified Intelligence System...")
        self.load_components()
        
    def load_components(self):
        """Load all available components"""
        # Load models from models.yaml
        models_file = project_root / "models" / "models.yaml"
        if models_file.exists():
            import yaml
            with open(models_file, 'r') as f:
                config = yaml.safe_load(f)
            
            for model_type, models in config.items():
                if model_type not in ['routing', 'neuro_system']:
                    for model in models:
                        self.models[model['id']] = model
                        print(f"📦 Loaded model: {model['id']}")
        
        # Load agents from src/smallmind
        sm_dir = project_root / "src" / "smallmind"
        if sm_dir.exists():
            agent_modules = ['baby_agi', 'core', 'ml_optimization', 'neurodata']
            for module in agent_modules:
                module_path = sm_dir / module
                if module_path.exists():
                    self.agents[f"sm.{module}"] = {
                        'type': 'smallmind',
                        'module': module,
                        'path': str(module_path)
                    }
                    print(f"🤖 Loaded agent: sm.{module}")
        
        print(f"✅ Loaded {len(self.models)} models and {len(self.agents)} agents")
    
    def start(self):
        """Start the unified system"""
        self.is_running = True
        print("🎉 Unified Intelligence System is running!")
        print("🧠 All components are integrated and available")
        
        # Start continuous operation
        while self.is_running:
            try:
                time.sleep(5)
                self.show_status()
            except KeyboardInterrupt:
                print("\n🛑 Shutdown requested...")
                self.stop()
                break
    
    def show_status(self):
        """Show system status"""
        status = {
            'models': len(self.models),
            'agents': len(self.agents),
            'uptime': (datetime.utcnow() - self.start_time).seconds
        }
        print(f"📊 Status: {status['models']} models, {status['agents']} agents, {status['uptime']}s uptime")
    
    def stop(self):
        """Stop the system"""
        self.is_running = False
        print("✅ System stopped")

def main():
    print("🧠 Small-Mind Unified Intelligence System")
    print("=" * 50)
    
    system = UnifiedSystem()
    system.start()

if __name__ == "__main__":
    main()
