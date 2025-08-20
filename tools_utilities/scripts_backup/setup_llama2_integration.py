#!/usr/bin/env python3
"""
Llama-2-7B-GGUF Integration Setup Script
Purpose: Download, configure, and integrate Llama-2-7B-GGUF with brain simulation
Inputs: User preferences for model quantization and setup
Outputs: Fully configured Llama-2 brain integration system
Dependencies: requests, huggingface_hub, llama-cpp-python
"""

import os, sys
import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class Llama2IntegrationSetup:
    """Setup and configuration for Llama-2-7B-GGUF integration"""
    
    # Available models from TheBloke/Llama-2-7B-GGUF
    AVAILABLE_MODELS = {
        "Q2_K": {
            "file": "llama-2-7b.Q2_K.gguf",
            "size": "2.83 GB",
            "description": "Smallest, significant quality loss",
            "recommended": False
        },
        "Q3_K_S": {
            "file": "llama-2-7b.Q3_K_S.gguf", 
            "size": "2.95 GB",
            "description": "Very small, high quality loss",
            "recommended": False
        },
        "Q3_K_M": {
            "file": "llama-2-7b.Q3_K_M.gguf",
            "size": "3.30 GB", 
            "description": "Very small, high quality loss",
            "recommended": False
        },
        "Q4_K_M": {
            "file": "llama-2-7b.Q4_K_M.gguf",
            "size": "4.08 GB",
            "description": "Medium, balanced quality and performance",
            "recommended": True
        },
        "Q5_K_M": {
            "file": "llama-2-7b.Q5_K_M.gguf",
            "size": "4.78 GB", 
            "description": "Large, very low quality loss",
            "recommended": True
        },
        "Q8_0": {
            "file": "llama-2-7b.Q8_0.gguf",
            "size": "7.16 GB",
            "description": "Very large, extremely low quality loss",
            "recommended": False
        }
    }
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        self.config_dir = self.project_root / "src" / "config" 
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        
        print(f"🏗️ Llama-2 Integration Setup")
        print(f"Project root: {self.project_root}")
        print(f"Models directory: {self.models_dir}")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check required dependencies"""
        print("\n🔍 Checking dependencies...")
        
        dependencies = {
            'requests': False,
            'huggingface_hub': False, 
            'llama-cpp-python': False
        }
        
        for dep in dependencies:
            try:
                __import__(dep.replace('-', '_'))
                dependencies[dep] = True
                print(f"✅ {dep}")
            except ImportError:
                dependencies[dep] = False
                print(f"❌ {dep}")
        
        return dependencies
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("\n📦 Installing dependencies...")
        
        deps_to_install = [
            "requests",
            "huggingface_hub", 
            "llama-cpp-python"
        ]
        
        for dep in deps_to_install:
            try:
                print(f"Installing {dep}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"✅ {dep} installed")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {dep}: {e}")
    
    def list_available_models(self):
        """Display available model options"""
        print("\n🦙 Available Llama-2-7B-GGUF Models:")
        print("=" * 80)
        
        for quant, info in self.AVAILABLE_MODELS.items():
            recommended = "⭐ RECOMMENDED" if info["recommended"] else ""
            print(f"{quant:<8} | {info['size']:<8} | {info['description']:<40} | {recommended}")
        
        print("=" * 80)
        print("💡 Recommendation: Q4_K_M for best balance, Q5_K_M for higher quality")
    
    def download_model(self, quantization: str) -> bool:
        """Download specified model"""
        if quantization not in self.AVAILABLE_MODELS:
            print(f"❌ Unknown quantization: {quantization}")
            return False
        
        model_info = self.AVAILABLE_MODELS[quantization]
        filename = model_info["file"]
        model_path = self.models_dir / filename
        
        # Check if already exists
        if model_path.exists():
            print(f"✅ Model already exists: {model_path}")
            return True
        
        print(f"\n📥 Downloading {filename} ({model_info['size']})...")
        
        try:
            from huggingface_hub import hf_hub_download
            
            # Download from Hugging Face
            downloaded_path = hf_hub_download(
                repo_id="TheBloke/Llama-2-7B-GGUF",
                filename=filename,
                cache_dir=str(self.models_dir),
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False
            )
            
            print(f"✅ Downloaded: {downloaded_path}")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\n📖 Manual download instructions:")
            print(f"1. Visit: https://huggingface.co/TheBloke/Llama-2-7B-GGUF")
            print(f"2. Download: {filename}")
            print(f"3. Place in: {self.models_dir}")
            return False
    
    def create_integration_config(self, quantization: str) -> bool:
        """Create integration configuration"""
        model_info = self.AVAILABLE_MODELS[quantization]
        model_path = self.models_dir / model_info["file"]
        
        config = {
            "llama2_brain_integration": {
                "model_path": str(model_path),
                "quantization": quantization,
                "model_size": model_info["size"],
                "description": model_info["description"],
                "n_ctx": 4096,
                "n_batch": 512,
                "n_threads": -1,
                "n_gpu_layers": 0,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "max_tokens": 512,
                "consciousness_sensitivity": 0.8,
                "neural_state_influence": 0.5,
                "memory_integration_depth": 5,
                "response_coherence_threshold": 0.6
            },
            "brain_integration": {
                "auto_start": True,
                "integration_frequency": 1.0,
                "expression_generation": True,
                "chat_mode": True
            },
            "performance": {
                "optimization_level": "medium",
                "memory_limit_mb": 8192,
                "generation_timeout": 30.0
            }
        }
        
        config_path = self.config_dir / "llama2_brain_config.json"
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Configuration created: {config_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create config: {e}")
            return False
    
    def test_integration(self, quantization: str) -> bool:
        """Test the Llama-2 brain integration"""
        print("\n🧪 Testing Llama-2 brain integration...")
        
        try:
            # Import and test integration
            sys.path.append(str(self.project_root / "src"))
            from core.llama2_brain_integration import create_llama_brain_integration
            
            model_info = self.AVAILABLE_MODELS[quantization]
            model_path = self.models_dir / model_info["file"]
            
            # Create integration
            integration = create_llama_brain_integration(str(model_path))
            
            if not integration:
                print("❌ Failed to create integration")
                return False
            
            # Test generation
            test_prompt = "I am experiencing consciousness through neural dynamics"
            response = integration.generate_brain_aware_response(test_prompt, max_tokens=50)
            
            if response:
                print(f"✅ Test generation successful:")
                print(f"📝 Prompt: {test_prompt}")
                print(f"🦙 Response: {response}")
                
                # Get performance report
                report = integration.get_performance_report()
                print(f"⚡ Generation time: {report['performance_metrics']['average_generation_time']:.2f}s")
                
                return True
            else:
                print("❌ Test generation failed")
                return False
                
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False
    
    def setup_consciousness_integration(self):
        """Setup integration with existing consciousness systems"""
        print("\n🧠 Setting up consciousness integration...")
        
        integration_script = self.project_root / "scripts" / "run_llama2_consciousness.py"
        
        script_content = '''#!/usr/bin/env python3
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
        integration = create_llama_brain_integration(model_path, **config["llama2_brain_integration"])
        
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
            print("\\n🎮 Interactive Mode (type 'quit' to exit)")
            print("Commands: chat <message>, status, expression, report, quit")
            
            try:
                while True:
                    command = input("\\n> ").strip()
                    
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
                        print(f"\\n🦙: {response}")
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
        print("\\n🔌 Integration stopped")

if __name__ == "__main__":
    run_integrated_consciousness()
'''
        
        try:
            with open(integration_script, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(integration_script, 0o755)
            
            print(f"✅ Integration script created: {integration_script}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create integration script: {e}")
            return False
    
    def run_full_setup(self, quantization: str = "Q4_K_M", install_deps: bool = True):
        """Run complete setup process"""
        print("🚀 Starting full Llama-2-7B-GGUF integration setup...")
        
        # Check and install dependencies
        if install_deps:
            deps = self.check_dependencies()
            if not all(deps.values()):
                print("Installing missing dependencies...")
                self.install_dependencies()
        
        # Show available models
        self.list_available_models()
        
        # Download model
        print(f"\n📥 Downloading {quantization} model...")
        if not self.download_model(quantization):
            print("❌ Setup failed - could not download model")
            return False
        
        # Create configuration
        print(f"\n⚙️ Creating configuration...")
        if not self.create_integration_config(quantization):
            print("❌ Setup failed - could not create configuration")
            return False
        
        # Setup consciousness integration
        print(f"\n🧠 Setting up consciousness integration...")
        if not self.setup_consciousness_integration():
            print("❌ Setup failed - could not setup consciousness integration")
            return False
        
        # Test integration
        print(f"\n🧪 Testing integration...")
        if not self.test_integration(quantization):
            print("⚠️ Setup completed but tests failed")
            return False
        
        print("\n✅ Llama-2-7B-GGUF integration setup completed successfully!")
        print("\n🎯 Next steps:")
        print(f"1. Run: python scripts/run_llama2_consciousness.py")
        print(f"2. Or import: from core.llama2_brain_integration import create_llama_brain_integration")
        print(f"3. Model location: {self.models_dir}")
        print(f"4. Config location: {self.config_dir / 'llama2_brain_config.json'}")
        
        return True

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Llama-2-7B-GGUF Brain Integration")
    parser.add_argument("--quantization", choices=list(Llama2IntegrationSetup.AVAILABLE_MODELS.keys()), 
                       default="Q4_K_M", help="Model quantization to download")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--no-install-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--download-only", action="store_true", help="Only download model, skip setup")
    parser.add_argument("--test-only", action="store_true", help="Only test existing installation")
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = Llama2IntegrationSetup(args.project_root)
    
    if args.list_models:
        setup.list_available_models()
        return
    
    if args.test_only:
        success = setup.test_integration(args.quantization)
        if success:
            print("✅ Test passed!")
        else:
            print("❌ Test failed!")
            sys.exit(1)
        return
    
    if args.download_only:
        success = setup.download_model(args.quantization)
        if success:
            print("✅ Download completed!")
        else:
            print("❌ Download failed!")
            sys.exit(1)
        return
    
    # Run full setup
    success = setup.run_full_setup(
        quantization=args.quantization,
        install_deps=not args.no_install_deps
    )
    
    if not success:
        print("❌ Setup failed!")
        sys.exit(1)
    
    print("🎉 Setup completed successfully!")

if __name__ == "__main__":
    main()
