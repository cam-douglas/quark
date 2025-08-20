#!/usr/bin/env python3
"""
Llama-2 Integration Status Check
Purpose: Verify complete Llama-2-7B-GGUF integration status
Inputs: System configuration and component availability
Outputs: Comprehensive status report
Dependencies: All Llama-2 integration components
"""

import os, sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "database"))

class Llama2StatusChecker:
    """Comprehensive status checker for Llama-2 integration"""
    
    def __init__(self):
        self.project_root = project_root
        self.status_report = {
            'dependencies': {},
            'models': {},
            'configuration': {},
            'components': {},
            'integration': {},
            'overall_status': 'unknown'
        }
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check required Python dependencies"""
        dependencies = {
            'llama-cpp-python': False,
            'transformers': False,
            'torch': False,
            'numpy': False,
            'requests': False,
            'huggingface_hub': False,
            'peft': False,
            'datasets': False,
            'accelerate': False
        }
        
        for dep in dependencies:
            try:
                __import__(dep.replace('-', '_'))
                dependencies[dep] = True
            except ImportError:
                dependencies[dep] = False
        
        self.status_report['dependencies'] = dependencies
        return dependencies
    
    def check_models(self) -> Dict[str, Any]:
        """Check available models"""
        models_dir = self.project_root / "models"
        models_status = {
            'models_directory_exists': models_dir.exists(),
            'available_models': [],
            'model_sizes': {},
            'recommended_model_present': False
        }
        
        if models_dir.exists():
            # Look for GGUF files
            gguf_files = list(models_dir.glob("*.gguf"))
            
            for model_file in gguf_files:
                model_info = {
                    'filename': model_file.name,
                    'size_mb': model_file.stat().st_size // (1024 * 1024),
                    'path': str(model_file)
                }
                models_status['available_models'].append(model_info)
                models_status['model_sizes'][model_file.name] = model_info['size_mb']
                
                # Check for recommended models
                if any(recommended in model_file.name.lower() for recommended in ['q4_k_m', 'q5_k_m']):
                    models_status['recommended_model_present'] = True
        
        self.status_report['models'] = models_status
        return models_status
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration files"""
        config_status = {
            'config_directory_exists': False,
            'main_config_exists': False,
            'config_valid': False,
            'config_content': None
        }
        
        # Check main config
        config_path = self.project_root / "src" / "config" / "llama2_brain_config.json"
        config_status['config_directory_exists'] = config_path.parent.exists()
        config_status['main_config_exists'] = config_path.exists()
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_content = json.load(f)
                config_status['config_valid'] = True
                config_status['config_content'] = config_content
            except Exception as e:
                config_status['config_valid'] = False
                config_status['config_error'] = str(e)
        
        self.status_report['configuration'] = config_status
        return config_status
    
    def check_components(self) -> Dict[str, Any]:
        """Check integration components"""
        components_status = {
            'core_integration': False,
            'consciousness_bridge': False,
            'training_pipeline': False,
            'setup_script': False,
            'runner_script': False,
            'demo_script': False
        }
        
        # Check core files
        core_files = {
            'core_integration': 'src/core/llama2_brain_integration.py',
            'consciousness_bridge': 'database/consciousness_agent/llama2_consciousness_bridge.py',
            'training_pipeline': 'src/training/llama2_brain_trainer.py',
            'setup_script': 'scripts/setup_llama2_integration.py',
            'runner_script': 'scripts/run_llama2_consciousness.py',
            'demo_script': 'examples/llama2_consciousness_demo.py'
        }
        
        for component, file_path in core_files.items():
            full_path = self.project_root / file_path
            components_status[component] = full_path.exists()
        
        self.status_report['components'] = components_status
        return components_status
    
    def test_integration(self) -> Dict[str, Any]:
        """Test integration functionality"""
        integration_status = {
            'llama_integration_import': False,
            'consciousness_bridge_import': False,
            'training_pipeline_import': False,
            'basic_functionality': False,
            'test_generation': False,
            'error_messages': []
        }
        
        # Test imports
        try:
            from core.llama2_brain_integration import create_llama_brain_integration
            integration_status['llama_integration_import'] = True
        except Exception as e:
            integration_status['error_messages'].append(f"Llama integration import: {e}")
        
        try:
            from consciousness_agent.llama2_consciousness_bridge import create_llama2_consciousness_bridge
            integration_status['consciousness_bridge_import'] = True
        except Exception as e:
            integration_status['error_messages'].append(f"Consciousness bridge import: {e}")
        
        try:
            from ml_architecture.training_pipelines.llama2_brain_trainer import BrainTrainingConfig
            integration_status['training_pipeline_import'] = True
        except Exception as e:
            integration_status['error_messages'].append(f"Training pipeline import: {e}")
        
        # Test basic functionality
        if integration_status['llama_integration_import']:
            try:
                # This will fail if no model is available, but tests the code path
                integration = create_llama_brain_integration("test_path")
                integration_status['basic_functionality'] = True
            except Exception as e:
                if "not found" in str(e).lower():
                    integration_status['basic_functionality'] = True  # Code works, just no model
                else:
                    integration_status['error_messages'].append(f"Basic functionality: {e}")
        
        self.status_report['integration'] = integration_status
        return integration_status
    
    def determine_overall_status(self) -> str:
        """Determine overall integration status"""
        # Required for basic functionality
        required_deps = ['llama-cpp-python', 'numpy']
        required_components = ['core_integration', 'setup_script', 'runner_script']
        
        deps = self.status_report['dependencies']
        components = self.status_report['components']
        models = self.status_report['models']
        integration = self.status_report['integration']
        
        # Check critical requirements
        deps_ok = all(deps.get(dep, False) for dep in required_deps)
        components_ok = all(components.get(comp, False) for comp in required_components)
        imports_ok = integration.get('llama_integration_import', False)
        
        if deps_ok and components_ok and imports_ok:
            if models.get('recommended_model_present', False):
                return 'fully_ready'
            elif models.get('available_models', []):
                return 'ready_with_model'
            else:
                return 'ready_needs_model'
        elif deps_ok and components_ok:
            return 'components_ready'
        elif components_ok:
            return 'needs_dependencies'
        else:
            return 'incomplete'
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        print("🔍 Checking Llama-2 Integration Status...")
        
        # Run all checks
        self.check_dependencies()
        self.check_models()
        self.check_configuration()
        self.check_components()
        self.test_integration()
        
        # Determine overall status
        self.status_report['overall_status'] = self.determine_overall_status()
        
        return self.status_report
    
    def print_status_report(self):
        """Print formatted status report"""
        report = self.generate_comprehensive_report()
        
        print("\n🧠🦙 Llama-2-7B-GGUF Brain Integration Status Report")
        print("=" * 70)
        
        # Overall status
        status = report['overall_status']
        status_icons = {
            'fully_ready': '🟢',
            'ready_with_model': '🟡',
            'ready_needs_model': '🟠',
            'components_ready': '🔵',
            'needs_dependencies': '🟡',
            'incomplete': '🔴'
        }
        
        status_messages = {
            'fully_ready': 'Fully Ready - All components and recommended model available',
            'ready_with_model': 'Ready - Model available, may not be optimal',
            'ready_needs_model': 'Ready - Need to download model',
            'components_ready': 'Components Ready - Need to install dependencies',
            'needs_dependencies': 'Needs Dependencies - Components available',
            'incomplete': 'Incomplete - Missing critical components'
        }
        
        print(f"\n{status_icons.get(status, '❓')} Overall Status: {status_messages.get(status, 'Unknown')}")
        
        # Dependencies
        print(f"\n📦 Dependencies:")
        deps = report['dependencies']
        for dep, available in deps.items():
            icon = "✅" if available else "❌"
            print(f"  {icon} {dep}")
        
        # Models
        print(f"\n🦙 Models:")
        models = report['models']
        print(f"  📁 Models directory: {'✅' if models['models_directory_exists'] else '❌'}")
        print(f"  🎯 Recommended model: {'✅' if models['recommended_model_present'] else '❌'}")
        
        if models['available_models']:
            print(f"  📋 Available models ({len(models['available_models'])}):")
            for model in models['available_models']:
                print(f"     • {model['filename']} ({model['size_mb']} MB)")
        else:
            print(f"  📋 Available models: None")
        
        # Configuration
        print(f"\n⚙️ Configuration:")
        config = report['configuration']
        print(f"  📁 Config directory: {'✅' if config['config_directory_exists'] else '❌'}")
        print(f"  📄 Main config file: {'✅' if config['main_config_exists'] else '❌'}")
        print(f"  ✔️ Config valid: {'✅' if config['config_valid'] else '❌'}")
        
        # Components
        print(f"\n🔧 Components:")
        components = report['components']
        component_names = {
            'core_integration': 'Core Llama Integration',
            'consciousness_bridge': 'Consciousness Bridge',
            'training_pipeline': 'Training Pipeline',
            'setup_script': 'Setup Script',
            'runner_script': 'Runner Script',
            'demo_script': 'Demo Script'
        }
        
        for comp, available in components.items():
            icon = "✅" if available else "❌"
            name = component_names.get(comp, comp)
            print(f"  {icon} {name}")
        
        # Integration tests
        print(f"\n🧪 Integration Tests:")
        integration = report['integration']
        test_names = {
            'llama_integration_import': 'Llama Integration Import',
            'consciousness_bridge_import': 'Consciousness Bridge Import',
            'training_pipeline_import': 'Training Pipeline Import',
            'basic_functionality': 'Basic Functionality'
        }
        
        for test, passed in integration.items():
            if test in test_names:
                icon = "✅" if passed else "❌"
                name = test_names[test]
                print(f"  {icon} {name}")
        
        if integration.get('error_messages'):
            print(f"\n⚠️ Errors:")
            for error in integration['error_messages']:
                print(f"  • {error}")
        
        # Recommendations
        print(f"\n💡 Next Steps:")
        self._print_recommendations(report)
    
    def _print_recommendations(self, report: Dict[str, Any]):
        """Print recommendations based on status"""
        status = report['overall_status']
        
        if status == 'fully_ready':
            print("  🎉 System fully ready! You can:")
            print("     • python scripts/run_llama2_consciousness.py")
            print("     • python examples/llama2_consciousness_demo.py")
            
        elif status == 'ready_with_model':
            print("  🚀 System ready with existing model! You can:")
            print("     • python scripts/run_llama2_consciousness.py")
            print("     • Consider downloading Q4_K_M or Q5_K_M for better quality")
            
        elif status == 'ready_needs_model':
            print("  📥 Download a model to complete setup:")
            print("     • python scripts/setup_llama2_integration.py")
            print("     • Recommended: Q4_K_M (4GB) or Q5_K_M (4.8GB)")
            
        elif status == 'components_ready':
            print("  📦 Install missing dependencies:")
            missing_deps = [dep for dep, avail in report['dependencies'].items() if not avail]
            print(f"     • pip install {' '.join(missing_deps)}")
            
        elif status == 'needs_dependencies':
            print("  📦 Install dependencies first:")
            print("     • pip install llama-cpp-python transformers torch")
            
        elif status == 'incomplete':
            print("  🔧 Complete the integration setup:")
            print("     • Ensure all project files are in place")
            print("     • Check project structure and paths")
        
        # Additional recommendations
        if not report['models']['recommended_model_present']:
            print("  🎯 For best performance, download a recommended model:")
            print("     • Q4_K_M: Good balance of size and quality")
            print("     • Q5_K_M: Higher quality, larger size")
        
        if not report['configuration']['config_valid']:
            print("  ⚙️ Create or fix configuration file:")
            print("     • Run setup script to generate default config")

def main():
    """Main status check function"""
    checker = Llama2StatusChecker()
    checker.print_status_report()

if __name__ == "__main__":
    main()
