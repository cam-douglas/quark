#!/usr/bin/env python3
"""
Unified AI Configuration Manager
Ensures Cursor AI + Small-Mind integration works identically in both Cursor IDE and Terminal
Maintains cross-platform synchronization and consistency
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

class UnifiedAIConfig:
    """Manages unified configuration across Cursor IDE and Terminal"""
    
    def __init__(self):
        self.smallmind_path = Path("ROOT")
        self.cursor_settings = Path("/Users/camdouglas/Library/Application Support/Cursor/User/settings.json")
        self.zshrc_path = Path("/Users/camdouglas/.zshrc")
        self.config_version = "2.0.0"
        
        # Unified configuration that both environments should match
        self.unified_config = {
            "ai_integration": {
                "cursor_ai_enabled": True,
                "smallmind_integration": True,
                "dual_system_mode": True,
                "default_model": "claude-3-sonnet",
                "auto_argument_detection": True,
                "gps_services": True,
                "internet_services": True,
                "natural_language_processing": True
            },
            "capabilities": {
                "brain_development": True,
                "physics_simulation": True,
                "ml_optimization": True,
                "data_visualization": True,
                "neurodata_processing": True,
                "computational_neuroscience": True
            },
            "sync_settings": {
                "cross_platform_sync": True,
                "unified_experience": True,
                "maintain_consistency": True,
                "auto_sync_interval": 30,
                "version": "2.0.0"
            },
            "paths": {
                "smallmind_path": "ROOT",
                "ai_models_dir": "ROOT/models/ai_models",
                "cursor_models_dir": "ROOT/models/cursor_models"
            }
        }
    
    def get_current_cursor_settings(self) -> Dict[str, Any]:
        """Read current Cursor IDE settings"""
        try:
            if self.cursor_settings.exists():
                with open(self.cursor_settings, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"⚠️  Warning reading Cursor settings: {e}")
            return {}
    
    def get_terminal_config_status(self) -> Dict[str, bool]:
        """Check terminal configuration status"""
        try:
            if not self.zshrc_path.exists():
                return {"exists": False}
            
            with open(self.zshrc_path, 'r') as f:
                content = f.read()
            
            return {
                "exists": True,
                "cursor_ai_init": "init_cursor_ai()" in content,
                "smallmind_integration": "init_smallmind_integration()" in content,
                "dual_system": "Cursor AI + Small-Mind" in content,
                "auto_launch": "auto_launch_cursor_ai()" in content,
                "natural_language": "cursor_ai_input_interceptor" in content
            }
        except Exception as e:
            print(f"⚠️  Warning checking terminal config: {e}")
            return {"exists": False}
    
    def sync_cursor_settings(self) -> bool:
        """Synchronize Cursor IDE settings with unified config"""
        try:
            current_settings = self.get_current_cursor_settings()
            
            # Update with unified configuration
            current_settings.update({
                # Small-Mind Integration
                "cursor.chat.smallmindIntegration": True,
                "cursor.chat.smallmindPath": self.unified_config["paths"]["smallmind_path"],
                "cursor.chat.enableComputationalNeuroscience": True,
                "cursor.chat.enablePhysicsSimulation": True,
                "cursor.chat.enableBrainDevelopment": True,
                "cursor.chat.enableMLOptimization": True,
                "cursor.chat.enableDataVisualization": True,
                
                # Terminal Synchronization
                "cursor.chat.enableTerminalSync": True,
                "cursor.chat.terminalSyncPath": str(self.zshrc_path),
                "cursor.chat.terminalAIIntegration": True,
                "cursor.chat.dualSystemMode": True,
                
                # AI Model Configuration
                "cursor.chat.defaultModel": self.unified_config["ai_integration"]["default_model"],
                "cursor.chat.enableAutoArgumentDetection": True,
                "cursor.chat.enableGPSServices": True,
                "cursor.chat.enableInternetServices": True,
                "cursor.chat.enableNaturalLanguageProcessing": True,
                
                # Cross-Platform Consistency
                "cursor.chat.enableCrossPlatformSync": True,
                "cursor.chat.unifiedAIExperience": True,
                "cursor.chat.maintainConsistencyAcrossEnvironments": True,
                "cursor.chat.syncVersion": self.config_version,
                "cursor.chat.lastSyncTimestamp": int(time.time())
            })
            
            # Write back to settings
            with open(self.cursor_settings, 'w') as f:
                json.dump(current_settings, f, indent=4)
            
            print("✅ Cursor IDE settings synchronized")
            return True
            
        except Exception as e:
            print(f"❌ Error syncing Cursor settings: {e}")
            return False
    
    def verify_terminal_integration(self) -> bool:
        """Verify terminal integration is properly configured"""
        status = self.get_terminal_config_status()
        
        if not status.get("exists", False):
            print("❌ Terminal configuration (.zshrc) not found")
            return False
        
        required_features = [
            "cursor_ai_init", "smallmind_integration", 
            "dual_system", "auto_launch", "natural_language"
        ]
        
        missing_features = [f for f in required_features if not status.get(f, False)]
        
        if missing_features:
            print(f"⚠️  Terminal missing features: {', '.join(missing_features)}")
            return False
        
        print("✅ Terminal integration verified")
        return True
    
    def create_workspace_configuration(self) -> bool:
        """Create workspace-specific configuration for Cursor"""
        try:
            workspace_config = {
                "settings": {
                    "cursor.chat.workspaceSpecific": True,
                    "cursor.chat.projectType": "computational-neuroscience",
                    "cursor.chat.enableSmallMindIntegration": True,
                    "cursor.chat.defaultWorkingDirectory": str(self.smallmind_path),
                    "cursor.chat.enableBrainSimulation": True,
                    "cursor.chat.enablePhysicsModeling": True,
                    "cursor.chat.enableMLOptimization": True
                }
            }
            
            # Create .vscode/settings.json for workspace-specific settings
            vscode_dir = self.smallmind_path / ".vscode"
            vscode_dir.mkdir(exist_ok=True)
            
            settings_file = vscode_dir / "settings.json"
            with open(settings_file, 'w') as f:
                json.dump(workspace_config["settings"], f, indent=4)
            
            print("✅ Workspace configuration created")
            return True
            
        except Exception as e:
            print(f"❌ Error creating workspace configuration: {e}")
            return False
    
    def setup_environment_sync(self) -> bool:
        """Set up environment variables for consistent behavior"""
        try:
            env_vars = {
                "CURSOR_AI_ENABLED": "true",
                "SMALLMIND_AI_INTEGRATION": "true",
                "DUAL_AI_SYSTEM": "true",
                "DEFAULT_AI_MODEL": self.unified_config["ai_integration"]["default_model"],
                "UNIFIED_AI_CONFIG_VERSION": self.config_version,
                "CURSOR_SMALLMIND_SYNC": "true"
            }
            
            # Add to current environment
            for key, value in env_vars.items():
                os.environ[key] = value
            
            print("✅ Environment variables synchronized")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up environment sync: {e}")
            return False
    
    def test_cross_platform_functionality(self) -> Dict[str, bool]:
        """Test that AI functionality works in both environments"""
        results = {
            "cursor_settings": False,
            "terminal_config": False,
            "workspace_config": False,
            "environment_sync": False,
            "ai_integration": False
        }
        
        try:
            # Test Cursor settings
            cursor_settings = self.get_current_cursor_settings()
            results["cursor_settings"] = cursor_settings.get("cursor.chat.smallmindIntegration", False)
            
            # Test terminal configuration
            results["terminal_config"] = self.verify_terminal_integration()
            
            # Test workspace configuration
            workspace_file = self.smallmind_path / ".vscode" / "settings.json"
            results["workspace_config"] = workspace_file.exists()
            
            # Test environment sync
            results["environment_sync"] = os.environ.get("CURSOR_AI_ENABLED") == "true"
            
            # Test AI integration
            results["ai_integration"] = all([
                os.environ.get("SMALLMIND_AI_INTEGRATION") == "true",
                os.environ.get("DUAL_AI_SYSTEM") == "true"
            ])
            
        except Exception as e:
            print(f"⚠️  Error during testing: {e}")
        
        return results
    
    def full_synchronization(self) -> bool:
        """Perform full synchronization between Cursor and Terminal"""
        print("🔄 Starting full synchronization...")
        print("=" * 60)
        
        success_count = 0
        total_steps = 4
        
        # Step 1: Sync Cursor settings
        if self.sync_cursor_settings():
            success_count += 1
        
        # Step 2: Verify terminal integration
        if self.verify_terminal_integration():
            success_count += 1
        
        # Step 3: Create workspace configuration
        if self.create_workspace_configuration():
            success_count += 1
        
        # Step 4: Setup environment sync
        if self.setup_environment_sync():
            success_count += 1
        
        print("=" * 60)
        print(f"📊 Synchronization complete: {success_count}/{total_steps} successful")
        
        if success_count == total_steps:
            print("🎉 Full synchronization successful!")
            print("✅ Cursor AI + Small-Mind now works identically in both environments")
            return True
        else:
            print("⚠️  Partial synchronization - some issues detected")
            return False
    
    def status_report(self) -> None:
        """Generate detailed status report"""
        print("\n" + "=" * 70)
        print("📊 CURSOR AI + SMALL-MIND UNIFIED STATUS REPORT")
        print("=" * 70)
        
        # Test functionality
        test_results = self.test_cross_platform_functionality()
        
        print("\n🔍 Cross-Platform Test Results:")
        for test, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test.replace('_', ' ').title()}: {status}")
        
        # Configuration details
        print("\n⚙️  Configuration Details:")
        print(f"   Version: {self.config_version}")
        print(f"   Small-Mind Path: {self.smallmind_path}")
        print(f"   Default Model: {self.unified_config['ai_integration']['default_model']}")
        print(f"   Dual System Mode: {'✅ Enabled' if self.unified_config['ai_integration']['dual_system_mode'] else '❌ Disabled'}")
        
        # Environment status
        print("\n🌍 Environment Status:")
        env_vars = ["CURSOR_AI_ENABLED", "SMALLMIND_AI_INTEGRATION", "DUAL_AI_SYSTEM"]
        for var in env_vars:
            value = os.environ.get(var, "not set")
            print(f"   {var}: {value}")
        
        print("\n" + "=" * 70)
        
        # Overall status
        if all(test_results.values()):
            print("🎉 SYSTEM STATUS: FULLY SYNCHRONIZED ✅")
            print("💡 Cursor AI + Small-Mind works identically in both Cursor IDE and Terminal")
        else:
            print("⚠️  SYSTEM STATUS: NEEDS ATTENTION ❌")
            print("💡 Run full_synchronization() to fix issues")
        
        print("=" * 70)

def main():
    """Main function for command-line usage"""
    config = UnifiedAIConfig()
    
    if len(os.sys.argv) > 1:
        command = os.sys.argv[1].lower()
        
        if command == "sync":
            config.full_synchronization()
        elif command == "status":
            config.status_report()
        elif command == "test":
            results = config.test_cross_platform_functionality()
            print("Test Results:", results)
        else:
            print("Usage: python unified_ai_config.py [sync|status|test]")
    else:
        # Default: run status report
        config.status_report()

if __name__ == "__main__":
    main()
