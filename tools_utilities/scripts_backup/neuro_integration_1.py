#!/usr/bin/env python3
"""
Neuro Agent Integration
Integration layer between the command system and neuro agents.
"""

import os, sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add the neuro module to path
NEURO_PATH = Path(__file__).parent.parent.parent.parent / "multi_model_training" / "neuro"
if NEURO_PATH.exists():
    sys.path.insert(0, str(NEURO_PATH.parent))

class NeuroAgentConnector:
    """Connector to integrate with neuro agents for command discovery and execution."""
    
    def __init__(self):
        self.neuro_available = self._check_neuro_availability()
        self.neuro_commands = {}
        if self.neuro_available:
            self._discover_neuro_commands()
    
    def _check_neuro_availability(self) -> bool:
        """Check if neuro agents are available."""
        try:
            if NEURO_PATH.exists():
                # Try importing neuro CLI
                import neuro.cli
                return True
        except ImportError:
            pass
        
        # Check if neuro command is available in PATH
        try:
            result = subprocess.run(['python', '-m', 'neuro.cli', '--help'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _discover_neuro_commands(self):
        """Discover available neuro commands."""
        if not self.neuro_available:
            return
        
        try:
            # Import neuro modules dynamically
            from neuro import cli, scanners, analyzers, connectome, composer
            
            self.neuro_commands = {
                "scan": {
                    "module": "neuro.cli",
                    "function": "cmd_scan",
                    "description": "Scan files and build connectivity analysis",
                    "args": []
                },
                "analyze": {
                    "module": "neuro.cli", 
                    "function": "cmd_analyze",
                    "description": "Analyze files and extract metadata",
                    "args": ["--titles"]
                },
                "connectome": {
                    "module": "neuro.cli",
                    "function": "cmd_connectome", 
                    "description": "Build connectome and detect communities",
                    "args": []
                },
                "compose": {
                    "module": "neuro.cli",
                    "function": "cmd_compose",
                    "description": "Compose neural networks and agents",
                    "args": ["--learn"]
                },
                "organize": {
                    "module": "neuro.cli",
                    "function": "cmd_organize",
                    "description": "Organize files based on connectivity analysis", 
                    "args": ["--dry-run", "--execute"]
                }
            }
        except ImportError as e:
            print(f"Warning: Could not import neuro modules: {e}")
    
    def execute_neuro_command(self, command: str, args: List[str] = None) -> Tuple[bool, str, str]:
        """Execute a neuro command and return result."""
        if not self.neuro_available:
            return False, "", "Neuro agents not available"
        
        if command not in self.neuro_commands:
            return False, "", f"Unknown neuro command: {command}"
        
        try:
            # Build command line
            cmd_line = ['python', '-m', 'neuro.cli', command]
            if args:
                cmd_line.extend(args)
            
            # Execute command
            result = subprocess.run(
                cmd_line,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(NEURO_PATH.parent)
            )
            
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", f"Execution error: {str(e)}"
    
    def get_neuro_file_analysis(self, file_pattern: str = "*") -> Dict[str, Any]:
        """Get file analysis from neuro agents."""
        if not self.neuro_available:
            return {}
        
        try:
            # Use neuro scan to analyze files
            success, stdout, stderr = self.execute_neuro_command("scan")
            if success:
                # Parse output (assuming JSON format)
                try:
                    return json.loads(stdout)
                except json.JSONDecodeError:
                    return {"raw_output": stdout}
            else:
                return {"error": stderr}
        except Exception as e:
            return {"error": str(e)}
    
    def get_connectome_data(self) -> Dict[str, Any]:
        """Get connectome data from neuro agents."""
        if not self.neuro_available:
            return {}
        
        try:
            success, stdout, stderr = self.execute_neuro_command("connectome")
            if success:
                try:
                    return json.loads(stdout)
                except json.JSONDecodeError:
                    return {"raw_output": stdout}
            else:
                return {"error": stderr}
        except Exception as e:
            return {"error": str(e)}
    
    def suggest_organization(self) -> Dict[str, Any]:
        """Get file organization suggestions from neuro agents."""
        if not self.neuro_available:
            return {}
        
        try:
            success, stdout, stderr = self.execute_neuro_command("organize", ["--dry-run"])
            if success:
                try:
                    return json.loads(stdout)
                except json.JSONDecodeError:
                    return {"suggestions": stdout.split('\n')}
            else:
                return {"error": stderr}
        except Exception as e:
            return {"error": str(e)}
    
    def compose_agents(self, learn_mode: bool = False) -> Dict[str, Any]:
        """Compose neural agents using neuro system."""
        if not self.neuro_available:
            return {}
        
        try:
            args = ["--learn"] if learn_mode else []
            success, stdout, stderr = self.execute_neuro_command("compose", args)
            if success:
                try:
                    return json.loads(stdout)
                except json.JSONDecodeError:
                    return {"composition": stdout}
            else:
                return {"error": stderr}
        except Exception as e:
            return {"error": str(e)}
    
    def is_available(self) -> bool:
        """Check if neuro agents are available."""
        return self.neuro_available
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of neuro integration."""
        return {
            "available": self.neuro_available,
            "commands": list(self.neuro_commands.keys()) if self.neuro_available else [],
            "neuro_path": str(NEURO_PATH) if NEURO_PATH.exists() else None
        }

class SmartCommandDiscovery:
    """Smart command discovery using neuro agents."""
    
    def __init__(self, neuro_connector: NeuroAgentConnector):
        self.neuro = neuro_connector
    
    def discover_project_commands(self, project_path: str = ".") -> List[Dict[str, Any]]:
        """Discover commands specific to the current project using neuro analysis."""
        discovered_commands = []
        
        if not self.neuro.is_available():
            return self._fallback_discovery(project_path)
        
        # Use neuro agents to analyze project structure
        file_analysis = self.neuro.get_neuro_file_analysis()
        
        if "error" not in file_analysis:
            # Extract potential commands from file analysis
            discovered_commands.extend(self._extract_commands_from_analysis(file_analysis))
        
        # Get connectome data for relationship analysis
        connectome_data = self.neuro.get_connectome_data()
        if "error" not in connectome_data:
            discovered_commands.extend(self._extract_commands_from_connectome(connectome_data))
        
        return discovered_commands
    
    def _extract_commands_from_analysis(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract potential commands from neuro file analysis."""
        commands = []
        
        # Look for common patterns in analysis
        if "files" in analysis:
            for file_info in analysis["files"]:
                if isinstance(file_info, dict):
                    # Look for Python scripts with main functions
                    if file_info.get("type") == "python" and "main" in file_info.get("functions", []):
                        commands.append({
                            "name": f"run {Path(file_info['path']).stem}",
                            "executable": "python",
                            "args": [file_info["path"]],
                            "description": f"Execute {file_info['path']}",
                            "source": "neuro_analysis",
                            "category": "discovered"
                        })
                    
                    # Look for shell scripts
                    elif file_info.get("type") == "shell":
                        commands.append({
                            "name": f"run {Path(file_info['path']).stem}",
                            "executable": "bash",
                            "args": [file_info["path"]],
                            "description": f"Execute {file_info['path']}",
                            "source": "neuro_analysis",
                            "category": "discovered"
                        })
        
        return commands
    
    def _extract_commands_from_connectome(self, connectome: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract potential commands from connectome analysis."""
        commands = []
        
        # Look for highly connected nodes that might be command entry points
        if "nodes" in connectome:
            for node in connectome["nodes"]:
                if isinstance(node, dict) and node.get("centrality", 0) > 0.8:
                    file_path = node.get("path", "")
                    if file_path and (file_path.endswith(".py") or file_path.endswith(".sh")):
                        commands.append({
                            "name": f"execute {Path(file_path).stem}",
                            "executable": "python" if file_path.endswith(".py") else "bash",
                            "args": [file_path],
                            "description": f"High-centrality executable: {file_path}",
                            "source": "connectome_analysis",
                            "category": "discovered"
                        })
        
        return commands
    
    def _fallback_discovery(self, project_path: str) -> List[Dict[str, Any]]:
        """Fallback command discovery without neuro agents."""
        commands = []
        project_root = Path(project_path)
        
        # Look for common executable files
        for pattern in ["*.py", "*.sh", "*.bash"]:
            for file_path in project_root.rglob(pattern):
                if file_path.is_file():
                    # Skip hidden files and common non-executable patterns
                    if (not file_path.name.startswith('.') and 
                        not file_path.name.startswith('_') and
                        file_path.name not in ['setup.py', 'conftest.py']):
                        
                        commands.append({
                            "name": f"run {file_path.stem}",
                            "executable": "python" if file_path.suffix == ".py" else "bash",
                            "args": [str(file_path)],
                            "description": f"Execute {file_path.name}",
                            "source": "fallback_discovery",
                            "category": "discovered"
                        })
        
        return commands

def test_neuro_integration():
    """Test the neuro integration functionality."""
    print("🧠 Testing Neuro Agent Integration")
    print("=" * 50)
    
    connector = NeuroAgentConnector()
    status = connector.get_status()
    
    print(f"Neuro Available: {status['available']}")
    print(f"Available Commands: {status['commands']}")
    print(f"Neuro Path: {status['neuro_path']}")
    
    if connector.is_available():
        print("\n🔍 Testing command discovery...")
        discovery = SmartCommandDiscovery(connector)
        commands = discovery.discover_project_commands()
        
        print(f"Discovered {len(commands)} commands:")
        for cmd in commands[:5]:  # Show first 5
            print(f"  • {cmd['name']}: {cmd['description']}")
    
    else:
        print("\n⚠️  Neuro agents not available - using fallback discovery")

if __name__ == "__main__":
    test_neuro_integration()
