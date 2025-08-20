#!/usr/bin/env python3
"""
Small-Mind Shell Integration
Replaces terminal prompt with natural language interface that feeds into small-mind models.
"""

import os, sys
import subprocess
import readline
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmallMindShell:
    """Natural language terminal interface for Small-Mind."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.config_file = self.project_root / "shell_config.json"
        self.history_file = Path.home() / ".smallmind_history"
        self.model_manager = None
        self.environment_ready = False
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize readline for history
        self.setup_readline()
        
        # Ensure environment is ready
        self.prepare_environment()
    
    def load_config(self) -> Dict[str, Any]:
        """Load shell configuration."""
        default_config = {
            "prompt": "🤖 small-mind> ",
            "model": "default",
            "auto_environment_setup": True,
            "max_history": 1000,
            "enable_autocomplete": True,
            "default_actions": {
                "help": "Show available commands and help",
                "status": "Show system status",
                "clear": "Clear terminal",
                "exit": "Exit small-mind shell"
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        
        return default_config
    
    def setup_readline(self):
        """Setup readline for command history and autocomplete."""
        try:
            readline.set_history_length(self.config["max_history"])
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
            
            if self.config["enable_autocomplete"]:
                readline.parse_and_bind("tab: complete")
                readline.set_completer(self.completer)
        except Exception as e:
            logger.warning(f"Readline setup failed: {e}")
    
    def completer(self, text: str, state: int) -> Optional[str]:
        """Command completion function."""
        commands = list(self.config["default_actions"].keys()) + [
            "run", "simulate", "analyze", "train", "optimize", "deploy"
        ]
        
        matches = [cmd for cmd in commands if cmd.startswith(text)]
        return matches[state] if state < len(matches) else None
    
    def prepare_environment(self):
        """Ensure the Python environment is ready with all dependencies."""
        try:
            # Check if we're in the right environment
            if not self.environment_ready:
                logger.info("Preparing Small-Mind environment...")
                
                # Add project root to Python path
                if str(self.project_root) not in sys.path:
                    sys.path.insert(0, str(self.project_root))
                
                # Check and install dependencies if needed
                self.check_dependencies()
                
                # Initialize model manager
                self.initialize_models()
                
                self.environment_ready = True
                logger.info("Environment ready!")
                
        except Exception as e:
            logger.error(f"Environment preparation failed: {e}")
            self.environment_ready = False
    
    def check_dependencies(self):
        """Check and install required dependencies."""
        required_packages = [
            "numpy", "torch", "transformers", "requests", "tqdm"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.info(f"Installing missing packages: {missing_packages}")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install"
                ] + missing_packages)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install packages: {e}")
                raise
    
    def initialize_models(self):
        """Initialize the model manager."""
        try:
            # Import model manager
            from src.smallmind.models.model_manager import ModelManager
            self.model_manager = ModelManager()
            logger.info("Model manager initialized")
        except Exception as e:
            logger.warning(f"Could not initialize model manager: {e}")
            self.model_manager = None
    
    def process_natural_language(self, user_input: str) -> str:
        """Process natural language input and return response."""
        try:
            # Simple command parsing for now
            if user_input.lower().startswith(("help", "?")):
                return self.show_help()
            elif user_input.lower().startswith("status"):
                return self.show_status()
            elif user_input.lower().startswith("clear"):
                os.system("clear")
                return "Terminal cleared."
            elif user_input.lower().startswith("exit"):
                return "exit"
            elif user_input.lower().startswith("run "):
                return self.execute_command(user_input[4:])
            elif user_input.lower().startswith("simulate "):
                return self.run_simulation(user_input[9:])
            elif user_input.lower().startswith("analyze "):
                return self.analyze_data(user_input[8:])
            else:
                # Use AI model for general queries
                return self.ai_response(user_input)
                
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return f"Error: {str(e)}"
    
    def show_help(self) -> str:
        """Show available commands and help."""
        help_text = """
🤖 Small-Mind Terminal Interface

Available Commands:
"""
        for cmd, desc in self.config["default_actions"].items():
            help_text += f"  {cmd:<10} - {desc}\n"
        
        help_text += """
Special Commands:
  run <command>   - Execute a system command
  simulate <task> - Run brain development simulation
  analyze <data>  - Analyze data or models
  help            - Show this help message
  status          - Show system status
  clear           - Clear terminal
  exit            - Exit small-mind shell

Natural Language:
  You can also type natural language queries and the AI will respond!
"""
        return help_text
    
    def show_status(self) -> str:
        """Show system status."""
        status = f"""
🤖 Small-Mind Status

Environment: {'✅ Ready' if self.environment_ready else '❌ Not Ready'}
Model Manager: {'✅ Active' if self.model_manager else '❌ Not Available'}
Project Root: {self.project_root}
Python Path: {sys.executable}
Working Directory: {os.getcwd()}
"""
        return status
    
    def execute_command(self, command: str) -> str:
        """Execute a system command."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return f"Command executed successfully:\n{result.stdout}"
            else:
                return f"Command failed:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds."
        except Exception as e:
            return f"Error executing command: {e}"
    
    def run_simulation(self, task: str) -> str:
        """Run brain development simulation."""
        try:
            # This would integrate with your simulation modules
            return f"Running simulation for: {task}\n(Simulation functionality to be implemented)"
        except Exception as e:
            return f"Simulation error: {e}"
    
    def analyze_data(self, data_desc: str) -> str:
        """Analyze data or models."""
        try:
            # This would integrate with your analysis modules
            return f"Analyzing: {data_desc}\n(Analysis functionality to be implemented)"
        except Exception as e:
            return f"Analysis error: {e}"
    
    def ai_response(self, query: str) -> str:
        """Generate AI response for general queries."""
        try:
            if self.model_manager:
                # Use the model manager for AI responses
                return f"AI Response to '{query}':\n(To be implemented with your models)"
            else:
                return f"I understand you're asking: '{query}'\n(AI models not available - check environment setup)"
        except Exception as e:
            return f"AI response error: {e}"
    
    def run(self):
        """Main shell loop."""
        print("🤖 Welcome to Small-Mind Terminal Interface!")
        print("Type 'help' for available commands or ask questions in natural language.")
        print("Type 'exit' to quit.\n")
        
        while True:
            try:
                # Get user input
                user_input = input(self.config["prompt"]).strip()
                
                if not user_input:
                    continue
                
                # Save to history
                readline.add_history(user_input)
                
                # Process input
                response = self.process_natural_language(user_input)
                
                if response == "exit":
                    print("Goodbye! 👋")
                    break
                
                # Display response
                print(f"\n{response}\n")
                
            except KeyboardInterrupt:
                print("\n\nUse 'exit' to quit the shell.")
            except EOFError:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"Error: {e}")
        
        # Save history
        try:
            readline.write_history_file(str(self.history_file))
        except Exception as e:
            logger.warning(f"Could not save history: {e}")

def main():
    """Main entry point."""
    shell = SmallMindShell()
    shell.run()

if __name__ == "__main__":
    main()
