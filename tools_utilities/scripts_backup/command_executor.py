#!/usr/bin/env python3
"""
Command Execution Engine
Secure command execution with safety checks, monitoring, and help system.
"""

import os, sys
import subprocess
import shlex
import json
import time
import signal
import threading
import re
import shutil
from typing import Dict, List, Optional, Any, Callable, Tuple
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, asdict
from pathlib import Path
import logging
from datetime import datetime
import psutil
import tempfile
from contextlib import contextmanager

from .....................................................command_database import CommandDatabase, Command
from .....................................................natural_language_parser import NaturalLanguageParser, ParsedIntent

@dataclass
class ExecutionResult:
    """Result of command execution."""
    command_id: str
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    resource_usage: Dict[str, Any]
    start_time: str
    end_time: str
    working_directory: str
    environment_vars: Dict[str, str]
    process_id: Optional[int] = None
    warning_messages: List[str] = None

@dataclass
class ExecutionContext:
    """Context for command execution."""
    working_directory: str
    environment_vars: Dict[str, str]
    timeout: Optional[float]
    dry_run: bool
    safe_mode: bool
    interactive: bool
    resource_limits: Dict[str, Any]

class CommandExecutor:
    """Secure command execution engine with monitoring and safety checks."""
    
    def __init__(self, command_database: CommandDatabase):
        self.db = command_database
        self.parser = NaturalLanguageParser()
        self.logger = logging.getLogger(__name__)
        self.active_processes = {}
        self.execution_history = []
        self.safety_checks_enabled = True
        self.resource_monitor = ResourceMonitor()
        
        # Default execution context
        self.default_context = ExecutionContext(
            working_directory=str(Path.cwd()),
            environment_vars=dict(os.environ),
            timeout=300.0,  # 5 minutes default
            dry_run=False,
            safe_mode=True,
            interactive=False,
            resource_limits={
                "max_memory_mb": 4096,
                "max_cpu_percent": 80,
                "max_execution_time": 600
            }
        )
    
    def execute_natural_language(self, user_input: str, context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """Execute command from natural language input."""
        if context is None:
            context = self.default_context
        
        # Parse user input
        parsed_intent = self.parser.parse_user_input(user_input, self.db)
        
        # Handle different intent types
        if parsed_intent.intent_type == "help":
            return self._handle_help_request(user_input, parsed_intent)
        
        if not parsed_intent.command_ids:
            return self._handle_no_matches(user_input, parsed_intent)
        
        if len(parsed_intent.command_ids) > 1:
            return self._handle_multiple_matches(user_input, parsed_intent)
        
        # Execute single command
        command_id = parsed_intent.command_ids[0]
        command = self.db.get_command(command_id)
        
        if not command:
            return ExecutionResult(
                command_id=command_id,
                success=False,
                return_code=1,
                stdout="",
                stderr=f"Command {command_id} not found",
                execution_time=0.0,
                resource_usage={},
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                working_directory=context.working_directory,
                environment_vars={}
            )
        
        # Safety check
        if parsed_intent.requires_confirmation and context.safe_mode:
            return self._request_confirmation(command, parsed_intent, context)
        
        # Execute command
        return self.execute_command(command, parsed_intent.parameters, context)
    
    def execute_command(self, command: Command, parameters: Dict[str, Any], context: ExecutionContext) -> ExecutionResult:
        """Execute a specific command with parameters."""
        start_time = datetime.now()
        
        # Build command line
        cmd_line = self._build_command_line(command, parameters)
        
        self.logger.info(f"Executing command: {command.name}")
        self.logger.debug(f"Command line: {cmd_line}")
        
        # Dry run mode
        if context.dry_run:
            return ExecutionResult(
                command_id=command.id,
                success=True,
                return_code=0,
                stdout=f"[DRY RUN] Would execute: {' '.join(cmd_line)}",
                stderr="",
                execution_time=0.0,
                resource_usage={},
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                working_directory=context.working_directory,
                environment_vars=context.environment_vars
            )
        
        # Safety checks
        if self.safety_checks_enabled and context.safe_mode:
            safety_result = self._perform_safety_checks(command, cmd_line, context)
            if not safety_result[0]:
                return ExecutionResult(
                    command_id=command.id,
                    success=False,
                    return_code=1,
                    stdout="",
                    stderr=f"Safety check failed: {safety_result[1]}",
                    execution_time=0.0,
                    resource_usage={},
                    start_time=start_time.isoformat(),
                    end_time=datetime.now().isoformat(),
                    working_directory=context.working_directory,
                    environment_vars=context.environment_vars
                )
        
        # Execute command
        try:
            with self.resource_monitor.monitor_execution(context.resource_limits) as monitor:
                result = self._execute_subprocess(cmd_line, context, command.id)
                result.resource_usage = monitor.get_stats()
                
                # Log execution
                self.db.log_usage(
                    command.id,
                    ' '.join(cmd_line),
                    result.success,
                    result.execution_time
                )
                
                self.execution_history.append(result)
                return result
                
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return ExecutionResult(
                command_id=command.id,
                success=False,
                return_code=1,
                stdout="",
                stderr=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
                resource_usage={},
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                working_directory=context.working_directory,
                environment_vars=context.environment_vars
            )
    
    def _build_command_line(self, command: Command, parameters: Dict[str, Any]) -> List[str]:
        """Build command line from command definition and parameters."""
        cmd_line = [command.executable] + command.args.copy()
        
        # Add flags from parameters
        for flag, description in command.flags.items():
            param_name = flag.lstrip('-').replace('-', '_')
            if param_name in parameters:
                if isinstance(parameters[param_name], bool) and parameters[param_name]:
                    cmd_line.append(flag)
                elif not isinstance(parameters[param_name], bool):
                    cmd_line.extend([flag, str(parameters[param_name])])
        
        # Add positional arguments
        if "args" in parameters:
            if isinstance(parameters["args"], list):
                cmd_line.extend(parameters["args"])
            else:
                cmd_line.append(str(parameters["args"]))
        
        return cmd_line
    
    def _execute_subprocess(self, cmd_line: List[str], context: ExecutionContext, command_id: str) -> ExecutionResult:
        """Execute subprocess with monitoring."""
        start_time = datetime.now()
        
        # Set up environment
        env = context.environment_vars.copy()
        
        # Create process
        try:
            process = subprocess.Popen(
                cmd_line,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=context.working_directory,
                env=env,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Track active process
            self.active_processes[command_id] = process
            
            # Execute with timeout
            try:
                stdout, stderr = process.communicate(timeout=context.timeout)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                # Kill process tree
                if os.name != 'nt':
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                
                stdout, stderr = process.communicate()
                return_code = -1
                stderr += "\n[TIMEOUT] Command execution exceeded timeout limit"
            
            finally:
                # Remove from active processes
                self.active_processes.pop(command_id, None)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                command_id=command_id,
                success=(return_code == 0),
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                resource_usage={},  # Will be filled by monitor
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                working_directory=context.working_directory,
                environment_vars=env,
                process_id=process.pid
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ExecutionResult(
                command_id=command_id,
                success=False,
                return_code=1,
                stdout="",
                stderr=f"Process execution error: {str(e)}",
                execution_time=execution_time,
                resource_usage={},
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                working_directory=context.working_directory,
                environment_vars=env
            )
    
    def _perform_safety_checks(self, command: Command, cmd_line: List[str], context: ExecutionContext) -> Tuple[bool, str]:
        """Perform safety checks before execution."""
        # Check for dangerous operations
        dangerous_patterns = [
            r'\brm\s+-rf\s+/',
            r'\bdd\s+if=',
            r'\bmkfs\.',
            r'\bfdisk\b',
            r'\bparted\b',
            r':\(\)\{\s*:\|:\&\s*\};:',  # Fork bomb
            r'\bchmod\s+777',
            r'\bsudo\s+.*\brm\b'
        ]
        
        cmd_str = ' '.join(cmd_line)
        for pattern in dangerous_patterns:
            if re.search(pattern, cmd_str):
                return False, f"Dangerous pattern detected: {pattern}"
        
        # Check file system permissions
        if context.working_directory:
            if not os.access(context.working_directory, os.W_OK):
                return False, f"No write permission in {context.working_directory}"
        
        # Check resource limits
        current_memory = psutil.virtual_memory().percent
        if current_memory > 90:
            return False, "System memory usage too high"
        
        # Check if executable exists
        if command.executable and not self._check_executable_exists(command.executable):
            return False, f"Executable not found: {command.executable}"
        
        return True, ""
    
    def _check_executable_exists(self, executable: str) -> bool:
        """Check if executable exists in PATH."""
        return shutil.which(executable) is not None
    
    def _handle_help_request(self, user_input: str, parsed_intent: ParsedIntent) -> ExecutionResult:
        """Handle help requests."""
        help_content = self.generate_help_content(user_input)
        
        return ExecutionResult(
            command_id="help",
            success=True,
            return_code=0,
            stdout=help_content,
            stderr="",
            execution_time=0.0,
            resource_usage={},
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            working_directory="",
            environment_vars={}
        )
    
    def _handle_no_matches(self, user_input: str, parsed_intent: ParsedIntent) -> ExecutionResult:
        """Handle cases where no commands match."""
        suggestions = self.parser.suggest_similar_commands(user_input, self.db)
        
        response = f"❌ No exact matches found for '{user_input}'\n\n"
        if suggestions:
            response += "💡 Similar commands:\n"
            for suggestion in suggestions:
                response += f"   {suggestion}\n"
        else:
            response += "💡 Try 'help' to see all available commands"
        
        return ExecutionResult(
            command_id="no_match",
            success=False,
            return_code=1,
            stdout=response,
            stderr="",
            execution_time=0.0,
            resource_usage={},
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            working_directory="",
            environment_vars={}
        )
    
    def _handle_multiple_matches(self, user_input: str, parsed_intent: ParsedIntent) -> ExecutionResult:
        """Handle cases with multiple command matches."""
        prompt = self.parser.generate_disambiguation_prompt(parsed_intent, self.db)
        
        return ExecutionResult(
            command_id="multiple_matches",
            success=False,
            return_code=1,
            stdout=prompt,
            stderr="",
            execution_time=0.0,
            resource_usage={},
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            working_directory="",
            environment_vars={}
        )
    
    def _request_confirmation(self, command: Command, parsed_intent: ParsedIntent, context: ExecutionContext) -> ExecutionResult:
        """Request user confirmation for potentially dangerous commands."""
        warning_msg = f"""
⚠️  SAFETY WARNING ⚠️

Command: {command.name}
Description: {command.description}
Warning: {parsed_intent.safety_warning}

This command requires confirmation because:
• Requires shell access: {command.requires_shell}
• Requires sudo: {command.requires_sudo}
• Safe mode: {command.safe_mode}

To proceed, add --confirm-dangerous to your command or set safe_mode=False in context.
"""
        
        return ExecutionResult(
            command_id=command.id,
            success=False,
            return_code=1,
            stdout=warning_msg,
            stderr="",
            execution_time=0.0,
            resource_usage={},
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            working_directory=context.working_directory,
            environment_vars={}
        )
    
    def generate_help_content(self, query: str = "") -> str:
        """Generate help content based on query."""
        if not query or query.lower() in ["help", "?"]:
            return self._generate_general_help()
        
        # Search for specific help
        commands = self.db.search_commands(query)
        if commands:
            return self._generate_command_help(commands)
        
        # Category help
        categories = self.db.get_categories()
        for cat in categories:
            if query.lower() in cat.name.lower():
                return self._generate_category_help(cat.number)
        
        return f"No help found for '{query}'. Try 'help' for general help."
    
    def _generate_general_help(self) -> str:
        """Generate general help content."""
        categories = self.db.get_categories()
        main_categories = [cat for cat in categories if cat.parent is None]
        
        help_content = """
🧠 SmallMind Command System Help

Available Categories:
"""
        
        for cat in main_categories:
            commands = self.db.get_commands_by_category(cat.number)
            help_content += f"\n{cat.number}. {cat.name} ({len(commands)} commands)\n"
            help_content += f"   {cat.description}\n"
        
        help_content += """

Usage Examples:
• "train a neural network" - Natural language command
• "help neural" - Get help for neural commands  
• "help 1.1" - Get help for category 1.1
• "{number}" - Execute command by number

For detailed help on any command, type: help {command_name}
"""
        
        return help_content
    
    def _generate_command_help(self, commands: List[Command]) -> str:
        """Generate help for specific commands."""
        if not commands:
            return "No commands found."
        
        help_content = f"📚 Command Help ({len(commands)} commands found)\n\n"
        
        for cmd in commands[:10]:  # Limit to 10 commands
            help_content += f"{cmd.number}. {cmd.name}\n"
            help_content += f"   Description: {cmd.description}\n"
            help_content += f"   Category: {cmd.category}\n"
            help_content += f"   Complexity: {cmd.complexity}\n"
            
            if cmd.flags:
                help_content += "   Flags:\n"
                for flag, desc in cmd.flags.items():
                    help_content += f"     {flag}: {desc}\n"
            
            if cmd.examples:
                help_content += f"   Example: {cmd.examples[0]}\n"
            
            if cmd.requires_shell or cmd.requires_sudo:
                help_content += "   ⚠️  Requires: "
                if cmd.requires_shell:
                    help_content += "shell "
                if cmd.requires_sudo:
                    help_content += "sudo "
                help_content += "\n"
            
            help_content += "\n"
        
        return help_content
    
    def _generate_category_help(self, category_number: str) -> str:
        """Generate help for a specific category."""
        commands = self.db.get_commands_by_category(category_number)
        
        if not commands:
            return f"No commands found in category {category_number}"
        
        category = None
        categories = self.db.get_categories()
        for cat in categories:
            if cat.number == category_number:
                category = cat
                break
        
        help_content = f"📁 Category {category_number}"
        if category:
            help_content += f": {category.name}\n{category.description}\n"
        help_content += f"\nCommands ({len(commands)}):\n\n"
        
        for cmd in commands:
            help_content += f"{cmd.number}. {cmd.name}\n"
            help_content += f"   {cmd.description}\n"
            if cmd.examples:
                help_content += f"   Example: {cmd.examples[0]}\n"
            help_content += "\n"
        
        return help_content
    
    def stop_command(self, command_id: str) -> bool:
        """Stop a running command."""
        if command_id in self.active_processes:
            process = self.active_processes[command_id]
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                
                process.wait(timeout=5)
                return True
            except Exception as e:
                self.logger.error(f"Failed to stop command {command_id}: {e}")
                return False
        
        return False
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {
            "active_processes": len(self.active_processes),
            "execution_history_count": len(self.execution_history),
            "safety_checks_enabled": self.safety_checks_enabled,
            "resource_monitor_active": self.resource_monitor.is_active()
        }

class ResourceMonitor:
    """Monitor resource usage during command execution."""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {}
        self.start_time = None
        self.monitor_thread = None
        
    @contextmanager
    def monitor_execution(self, limits: Dict[str, Any]):
        """Context manager for monitoring execution."""
        self.start_monitoring(limits)
        try:
            yield self
        finally:
            self.stop_monitoring()
    
    def start_monitoring(self, limits: Dict[str, Any]):
        """Start resource monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.stats = {
            "peak_memory_mb": 0,
            "peak_cpu_percent": 0,
            "total_execution_time": 0,
            "memory_samples": [],
            "cpu_samples": []
        }
        
        def monitor():
            while self.monitoring:
                try:
                    memory_mb = psutil.virtual_memory().used / 1024 / 1024
                    cpu_percent = psutil.cpu_percent()
                    
                    self.stats["peak_memory_mb"] = max(self.stats["peak_memory_mb"], memory_mb)
                    self.stats["peak_cpu_percent"] = max(self.stats["peak_cpu_percent"], cpu_percent)
                    
                    self.stats["memory_samples"].append(memory_mb)
                    self.stats["cpu_samples"].append(cpu_percent)
                    
                    time.sleep(0.5)
                except Exception:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.start_time:
            self.stats["total_execution_time"] = time.time() - self.start_time
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        return self.stats.copy()
    
    def is_active(self) -> bool:
        """Check if monitoring is active."""
        return self.monitoring

if __name__ == "__main__":
    # Test the executor
    from command_database import CommandDatabase
    
    db = CommandDatabase()
    executor = CommandExecutor(db)
    
    test_commands = [
        "help",
        "show me neural commands",
        "list available models",
        "train a model with 100 epochs"
    ]
    
    print("⚡ Command Executor Test")
    print("=" * 50)
    
    for cmd in test_commands:
        print(f"\nExecuting: '{cmd}'")
        result = executor.execute_natural_language(cmd)
        
        print(f"Success: {result.success}")
        print(f"Output: {result.stdout[:200]}...")
        if result.stderr:
            print(f"Errors: {result.stderr[:100]}...")
    
    db.close()
