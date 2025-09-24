"""
Command Router Module
=====================
Routes commands to appropriate subsystems.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class CommandRouter:
    """Routes commands to appropriate Quark subsystems."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
        # Subsystem executables
        todo_core = self.project_root / 'state' / 'todo' / 'core'
        self.systems = {
            'task': todo_core / 'tasks_launcher.py',
            'validate': todo_core / 'validate_launcher.py',
            'state': todo_core / 'state_system_launcher.py',
            'test': 'pytest',  # System command
            'git': 'git',  # System command
        }
    
    def route_command(self, system: str, action: str, params: Dict) -> int:
        """
        Route command to appropriate subsystem.
        Returns exit code.
        """
        # Import handlers lazily to avoid circular imports
        if system == 'task':
            return self._route_task_command(action, params)
        elif system == 'validate':
            return self._route_validation_command(action, params)
        elif system == 'brain':
            from .brain_handler import BrainHandler
            handler = BrainHandler(self.project_root)
            return handler.route_command(action, params)
        elif system == 'simulation':
            from .simulation_handler import SimulationHandler
            handler = SimulationHandler(self.project_root)
            return handler.route_command(action, params)
        elif system == 'training':
            from .training_handler import TrainingHandler
            handler = TrainingHandler(self.project_root)
            return handler.route_command(action, params)
        elif system == 'deployment':
            from .deployment_handler import DeploymentHandler
            handler = DeploymentHandler(self.project_root)
            return handler.route_command(action, params)
        elif system == 'documentation':
            from .documentation_handler import DocumentationHandler
            handler = DocumentationHandler(self.project_root)
            return handler.route_command(action, params)
        elif system == 'benchmarking':
            from .benchmarking_handler import BenchmarkingHandler
            handler = BenchmarkingHandler(self.project_root)
            return handler.route_command(action, params)
        elif system == 'state':
            return self._route_state_command(action, params)
        elif system == 'test':
            return self._route_test_command(action, params)
        elif system == 'git':
            return self._route_git_command(action, params)
        elif system == 'workflow':
            return self._route_workflow_command(action, params)
        elif system == 'help':
            return self._show_help()
        else:
            print(f"⚠️ Unknown system: {system}")
            return 1
    
    def _route_task_command(self, action: str, params: Dict) -> int:
        """Route to task management system."""
        cmd = [sys.executable, str(self.systems['task']), action]
        
        if params.get('task_id'):
            cmd.extend(['--task-id', params['task_id']])
        if params.get('category'):
            cmd.extend(['--category', params['category']])
        if params.get('stage'):
            cmd.extend(['--stage', str(params['stage'])])
        
        return self._execute_command(cmd)
    
    def _route_validation_command(self, action: str, params: Dict) -> int:
        """Route to validation system."""
        cmd = [sys.executable, str(self.systems['validate'])]
        
        # Map actions
        action_map = {
            'quick': 'validate',
            'verify': 'verify',
            'sprint': 'sprint',
            'metrics': 'metrics',
            'dashboard': 'dashboard'
        }
        
        cmd.append(action_map.get(action, action))
        
        if params.get('domain'):
            cmd.extend(['--domain', params['domain']])
        if params.get('stage'):
            cmd.extend(['--stage', str(params['stage'])])
        
        return self._execute_command(cmd)
    
    def _route_state_command(self, action: str, params: Dict) -> int:
        """Route to state system."""
        if self.systems['state'].exists():
            cmd = [sys.executable, str(self.systems['state'])]
            
            if action == 'evolve':
                cmd.append('--evolve')
            elif action == 'status':
                cmd.append('--status')
            
            return self._execute_command(cmd)
        else:
            print("⚠️ State system not available")
            return 1
    
    def _route_test_command(self, action: str, params: Dict) -> int:
        """Route to test runner."""
        if action == 'run':
            cmd = ['pytest', 'tests/', '-v']
        elif action == 'coverage':
            cmd = ['pytest', 'tests/', '--cov=.', '--cov-report=term-missing']
        else:
            cmd = ['pytest', '--help']
        
        return self._execute_command(cmd)
    
    def _route_git_command(self, action: str, params: Dict) -> int:
        """Route to git commands."""
        if action == 'status':
            cmd = ['git', 'status']
        elif action == 'diff':
            cmd = ['git', 'diff', 'HEAD']
        elif action == 'commit':
            # Interactive commit
            print("📝 Preparing commit...")
            subprocess.run(['git', 'status', '--short'])
            message = input("\nCommit message: ").strip()
            if message:
                cmd = ['git', 'commit', '-am', message]
            else:
                print("⚠️ Commit cancelled")
                return 1
        else:
            cmd = ['git', '--help']
        
        return self._execute_command(cmd)
    
    def _route_workflow_command(self, action: str, params: Dict) -> int:
        """Handle workflow commands."""
        if action == 'next':
            print("\n📋 What's Next?")
            print("=" * 50)
            
            # Check for pending tasks
            print("\n1️⃣ Check pending tasks:")
            print("   todo list pending")
            
            # Check validation status
            print("\n2️⃣ Validate current work:")
            print("   todo validate current")
            
            # Check tests
            print("\n3️⃣ Run tests:")
            print("   todo test")
            
            return 0
        
        return 1
    
    def _show_help(self) -> int:
        """Show help information."""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                    🚀 QUARK TODO SYSTEM                          ║
╚══════════════════════════════════════════════════════════════════╝

NATURAL LANGUAGE COMMANDS
═════════════════════════
  todo what's next              → Show suggested next actions
  todo plan new task            → Start task planning
  todo work on [task]           → Execute a task
  todo track progress           → Show task progress
  todo review completed         → Review completed tasks
  todo commit                   → Commit changes
  todo status                   → Git status

TASK MANAGEMENT
═══════════════
  todo list                     → List all tasks
  todo generate from roadmap    → Generate tasks from roadmap
  todo sync tasks               → Sync with roadmap

VALIDATION
══════════
  todo validate foundation      → Validate specific domain
  todo validate current         → Validate current changes
  todo validate sprint          → Interactive validation guide
  todo metrics                  → Show validation metrics
  todo dashboard                → Generate dashboard

BRAIN SYSTEM
════════════
  todo simulate cerebellum      → Simulate cerebellum
  todo simulate cortex          → Simulate cortex  
  todo simulate hippocampus     → Simulate memory system
  todo simulate basal ganglia   → Simulate motor control
  todo simulate morphogen       → Morphogen gradients
  todo simulate e8              → E8 consciousness
  todo brain status             → Brain system status
  todo brain analyze            → Analyze performance
  todo brain test [component]   → Run component tests
  todo brain profile            → Profile performance
  todo brain visualize          → Visualize architecture
  todo brain list               → List all components
  todo run brain simulation     → Full brain simulation

TRAINING
════════
  todo train model --stage 1    → Start training stage 1
  todo train gcp --stage 2      → Train on GCP
  todo training status          → Check training status
  todo stop training            → Stop current training
  todo resume training          → Resume from checkpoint
  todo training metrics         → Show training metrics

DEPLOYMENT
══════════
  todo deploy to gcp            → Deploy to Google Cloud
  todo deploy docker            → Deploy with Docker
  todo deploy local             → Deploy locally
  todo deployment status        → Check deployment
  todo deployment logs          → Show logs
  todo rollback deployment      → Rollback version

DOCUMENTATION
═════════════
  todo generate docs            → Generate all docs
  todo generate api docs        → Generate API docs
  todo update readme            → Update README
  todo check docs               → Check doc status

BENCHMARKING
════════════
  todo benchmark performance    → Performance benchmark (GCP recommended)
  todo benchmark memory         → Memory benchmark
  todo benchmark inference      → Inference benchmark
  todo benchmark training       → Training benchmark
  todo compare metrics          → Compare benchmarks
  todo profile cpu              → Profile CPU usage
  todo profile gpu              → Profile GPU usage
  todo benchmark report         → Generate report

WORKFLOWS
═════════
  todo workflow new_feature     → New feature workflow
  todo workflow daily_standup   → Daily standup
  todo workflow sprint_review   → Sprint review
  todo workflow debug_issue     → Debug workflow

Type 'todo help' or 'make help' for more information
""")
        return 0
    
    def _execute_command(self, cmd: list) -> int:
        """Execute a command and return exit code."""
        try:
            result = subprocess.run(cmd)
            return result.returncode
        except FileNotFoundError:
            print(f"⚠️ Command not found: {cmd[0]}")
            return 1
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
            return 130
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
