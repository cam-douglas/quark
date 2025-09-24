"""
Workflow Orchestrator Module
============================
Orchestrates complex workflows across multiple systems.
"""

from typing import List, Dict, Optional
from pathlib import Path


class WorkflowOrchestrator:
    """Orchestrates multi-step workflows."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
        # Define common workflows
        self.workflows = {
            'new_feature': [
                ('task', 'plan', 'Plan the feature'),
                ('task', 'execute', 'Implement the feature'),
                ('test', 'run', 'Run tests'),
                ('validate', 'quick', 'Validate changes'),
                ('git', 'commit', 'Commit changes'),
                ('task', 'review', 'Review completion')
            ],
            'daily_standup': [
                ('git', 'status', 'Check git status'),
                ('task', 'track', 'Review task progress'),
                ('validate', 'metrics', 'Check validation metrics'),
                ('workflow', 'next', 'Determine next actions')
            ],
            'sprint_review': [
                ('task', 'review', 'Review completed tasks'),
                ('validate', 'dashboard', 'Generate dashboard'),
                ('task', 'sync', 'Sync with roadmap'),
                ('task', 'generate', 'Generate next sprint tasks')
            ],
            'debug_issue': [
                ('git', 'diff', 'Check recent changes'),
                ('test', 'run', 'Run tests to reproduce'),
                ('validate', 'quick', 'Validate current state'),
                ('git', 'status', 'Review file changes')
            ]
        }
    
    def get_workflow(self, name: str) -> List[tuple]:
        """Get a predefined workflow."""
        return self.workflows.get(name, [])
    
    def list_workflows(self) -> List[str]:
        """List available workflows."""
        return list(self.workflows.keys())
    
    def execute_workflow(self, name: str, router) -> int:
        """Execute a complete workflow."""
        workflow = self.get_workflow(name)
        
        if not workflow:
            print(f"⚠️ Unknown workflow: {name}")
            return 1
        
        print(f"\n🔄 Executing Workflow: {name}")
        print("=" * 50)
        
        for i, (system, action, description) in enumerate(workflow, 1):
            print(f"\n[{i}/{len(workflow)}] {description}...")
            
            # Ask for confirmation
            response = input("Continue? (y/n/skip): ").lower()
            if response == 'n':
                print("⚠️ Workflow cancelled")
                return 1
            elif response == 'skip':
                print("⏭️ Skipped")
                continue
            
            # Execute step
            exit_code = router.route_command(system, action, {})
            if exit_code != 0:
                print(f"⚠️ Step failed with exit code {exit_code}")
                response = input("Continue anyway? (y/n): ").lower()
                if response != 'y':
                    return exit_code
        
        print(f"\n✅ Workflow '{name}' completed!")
        return 0
    
    def suggest_workflow(self, context: Dict) -> Optional[str]:
        """Suggest a workflow based on context."""
        if context.get('has_changes') and not context.get('current_task'):
            return 'debug_issue'
        elif context.get('current_task'):
            return 'new_feature'
        else:
            return 'daily_standup'
    
    def create_custom_workflow(self, steps: List[Dict]) -> str:
        """Create a custom workflow from steps."""
        # This could be expanded to save custom workflows
        workflow_id = f"custom_{len(self.workflows)}"
        
        workflow_steps = []
        for step in steps:
            workflow_steps.append((
                step['system'],
                step['action'],
                step.get('description', f"{step['system']} {step['action']}")
            ))
        
        self.workflows[workflow_id] = workflow_steps
        return workflow_id
