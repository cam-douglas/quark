"""
Task Executor Module
====================
Guides task execution through P/F/A/O methodology.
"""

from typing import Dict, List, Optional


class TaskExecutor:
    """Guides and manages task execution."""
    
    def __init__(self):
        # P/F/A/O model stages
        self.pfao_stages = {
            "P": {
                "name": "Probe (build the right thing)",
                "steps": [
                    "Research and gather requirements",
                    "Define scope and success criteria",
                    "Validate approach with stakeholders"
                ]
            },
            "F": {
                "name": "Forge (decide & build safely)",
                "steps": [
                    "Design architecture/solution",
                    "Implement core functionality",
                    "Create unit tests"
                ]
            },
            "A": {
                "name": "Assure (prove quality & safety)",
                "steps": [
                    "Run validation tests",
                    "Measure KPIs against targets",
                    "Collect evidence for validation"
                ]
            },
            "O": {
                "name": "Operate (ship, monitor, improve)",
                "steps": [
                    "Deploy/integrate solution",
                    "Monitor performance",
                    "Document lessons learned"
                ]
            }
        }
    
    def get_pfao_guidance(self) -> Dict:
        """Get P/F/A/O execution guidance."""
        return self.pfao_stages
    
    def get_stage_steps(self, stage: str) -> List[str]:
        """Get steps for a specific P/F/A/O stage."""
        if stage in self.pfao_stages:
            return self.pfao_stages[stage]["steps"]
        return []
    
    def get_validation_steps(self, task_id: str) -> List[str]:
        """Get validation steps for a task."""
        return [
            "Run: make validate-quick",
            "Check relevant domain checklist",
            "Measure KPIs manually",
            "Record evidence in validation system",
            "Update task document with results"
        ]
    
    def get_task_requirements(self) -> List[str]:
        """Get general task execution requirements."""
        return [
            "Review task document for full details",
            "Check upstream dependencies are met",
            "Ensure development environment is ready",
            "Have measurement tools available for KPIs"
        ]
    
    def get_execution_reminders(self) -> List[str]:
        """Get reminders for task execution."""
        return [
            "Update task status after each step",
            "Collect evidence for validation",
            "Run tests after implementation",
            "Document any blockers or issues"
        ]
