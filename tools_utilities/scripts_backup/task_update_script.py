#!/usr/bin/env python3
"""
Task Update Script for Integrated Task Roadmap
Automatically updates task status based on development progress and file changes.
"""

import os
import re
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class TaskUpdater:
    """Manages automatic updates to the integrated task roadmap."""
    
    def __init__(self, project_root: str = "/Users/camdouglas/quark"):
        self.project_root = Path(project_root)
        self.task_roadmap_file = self.project_root / ".cursor" / "rules" / "integrated_task_roadmap.md"
        self.task_status_file = self.project_root / ".cursor" / "rules" / "task_status.json"
        self.biological_blueprint_file = self.project_root / ".cursor" / "rules" / "biological_agi_blueprint.md"
        
        # Task status mapping
        self.status_symbols = {
            "not_started": "📋",
            "in_progress": "🚧", 
            "blocked": "⚠️",
            "completed": "✅",
            "validated": "🎯",
            "integrated": "🔗"
        }
        
        # Load current task status
        self.task_status = self.load_task_status()
        
    def load_task_status(self) -> Dict:
        """Load current task status from JSON file."""
        if self.task_status_file.exists():
            with open(self.task_status_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_task_status(self):
        """Save current task status to JSON file."""
        with open(self.task_status_file, 'w') as f:
            json.dump(self.task_status, f, indent=2)
    
    def detect_task_completion(self, task_id: str) -> bool:
        """Detect if a task has been completed based on file changes and content."""
        task_info = self.get_task_info(task_id)
        if not task_info:
            return False
            
        # Check for task-specific completion indicators
        completion_indicators = self.get_completion_indicators(task_id)
        
        for indicator in completion_indicators:
            if self.check_completion_indicator(indicator):
                return True
                
        return False
    
    def get_task_info(self, task_id: str) -> Optional[Dict]:
        """Get task information from the roadmap."""
        # This would parse the markdown file to extract task info
        # For now, return a basic structure
        return {
            "id": task_id,
            "name": f"Task {task_id}",
            "biological_module": "Unknown",
            "dependencies": []
        }
    
    def get_completion_indicators(self, task_id: str) -> List[Dict]:
        """Get completion indicators for a specific task."""
        indicators = {
            "T1.1.1": [
                {"type": "file_exists", "path": "src/core/neural_components.py"},
                {"type": "content_contains", "path": "src/core/neural_components.py", "text": "STDP"},
                {"type": "test_passes", "path": "tests/test_neural_components.py"}
            ],
            "T1.1.2": [
                {"type": "file_exists", "path": "src/core/neural_components.py"},
                {"type": "content_contains", "path": "src/core/neural_components.py", "text": "BasicNeuralComponent"},
                {"type": "test_passes", "path": "tests/test_neural_components.py"}
            ],
            "T1.1.3": [
                {"type": "file_exists", "path": "src/core/neural_components.py"},
                {"type": "content_contains", "path": "src/core/neural_components.py", "text": "HebbianPlasticity"},
                {"type": "test_passes", "path": "tests/test_neural_components.py"}
            ],
            "T1.1.4": [
                {"type": "file_exists", "path": "tests/"},
                {"type": "file_exists", "path": "tests/conftest.py"},
                {"type": "content_contains", "path": "tests/conftest.py", "text": "pytest"}
            ],
            "T1.1.5": [
                {"type": "file_exists", "path": "tests/validation/"},
                {"type": "content_contains", "path": "tests/", "text": "biological"},
                {"type": "test_passes", "path": "tests/validation/"}
            ]
        }
        
        return indicators.get(task_id, [])
    
    def check_completion_indicator(self, indicator: Dict) -> bool:
        """Check if a completion indicator is satisfied."""
        indicator_type = indicator.get("type")
        
        if indicator_type == "file_exists":
            file_path = self.project_root / indicator["path"]
            return file_path.exists()
            
        elif indicator_type == "content_contains":
            file_path = self.project_root / indicator["path"]
            if not file_path.exists():
                return False
            with open(file_path, 'r') as f:
                content = f.read()
            return indicator["text"] in content
            
        elif indicator_type == "test_passes":
            # This would run the actual tests
            # For now, just check if test file exists
            test_path = self.project_root / indicator["path"]
            return test_path.exists()
            
        return False
    
    def update_task_status(self, task_id: str, new_status: str):
        """Update the status of a specific task."""
        if task_id not in self.task_status:
            self.task_status[task_id] = {}
            
        self.task_status[task_id]["status"] = new_status
        self.task_status[task_id]["last_updated"] = datetime.datetime.now().isoformat()
        
        # Update dependencies
        self.update_dependent_tasks(task_id)
        
        # Save updated status
        self.save_task_status()
        
        # Update the markdown file
        self.update_markdown_roadmap()
    
    def update_dependent_tasks(self, completed_task_id: str):
        """Update status of tasks that depend on the completed task."""
        # This would check the dependency graph and update blocked tasks
        # For now, just mark as in progress if dependencies are met
        pass
    
    def update_markdown_roadmap(self):
        """Update the markdown roadmap file with current task status."""
        if not self.task_roadmap_file.exists():
            return
            
        with open(self.task_roadmap_file, 'r') as f:
            content = f.read()
        
        # Update task status in markdown
        for task_id, task_data in self.task_status.items():
            status = task_data.get("status", "not_started")
            status_symbol = self.status_symbols.get(status, "📋")
            
            # Find and replace task status in markdown
            pattern = rf"(\| {task_id} \|.*?\|.*?\|.*?\|.*?)\|.*?\|"
            replacement = rf"\1| {status_symbol} |"
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        # Write updated content back
        with open(self.task_roadmap_file, 'w') as f:
            f.write(content)
    
    def scan_for_completed_tasks(self):
        """Scan the project for completed tasks and update status."""
        for task_id in self.task_status.keys():
            if self.task_status[task_id].get("status") != "completed":
                if self.detect_task_completion(task_id):
                    self.update_task_status(task_id, "completed")
                    print(f"✅ Task {task_id} marked as completed")
    
    def get_next_priority_tasks(self) -> List[str]:
        """Get the next priority tasks based on dependencies and current status."""
        priority_tasks = []
        
        for task_id, task_data in self.task_status.items():
            status = task_data.get("status", "not_started")
            
            if status == "not_started":
                # Check if dependencies are met
                dependencies_met = self.check_dependencies_met(task_id)
                if dependencies_met:
                    priority_tasks.append(task_id)
        
        return priority_tasks[:5]  # Return top 5 priority tasks
    
    def check_dependencies_met(self, task_id: str) -> bool:
        """Check if all dependencies for a task are met."""
        # This would check the dependency graph
        # For now, return True for basic tasks
        return True
    
    def generate_progress_report(self) -> str:
        """Generate a progress report for the current development status."""
        total_tasks = len(self.task_status)
        completed_tasks = sum(1 for task in self.task_status.values() 
                            if task.get("status") == "completed")
        in_progress_tasks = sum(1 for task in self.task_status.values() 
                              if task.get("status") == "in_progress")
        
        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        report = f"""
# Development Progress Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Progress
- **Total Tasks**: {total_tasks}
- **Completed**: {completed_tasks} ({progress_percentage:.1f}%)
- **In Progress**: {in_progress_tasks}
- **Remaining**: {total_tasks - completed_tasks - in_progress_tasks}

## Next Priority Tasks
"""
        
        priority_tasks = self.get_next_priority_tasks()
        for i, task_id in enumerate(priority_tasks, 1):
            task_info = self.get_task_info(task_id)
            report += f"{i}. **{task_id}**: {task_info.get('name', 'Unknown task')}\n"
        
        return report
    
    def run_automatic_update(self):
        """Run the automatic task update process."""
        print("🔄 Running automatic task update...")
        
        # Scan for completed tasks
        self.scan_for_completed_tasks()
        
        # Generate progress report
        report = self.generate_progress_report()
        
        # Save progress report
        report_file = self.project_root / "docs" / "progress_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print("✅ Automatic task update completed")
        print(f"📊 Progress report saved to: {report_file}")

def main():
    """Main function to run the task updater."""
    updater = TaskUpdater()
    updater.run_automatic_update()

if __name__ == "__main__":
    main()
