"""
Task Tracker Module
===================
Tracks task progress, status, and provides reporting functionality.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class TaskTracker:
    """Tracks and reports on task status and progress."""
    
    def __init__(self, tasks_dir: Path):
        self.tasks_dir = tasks_dir
    
    def load_tasks(self, status_filter: Optional[str] = None) -> List[Dict]:
        """Load tasks from task files with optional status filtering."""
        tasks = []
        
        for task_file in self.tasks_dir.glob("*.md"):
            task_data = self.parse_task_file(task_file)
            if task_data:
                if not status_filter or task_data.get("status") == status_filter:
                    tasks.append(task_data)
        
        return tasks
    
    def parse_task_file(self, task_file: Path) -> Optional[Dict]:
        """Parse a task markdown file to extract metadata."""
        task_data = {
            "id": task_file.stem,
            "file": task_file,
            "name": "Unknown",
            "status": "PENDING",
            "phase": "1",
            "category": "uncategorized",
            "priority": "Medium",
            "progress": 0
        }
        
        with open(task_file) as f:
            for line in f:
                if line.startswith("# "):
                    # Extract task name from header
                    parts = line[2:].strip().split(" - ")
                    if len(parts) >= 2:
                        task_data["category"] = parts[0].strip("📋 ").lower().replace(" ", "_")
                        task_data["name"] = parts[1].strip()
                elif "**Status**:" in line:
                    task_data["status"] = self._extract_status(line)
                    task_data["progress"] = self._calculate_progress(task_data["status"])
                elif "**Priority**:" in line:
                    task_data["priority"] = line.split("**Priority**:")[1].split("-")[0].strip()
                elif "Phase" in line and "▸" in line:
                    # Extract phase number
                    phase_match = re.search(r'Phase (\d+)', line)
                    if phase_match:
                        task_data["phase"] = phase_match.group(1)
        
        return task_data
    
    def get_task_summary(self, tasks: List[Dict]) -> Dict:
        """Get summary statistics for tasks."""
        total = len(tasks)
        pending = len([t for t in tasks if t.get("status") == "PENDING"])
        in_progress = len([t for t in tasks if t.get("status") == "IN_PROGRESS"])
        completed = len([t for t in tasks if t.get("status") == "COMPLETED"])
        blocked = len([t for t in tasks if t.get("status") == "BLOCKED"])
        
        avg_progress = 0
        if tasks:
            avg_progress = sum(t.get("progress", 0) for t in tasks) // len(tasks)
        
        return {
            "total": total,
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "blocked": blocked,
            "avg_progress": avg_progress
        }
    
    def group_by_category(self, tasks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group tasks by category."""
        by_category = {}
        for task in tasks:
            cat = task.get("category", "uncategorized")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(task)
        return by_category
    
    def sync_with_roadmap(self, update: bool = False) -> Dict[str, str]:
        """Sync task status with roadmap documents."""
        task_statuses = {}
        
        for task_file in self.tasks_dir.glob("*.md"):
            status = self._extract_task_status(task_file)
            task_statuses[task_file.stem] = status
        
        if update:
            self._create_status_report(task_statuses)
        
        return task_statuses
    
    def _extract_status(self, line: str) -> str:
        """Extract status from a line."""
        if "COMPLETED" in line:
            return "COMPLETED"
        elif "IN_PROGRESS" in line:
            return "IN_PROGRESS"
        elif "BLOCKED" in line:
            return "BLOCKED"
        else:
            return "PENDING"
    
    def _calculate_progress(self, status: str) -> int:
        """Calculate progress percentage based on status."""
        progress_map = {
            "COMPLETED": 100,
            "IN_PROGRESS": 50,
            "BLOCKED": 25,
            "PENDING": 0
        }
        return progress_map.get(status, 0)
    
    def _extract_task_status(self, task_file: Path) -> str:
        """Extract status from a task file."""
        with open(task_file) as f:
            for line in f:
                if "**Status**:" in line:
                    return self._extract_status(line)
        return "UNKNOWN"
    
    def _create_status_report(self, task_statuses: Dict[str, str]) -> None:
        """Create a status report file."""
        status_report = self.tasks_dir / "STATUS_REPORT.md"
        
        with open(status_report, 'w') as f:
            f.write(f"# Task Status Report\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group by status
            by_status = {}
            for task_id, status in task_statuses.items():
                if status not in by_status:
                    by_status[status] = []
                by_status[status].append(task_id)
            
            # Write sections
            for status in ["COMPLETED", "IN_PROGRESS", "PENDING", "BLOCKED", "UNKNOWN"]:
                if status in by_status:
                    icon = self._get_status_icon(status)
                    f.write(f"\n## {icon} {status} ({len(by_status[status])} tasks)\n\n")
                    for task_id in sorted(by_status[status]):
                        f.write(f"- `{task_id}`\n")
    
    def _get_status_icon(self, status: str) -> str:
        """Get icon for task status."""
        icons = {
            "PENDING": "⏳",
            "IN_PROGRESS": "🔄",
            "COMPLETED": "✅",
            "BLOCKED": "🚫",
            "UNKNOWN": "❓"
        }
        return icons.get(status, "❓")
