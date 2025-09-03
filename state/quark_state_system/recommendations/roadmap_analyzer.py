#!/usr/bin/env python3
"""Roadmap Analyzer Module - Extract status and tasks from active roadmap files.

Analyzes roadmap files to extract current status and active tasks for recommendations.

Integration: Roadmap analysis for QuarkDriver and AutonomousAgent state awareness.
Rationale: Specialized roadmap analysis separate from recommendation logic.
"""

from typing import List, Dict, Any
from pathlib import Path
import re

def get_active_roadmap_status() -> Dict[str, str]:
    """Get status from active roadmap files only, ignoring legacy docs."""
    roadmap_dir = Path("management/rules/roadmap")
    active_roadmaps = {}
    
    for roadmap_file in roadmap_dir.glob("*.md"):
        # Skip any files in archive directories
        if "archive" in str(roadmap_file).lower() or "backup" in str(roadmap_file).lower():
            continue
            
        try:
            content = roadmap_file.read_text(encoding='utf-8')
            # Extract roadmap status
            status_match = re.search(r'\*\*Roadmap Status:\*\*\s*📋\s*(.+)', content)
            if status_match:
                status = status_match.group(1).strip()
                stage_name = roadmap_file.stem.replace('_rules', '').replace('_', ' ').title()
                active_roadmaps[stage_name] = status
        except Exception:
            continue
            
    return active_roadmaps

def get_active_roadmap_tasks() -> List[str]:
    """Get only in-progress tasks from task loader registry."""
    try:
        # Import the task loader to get the current tasks
        from state.quark_state_system.task_loader import get_tasks
        
        # Get only in-progress tasks and format them for display
        tasks = get_tasks(status="in-progress")
        return [task.get("title", "Unnamed Task") for task in tasks[:10]]  # Limit to first 10 for display
        
    except ImportError:
        return []

def get_current_status_summary(active_roadmaps: Dict[str, str], in_progress_task_count: int) -> str:
    """Generate current status summary from roadmap data."""
    if active_roadmaps:
        status_text = ", ".join([f"{name}: {status}" for name, status in active_roadmaps.items()])
        return f"Active Roadmaps: {status_text}. In-progress tasks: {in_progress_task_count}"
    else:
        return f"No active roadmaps found. In-progress tasks: {in_progress_task_count}"

def analyze_roadmap_progress() -> Dict[str, Any]:
    """Analyze overall roadmap progress and completion status."""
    roadmap_status = get_active_roadmap_status()
    
    total_roadmaps = len(roadmap_status)
    in_progress_count = sum(1 for status in roadmap_status.values() if "Progress" in status)
    completed_count = sum(1 for status in roadmap_status.values() if "Complete" in status)
    planned_count = sum(1 for status in roadmap_status.values() if "Planned" in status)
    
    return {
        "total_roadmaps": total_roadmaps,
        "in_progress": in_progress_count,
        "completed": completed_count,
        "planned": planned_count,
        "completion_percentage": (completed_count / total_roadmaps * 100) if total_roadmaps > 0 else 0,
        "active_roadmaps": {name: status for name, status in roadmap_status.items() if "Progress" in status}
    }
