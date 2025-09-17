#!/usr/bin/env python3
"""Task Documentation Handler - Manages detailed phase-specific task documentation.

Handles requests for detailed task documentation from /state/tasks/roadmap_tasks/
Separate from roadmap recommendations to maintain clear separation of concerns.

Integration: Task documentation system for detailed phase-specific tasks.
Rationale: Clear separation between high-level roadmap recommendations and detailed task docs.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import re

def get_task_documentation_files() -> List[Path]:
    """Get all task documentation files from roadmap_tasks directory."""
    tasks_dir = Path("state/tasks/roadmap_tasks")
    if not tasks_dir.exists():
        return []
    
    return list(tasks_dir.glob("*.md"))

def get_current_phase_tasks() -> Dict[str, Any]:
    """Get current phase tasks from detailed task documentation."""
    tasks_files = get_task_documentation_files()
    current_tasks = {}
    
    for task_file in tasks_files:
        try:
            content = task_file.read_text(encoding='utf-8')
            
            # Extract phase information
            phase_match = re.search(r'\*\*Status\*\*:\s*Phase\s+(\d+)\s*▸\s*Batch\s+([A-Z])\s*▸\s*([^*\n]+)', content)
            if phase_match:
                phase = phase_match.group(1)
                batch = phase_match.group(2)
                status = phase_match.group(3).strip()
                
                current_tasks[task_file.stem] = {
                    "file": str(task_file),
                    "phase": phase,
                    "batch": batch,
                    "status": status,
                    "content_preview": content[:500] + "..." if len(content) > 500 else content
                }
        except Exception:
            continue
    
    return current_tasks

def detect_task_doc_request(query: str) -> bool:
    """Detect if user is requesting task documentation (not recommendations)."""
    query_lower = query.lower()
    
    task_doc_keywords = [
        "tasks doc", "task doc", "tasks documentation", "task documentation",
        "phase tasks", "detailed tasks", "current tasks doc", "task breakdown",
        "show me the tasks doc", "show tasks doc", "show me task doc", "show task doc"
    ]
    
    return any(keyword in query_lower for keyword in task_doc_keywords)

def format_task_documentation_response(current_tasks: Dict[str, Any]) -> str:
    """Format response for task documentation requests."""
    if not current_tasks:
        return """
📋 TASK DOCUMENTATION
====================

No detailed task documentation found in /state/tasks/roadmap_tasks/

To create task documentation:
1. Use 'tasks doc' requests for detailed phase-specific tasks
2. Documentation will be stored in /state/tasks/roadmap_tasks/
3. For roadmap recommendations, use 'quark recommendations' instead
"""
    
    response = """
📋 DETAILED TASK DOCUMENTATION
==============================

"""
    
    for task_name, task_info in current_tasks.items():
        response += f"""
📁 {task_name.replace('_', ' ').title()}
   Phase: {task_info['phase']} ▸ Batch: {task_info['batch']}
   Status: {task_info['status']}
   File: {task_info['file']}

"""
    
    response += f"""
Found {len(current_tasks)} detailed task documentation file(s).

💡 Note: This is separate from 'quark recommendations' which shows high-level roadmap tasks.
"""
    
    return response

def should_exclude_from_recommendations(file_path: str) -> bool:
    """Check if a file should be excluded from recommendations (but available for task docs)."""
    excluded_paths = [
        "/state/tasks/roadmap_tasks/",
        "/state/tasks/chat_tasks",
        "/state/tasks/detailed_",
        "/state/tasks/phase_"
    ]
    
    return any(excluded_path in file_path for excluded_path in excluded_paths)
