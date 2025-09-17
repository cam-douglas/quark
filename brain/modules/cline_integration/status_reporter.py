"""
Status Reporter - Comprehensive status reporting for Quark-Cline integration

Provides detailed status reports for Foundation Layer tasks, autonomous execution
capabilities, and integration with existing Quark State System.
"""

from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import sys

# Import your existing task management system
sys.path.append(str(Path(__file__).resolve().parents[3]))

from state.quark_state_system.task_management.task_loader import (
    get_tasks, get_sprint_summary
)


def get_quark_cline_status() -> Dict[str, Any]:
    """Get comprehensive status of Quark-Cline integration"""
    from .task_integration_core import QuarkClineIntegration
    
    integration = QuarkClineIntegration()
    
    return {
        "autonomous_execution_status": integration.get_autonomous_execution_status(),
        "foundation_layer_status": get_foundation_layer_status(),
        "integration_ready": True,
        "cline_adapter_available": integration.cline_adapter is not None,
        "timestamp": datetime.now().isoformat()
    }


def get_foundation_layer_status() -> Dict[str, Any]:
    """
    Get specific status for Foundation Layer tasks
    
    Returns:
        Foundation Layer task status and autonomous execution potential
    """
    from .task_integration_core import QuarkClineIntegration
    
    integration = QuarkClineIntegration()
    all_tasks = get_tasks()
    
    # Filter Foundation Layer tasks
    foundation_tasks = [
        task for task in all_tasks
        if any(term in task.get("title", "").lower() 
              for term in ["foundation", "morphogen", "gradient", "bmp", "wnt", "fgf", "shh"])
    ]
    
    # Categorize by status
    completed = [t for t in foundation_tasks if t.get("status") == "completed"]
    in_progress = [t for t in foundation_tasks if t.get("status") == "in-progress"]
    pending = [t for t in foundation_tasks if t.get("status") == "pending"]
    
    # Find autonomous-ready Foundation Layer tasks
    autonomous_ready = [
        task for task in pending
        if integration.can_execute_autonomously(task)
    ]
    
    return {
        "total_foundation_tasks": len(foundation_tasks),
        "completed": len(completed),
        "in_progress": len(in_progress), 
        "pending": len(pending),
        "completion_percentage": (len(completed) / len(foundation_tasks) * 100) if foundation_tasks else 0,
        "autonomous_ready": len(autonomous_ready),
        "next_autonomous_tasks": [
            {
                "title": task.get("title"),
                "priority": task.get("priority"),
                "phase": task.get("phase")
            }
            for task in autonomous_ready[:3]
        ],
        "task_breakdown": {
            "morphogen_systems": _count_tasks_by_keyword(foundation_tasks, ["morphogen", "gradient", "bmp", "wnt", "fgf"]),
            "spatial_structure": _count_tasks_by_keyword(foundation_tasks, ["ventricular", "meninges", "spatial"]),
            "validation": _count_tasks_by_keyword(foundation_tasks, ["validation", "atlas", "metrics"]),
            "ml_integration": _count_tasks_by_keyword(foundation_tasks, ["ml", "diffusion", "vit", "gnn"])
        }
    }


def get_execution_summary(execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate execution summary from history
    
    Args:
        execution_history: List of execution results
        
    Returns:
        Summary statistics and insights
    """
    if not execution_history:
        return {
            "total_executions": 0,
            "success_rate": 0,
            "avg_biological_compliance": 0,
            "recent_activity": "No executions yet"
        }
    
    successful = [e for e in execution_history if e.get("success")]
    failed = [e for e in execution_history if not e.get("success")]
    
    # Calculate biological compliance average
    compliance_scores = [
        e.get("biological_compliance", 0) 
        for e in successful 
        if e.get("biological_compliance") is not None
    ]
    avg_compliance = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0
    
    # Get recent activity
    recent_executions = execution_history[-5:]  # Last 5
    recent_activity = []
    for exec_result in recent_executions:
        status = "✅" if exec_result.get("success") else "❌"
        title = exec_result.get("quark_task_title", "Unknown Task")
        recent_activity.append(f"{status} {title}")
    
    return {
        "total_executions": len(execution_history),
        "successful_executions": len(successful),
        "failed_executions": len(failed),
        "success_rate": len(successful) / len(execution_history) * 100,
        "avg_biological_compliance": avg_compliance * 100,  # Convert to percentage
        "recent_activity": recent_activity,
        "last_execution": execution_history[-1].get("timestamp") if execution_history else None
    }


def generate_progress_report() -> str:
    """
    Generate a formatted progress report for Foundation Layer
    
    Returns:
        Formatted string report
    """
    status = get_quark_cline_status()
    foundation_status = status["foundation_layer_status"]
    autonomous_status = status["autonomous_execution_status"]
    
    report_lines = [
        "🧠 QUARK FOUNDATION LAYER - AUTONOMOUS EXECUTION REPORT",
        "=" * 55,
        "",
        f"📊 FOUNDATION LAYER STATUS",
        f"  Total Tasks: {foundation_status['total_foundation_tasks']}",
        f"  Completed: {foundation_status['completed']} ({foundation_status['completion_percentage']:.1f}%)",
        f"  In Progress: {foundation_status['in_progress']}",
        f"  Pending: {foundation_status['pending']}",
        "",
        f"🤖 AUTONOMOUS EXECUTION STATUS",
        f"  Ready for Autonomous Execution: {foundation_status['autonomous_ready']}",
        f"  Total Pending Tasks: {autonomous_status['pending_tasks']}",
        f"  Execution History: {autonomous_status['execution_history_count']} completed",
        "",
        f"🎯 NEXT AUTONOMOUS TASKS",
    ]
    
    # Add next tasks
    for i, task in enumerate(foundation_status['next_autonomous_tasks'], 1):
        report_lines.append(f"  {i}. {task['title']} (Priority: {task['priority']})")
    
    if not foundation_status['next_autonomous_tasks']:
        report_lines.append("  No tasks currently ready for autonomous execution")
    
    report_lines.extend([
        "",
        f"📈 TASK BREAKDOWN",
        f"  Morphogen Systems: {foundation_status['task_breakdown']['morphogen_systems']} tasks",
        f"  Spatial Structure: {foundation_status['task_breakdown']['spatial_structure']} tasks", 
        f"  Validation: {foundation_status['task_breakdown']['validation']} tasks",
        f"  ML Integration: {foundation_status['task_breakdown']['ml_integration']} tasks",
        "",
        f"🚀 INTEGRATION STATUS: {'✅ READY' if status['integration_ready'] else '❌ NOT READY'}",
        f"📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ])
    
    return "\n".join(report_lines)


def _count_tasks_by_keyword(tasks: List[Dict[str, Any]], keywords: List[str]) -> int:
    """Count tasks that contain any of the specified keywords"""
    count = 0
    for task in tasks:
        title = task.get("title", "").lower()
        description = task.get("description", "").lower()
        content = title + " " + description
        
        if any(keyword in content for keyword in keywords):
            count += 1
    
    return count


def get_task_priority_matrix() -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate task priority matrix for Foundation Layer
    
    Returns:
        Dictionary with tasks organized by priority levels
    """
    from .task_integration_core import QuarkClineIntegration
    
    integration = QuarkClineIntegration()
    all_tasks = get_tasks()
    
    # Filter Foundation Layer tasks
    foundation_tasks = [
        task for task in all_tasks
        if any(term in task.get("title", "").lower() 
              for term in ["foundation", "morphogen", "gradient", "bmp", "wnt", "fgf"])
    ]
    
    # Organize by priority
    priority_matrix = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": []
    }
    
    for task in foundation_tasks:
        if task.get("status") == "completed":
            continue  # Skip completed tasks
            
        priority = task.get("priority", "medium").lower()
        autonomous_ready = integration.can_execute_autonomously(task)
        
        task_info = {
            "title": task.get("title"),
            "status": task.get("status"),
            "phase": task.get("phase"),
            "autonomous_ready": autonomous_ready,
            "id": task.get("id")
        }
        
        if priority in priority_matrix:
            priority_matrix[priority].append(task_info)
        else:
            priority_matrix["medium"].append(task_info)
    
    return priority_matrix
