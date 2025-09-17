#!/usr/bin/env python3
"""Quark Guidance Router - Main entry point for user queries about recommendations and tasks.

Routes user queries to appropriate systems:
- Roadmap recommendations: High-level tasks from management/rules/roadmap/
- Task documentation: Detailed phase tasks from state/tasks/roadmap_tasks/

Integration: Main router for QuarkDriver and user queries.
Rationale: Clear separation between roadmap guidance and detailed task documentation.
"""

from typing import Dict, Any, List
from .recommendation_engine import detect_context_from_query, get_recommendations_by_context, format_guidance_response
from .roadmap_analyzer import get_active_roadmap_status, get_active_roadmap_tasks, get_current_status_summary
from .task_documentation_handler import (
    detect_task_doc_request, 
    get_current_phase_tasks, 
    format_task_documentation_response
)

def handle_user_query(query: str) -> str:
    """Main entry point for handling user queries about recommendations and tasks.
    
    Routes queries to appropriate systems based on intent:
    - 'quark recommendations', 'next tasks', 'roadmap tasks' → roadmap recommendations
    - 'tasks doc', 'task doc', 'phase tasks' → detailed task documentation
    """
    
    # First check if this is a task documentation request
    if detect_task_doc_request(query):
        return handle_task_documentation_request(query)
    
    # Otherwise, handle as roadmap recommendations
    return handle_roadmap_recommendations_request(query)

def handle_roadmap_recommendations_request(query: str) -> str:
    """Handle requests for roadmap recommendations (high-level guidance)."""
    
    # Detect context from query
    context = detect_context_from_query(query)
    
    # Get active roadmap data (NOT from roadmap_tasks directory)
    active_roadmaps = get_active_roadmap_status()
    active_tasks = get_active_roadmap_tasks()  # This now excludes roadmap_tasks directory
    status_summary = get_current_status_summary(active_roadmaps, len(active_tasks))
    
    # Get recommendations based on context
    recommendations = get_recommendations_by_context(context, active_tasks)
    
    # Generate next actions from roadmap tasks
    next_actions = []
    if active_tasks:
        next_actions = [f"🎯 {task}" for task in active_tasks[:3]]
    else:
        next_actions = ["📋 No active roadmap tasks found - check roadmap status"]
    
    # Format and return guidance
    return format_guidance_response(context, recommendations, next_actions, status_summary)

def handle_task_documentation_request(query: str) -> str:
    """Handle requests for detailed task documentation."""
    
    # Get current phase tasks from detailed documentation
    current_tasks = get_current_phase_tasks()
    
    # Format and return task documentation
    return format_task_documentation_response(current_tasks)

def get_system_status() -> Dict[str, Any]:
    """Get overall system status for debugging and monitoring."""
    
    active_roadmaps = get_active_roadmap_status()
    active_tasks = get_active_roadmap_tasks()
    current_phase_tasks = get_current_phase_tasks()
    
    return {
        "roadmap_recommendations": {
            "active_roadmaps_count": len(active_roadmaps),
            "active_tasks_count": len(active_tasks),
            "roadmaps": active_roadmaps
        },
        "task_documentation": {
            "detailed_task_files_count": len(current_phase_tasks),
            "current_phase_tasks": list(current_phase_tasks.keys())
        },
        "separation_working": True
    }

# Convenience functions for direct access
def get_quark_recommendations(query: str = "quark recommendations") -> str:
    """Get Quark's current recommendations (roadmap-based)."""
    return handle_roadmap_recommendations_request(query)

def get_task_documentation(query: str = "tasks doc") -> str:
    """Get detailed task documentation."""
    return handle_task_documentation_request(query)
