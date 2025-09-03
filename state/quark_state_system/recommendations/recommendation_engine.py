#!/usr/bin/env python3
"""Recommendation Engine Module - Core recommendation logic and context analysis.

Provides intelligent recommendations based on QUARK's current state and context.

Integration: Core recommendation logic for QuarkDriver and AutonomousAgent guidance.
Rationale: Centralized recommendation logic with context-aware suggestions.
"""

from typing import List, Dict
from pathlib import Path
import re

def get_recommendations_by_context(context: str = "general", active_tasks: List[str] = None) -> List[str]:
    """Get intelligent recommendations based on context and active tasks."""
    recommendations = []
    active_tasks = active_tasks or []
    
    if context in {"general", "development", "roadmap"}:
        recommendations.append("🗺️  Recommendations from active roadmap tasks:")
        for task in active_tasks[:5]:  # Show top 5 active tasks
            recommendations.append(f"   • {task}")
    
    elif context == "testing":
        recommendations.extend([
            "🧪 Run the test suite to validate current functionality",
            "🔍 Check for any failing tests and debug issues", 
            "📈 Monitor performance metrics and identify bottlenecks"
        ])
        # Add any testing-related tasks from roadmap
        testing_tasks = [t for t in active_tasks if any(kw in t.lower() for kw in ["test", "validate", "benchmark"])]
        if testing_tasks:
            recommendations.append("🔬 Testing tasks from roadmap:")
            for task in testing_tasks[:3]:
                recommendations.append(f"   • {task}")
    
    elif context == "evolution":
        recommendations.extend([
            "🚀 Execute active roadmap milestones",
            "🎯 Focus on current stage development priorities"
        ])
        # Add evolution-related tasks from roadmap
        if active_tasks:
            recommendations.append("🧬 Evolution priorities from roadmap:")
            for task in active_tasks[:3]:
                recommendations.append(f"   • {task}")
    
    return recommendations

def detect_context_from_query(query: str) -> str:
    """Detect context from user query using keyword analysis."""
    query_lower = query.lower()
    
    # Unified keyword → context mapping
    KEYWORDS = {
        "development": ["recommend", "recommendation", "next step", "next steps", "do next", "continue", "proceed"],
        "testing": ["test", "validate", "benchmark"],
        "evolution": ["evolve", "evolution", "stage"],
        "roadmap": ["roadmap", "milestone", "phase"],
        "tasks": ["task", "tasks", "quark tasks", "quarks tasks", "todo", "todos", "agenda"],
        "integrate": ["integrate", "ingest", "add resource"],
    }
    
    for ctx, words in KEYWORDS.items():
        if any(w in query_lower for w in words):
            return ctx
    
    return "general"

def format_guidance_response(context: str, recommendations: List[str], next_actions: List[str], status_summary: str) -> str:
    """Format the complete guidance response."""
    guidance = f"""
🧠 QUARK INTELLIGENT GUIDANCE
{'='*50}

📋 Current Status:
{status_summary}

🎯 Recommendations for {context.title()}:
"""
    for rec in recommendations:
        guidance += f"   {rec}\n"
    
    guidance += f"\n🚀 Immediate Next Actions:\n"
    for action in next_actions:
        guidance += f"   {action}\n"
    
    return guidance
