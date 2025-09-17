#!/usr/bin/env python3
"""Recommendation Engine Module - Core recommendation logic and context analysis.

Provides intelligent recommendations based on QUARK's current state and context.

Integration: Core recommendation logic for QuarkDriver and AutonomousAgent guidance.
Rationale: Centralized recommendation logic with context-aware suggestions.
"""

from typing import List

def get_recommendations_by_context(context: str = "general", active_tasks: List[str] = None) -> List[str]:
    """Get intelligent recommendations based on context and active tasks.
    
    For roadmap_recommendations: Returns high-level roadmap tasks from management/rules/roadmap/
    For task_documentation: This function should NOT be called - use separate task doc system
    """
    recommendations = []
    active_tasks = active_tasks or []

    if context == "task_documentation":
        # This should be handled by a separate system, not recommendations
        recommendations.append("⚠️  Task documentation should be accessed via separate task doc system")
        recommendations.append("📁 Check /Users/camdouglas/quark/state/tasks/roadmap_tasks/ for detailed phase tasks")
        return recommendations

    if context in {"general", "development", "roadmap", "roadmap_recommendations"}:
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
    """Detect context from user query using keyword analysis.
    
    Distinguishes between:
    - roadmap_recommendations: For 'quark recommendations', 'next tasks', 'roadmap tasks'
    - task_documentation: For 'tasks doc', 'task doc', specific phase documentation
    """
    query_lower = query.lower()

    # Check for task documentation requests first (more specific)
    TASK_DOC_KEYWORDS = ["tasks doc", "task doc", "tasks documentation", "phase tasks", "detailed tasks", "show me task doc", "show task doc"]
    if any(keyword in query_lower for keyword in TASK_DOC_KEYWORDS):
        return "task_documentation"

    # Unified keyword → context mapping for roadmap recommendations
    KEYWORDS = {
        "roadmap_recommendations": [
            "quark recommendations", "quarks recommendations", "recommend", "recommendation", 
            "next task", "next tasks", "roadmap task", "roadmap tasks", "what should i do", 
            "what to do next", "next step", "next steps", "do next", "continue", "proceed"
        ],
        "testing": ["test", "validate", "benchmark"],
        "evolution": ["evolve", "evolution", "stage"],
        "roadmap": ["roadmap", "milestone", "phase"],
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

    guidance += "\n🚀 Immediate Next Actions:\n"
    for action in next_actions:
        guidance += f"   {action}\n"

    return guidance
