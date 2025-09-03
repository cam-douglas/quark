#!/usr/bin/env python3
"""Recommendations Module - Main interface for QUARK recommendations system.

Provides unified interface to recommendation engine, roadmap analysis, and intelligent guidance.

Integration: Main recommendations interface for QuarkDriver and AutonomousAgent.
Rationale: Clean API abstraction over recommendation system modules.
"""

from typing import List, Dict
from .recommendation_engine import (
    get_recommendations_by_context,
    detect_context_from_query,
    format_guidance_response
)
from .roadmap_analyzer import (
    get_active_roadmap_status,
    get_active_roadmap_tasks,
    get_current_status_summary,
    analyze_roadmap_progress
)

# Import for compatibility
from importlib import import_module
loader = import_module('state.quark_state_system.task_loader')
roadmap_ctrl = import_module('management.rules.roadmap.roadmap_controller')

class QuarkRecommendationsEngine:
    """
    QUARK's intelligent recommendations engine that analyzes the current state
    and provides contextually appropriate suggestions.
    """
    
    def __init__(self):
        """Initialize but defer expensive look-ups to a refresh call."""
        self.status_map: dict = {}
        self.next_tasks: list = []
        # Back-compat placeholder so any legacy code referencing current_state works.
        self.current_state: dict = {}

    def _refresh_state(self, limit: int = 5):
        """Refresh internal caches and return any new tasks generated via roadmap sync."""
        # Use the current task_loader without reloading to avoid freezes
        
        # 1. Generate fresh tasks from active roadmaps
        added_from_active = loader.generate_tasks_from_active_roadmaps()

        # 2. Capture latest roadmap status snapshot.
        self.status_map = roadmap_ctrl.status_snapshot()

        # 3. Update convenience list of next high-priority actions.
        self.next_tasks = list(loader.next_actions(limit=limit))

        return []  # Return empty list since we're not tracking created vs existing

    def get_current_status_summary(self) -> str:
        """Return summary using live roadmap data only."""
        # Get status from active roadmap files, not legacy documentation
        active_roadmaps = get_active_roadmap_status()
        in_progress_count = len(self.next_tasks)
        
        return get_current_status_summary(active_roadmaps, in_progress_count)
    
    def get_recommendations(self, context: str = "general") -> List[str]:
        """Get intelligent recommendations based on QUARK's current state from active roadmap files only."""
        # Always refresh tasks from roadmaps first (auto-regenerate)
        self._refresh_state()
        
        # Auto-generate YAML file with categorized tasks
        from ..task_management.yaml_generator import ensure_yaml_consistency
        yaml_result, in_progress_tasks = ensure_yaml_consistency()
        print(f"🔄 {yaml_result}")
        
        # Get active roadmap tasks dynamically
        active_tasks = get_active_roadmap_tasks()
        
        return get_recommendations_by_context(context, active_tasks)
    
    def get_next_priority_actions(self) -> List[str]:
        """Get the immediate next priority actions based on current state."""
        if not self.next_tasks:
            return ["📋 No in-progress tasks in task registry"]
        
        # Use the new formatting with sub-headings
        from ..task_management.yaml_generator import format_tasks_by_sprint
        return [format_tasks_by_sprint()]
    
    def provide_intelligent_guidance(self, user_query: str) -> str:
        """Provide intelligent guidance based on user query and current state."""
        query_lower = user_query.lower()

        # Always start with a fresh snapshot
        created_tasks = self._refresh_state()

        # First handle explicit status/help queries
        if "status" in query_lower or "state" in query_lower:
            return self.get_current_status_summary()

        if "help" in query_lower:
            return """
🧠 QUARK HELP - Natural Language Examples
 - "What should I do next?"
 - "Show me QUARK's current status"
 - "How should QUARK evolve?"
 - "Give me the roadmap milestones"
 - "complete task X" - Mark task X as complete
 - "complete task" - Show numbered task list
"""

        # Handle task completion command BEFORE context detection
        if "complete task" in query_lower:
            import re
            task_match = re.search(r"complete task (\d+)", query_lower)
            if task_match:
                task_num = int(task_match.group(1))
                from state.quark_state_system.task_loader import complete_task_by_number, list_in_progress_tasks
                
                # Show current tasks first
                current_tasks = list_in_progress_tasks()
                if not current_tasks:
                    return "📋 No in-progress tasks to complete"
                
                if complete_task_by_number(task_num):
                    return f"✅ Task {task_num} marked as complete and moved to archive!"
                else:
                    task_list = "\n".join(current_tasks)
                    return f"❌ Invalid task number. Current in-progress tasks:\n{task_list}"
            else:
                from state.quark_state_system.task_loader import list_in_progress_tasks
                current_tasks = list_in_progress_tasks()
                if current_tasks:
                    task_list = "\n".join(current_tasks)
                    return f"📋 Current in-progress tasks:\n{task_list}\n\nUsage: 'complete task X' where X is the task number"
                else:
                    return "📋 No in-progress tasks to complete"

        context = detect_context_from_query(user_query)

        # Handle tasks context specially
        if context == "tasks":
            from collections import defaultdict
            from importlib import reload as _reload
            tl = _reload(loader)

            priority_grouped: dict[str, list[dict]] = {"high": [], "medium": [], "low": []}
            for t in tl.get_tasks(status="pending"):
                priority_grouped.get(t.get("priority", "medium"), priority_grouped["medium"]).append(t)

            lines: list[str] = []
            for prio_label in ("high", "medium", "low"):
                tasks = priority_grouped[prio_label]
                if not tasks:
                    continue
                lines.append(prio_label.upper() + ":")
                # group inside each priority by pillar
                by_pillar = defaultdict(list)
                for task in tasks:
                    title = task["title"]
                    if title.lower().startswith("pillar"):
                        pillar = title.split("▶")[0].strip()
                    else:
                        pillar = "Misc"
                    by_pillar[pillar].append(title)

                for pillar in sorted(by_pillar.keys()):
                    lines.append(f"  {pillar}:")
                    for item in by_pillar[pillar]:
                        lines.append(f"    - {item}")

            if not lines:
                lines.append("✅ All roadmap tasks completed!")

            return "\n".join(lines)

        # Handle resource integration command
        if context == "integrate":
            import re, subprocess, sys
            m = re.search(r"integrate\s+(.+)$", query_lower)
            if not m:
                return "⚠️  Usage: quark integrate <path_or_uri>"
            resource = m.group(1).strip()
            if not resource:
                return "⚠️  Provide a path or URI to integrate."
            try:
                subprocess.run([sys.executable, "-m", "state.quark_state_system.integrate_cli", resource], check=True)
                return f"✅ Integration triggered for {resource}"
            except subprocess.CalledProcessError as e:
                return f"❌ Integration failed: {e}"

        recommendations = self.get_recommendations(context)
        next_actions = self.get_next_priority_actions()
        
        return format_guidance_response(context, recommendations, next_actions, self.get_current_status_summary())

# Export main interface for backward compatibility
def main():
    """Main function to demonstrate QUARK's recommendation capabilities."""
    print("🧠 QUARK RECOMMENDATIONS ENGINE")
    print("=" * 50)
    
    quark = QuarkRecommendationsEngine()
    
    # Demonstrate current status
    print("\n📊 CURRENT STATUS:")
    print(quark.get_current_status_summary())
    
    # Demonstrate recommendations
    print("\n🎯 DEVELOPMENT RECOMMENDATIONS:")
    dev_recs = quark.get_recommendations("development")
    for rec in dev_recs:
        print(f"   {rec}")
    
    print("\n🚀 EVOLUTION RECOMMENDATIONS:")
    evo_recs = quark.get_recommendations("evolution")
    for rec in evo_recs:
        print(f"   {rec}")
    
    print("\n✅ IMMEDIATE NEXT ACTIONS:")
    next_actions = quark.get_next_priority_actions()
    for action in next_actions:
        print(f"   {action}")
    
    print("\n💡 QUARK is now ready to provide intelligent guidance!")
    print("   Ask me: 'What are QUARK's recommendations?' or 'What should I do next?'")
