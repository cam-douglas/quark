#!/usr/bin/env python3
"""QUARK Recommendations Engine

This script demonstrates how QUARK can now provide intelligent recommendations
based on its current state system, ensuring consistency across all interactions.

Integration: Indirect integration via QuarkDriver and AutonomousAgent; orchestrates simulator runs.
Rationale: State system validates, plans, and triggers actions that the simulator executes.
"""

import json
# Standard libs
from typing import List
from importlib import import_module

# Modules
roadmap_ctrl = import_module('management.rules.roadmaps.roadmap_controller')
loader = import_module('state.quark_state_system.task_loader')

ROADMAP_CTRL = roadmap_ctrl  # backward-compat alias

from QUARK_STATE_SYSTEM import (
    MASTER_ROADMAP,
    ROADMAP_TASK_TMPL,
    ROADMAP_USE_MARKERS,
)

class QuarkRecommendationsEngine:
    """
    QUARK's intelligent recommendations engine that analyzes the current state
    and provides contextually appropriate suggestions.
    """
    
    def __init__(self):
        """Initialise but defer expensive look-ups to a refresh call."""
        self.status_map: dict = {}
        self.next_tasks: list = []
        # Back-compat placeholder so any legacy code referencing current_state works.
        self.current_state: dict = {}

    # ------------------------------------------------------------------
    # Refresh helpers – always fetch the latest roadmap + task snapshot
    # ------------------------------------------------------------------

    def _refresh_state(self, limit: int = 5):
        """Refresh internal caches and return any **new** tasks generated via roadmap sync."""
        # Roadmap status may change between invocations (e.g. new commits).
        from importlib import reload

        # 1. Always reload the task_loader module so we get fresh on-disk YAML.
        tl = reload(loader)  # type: ignore[assignment]

        # Clear existing registry & YAML files before regeneration
        tl.reset_all()

        # Snapshot tasks **before** sync to detect what gets added.
        before_titles = {t.get("title") for t in tl.get_tasks()}

        # 2. Capture latest roadmap status snapshot.
        self.status_map = roadmap_ctrl.status_snapshot()

        # 3. Sync roadmap items → tasks (may create new tasks).
        tl.sync_with_roadmaps(self.status_map)

        # 3b. Generate fine-grained tasks from master roadmap '🚧 In Progress' sections.
        from pathlib import Path
        master_path = MASTER_ROADMAP
        added_from_master = tl.generate_tasks_from_master(master_path)

        # Identify tasks newly created by both sync steps.
        after_tasks = list(tl.get_tasks())
        created_tasks = [t for t in after_tasks if t.get("title") not in before_titles]

        # 4. Update convenience list of next high-priority actions.
        self.next_tasks = list(tl.next_actions(limit=limit))

        return created_tasks

    # Remove _load_current_state and associated regex methods

    def get_current_status_summary(self) -> str:
        """Return summary using structured data."""
        phase4_status = self.status_map.get('HIGH_LEVEL_ROADMAP', 'Unknown')
        return f"Current Phase-4 Status: {phase4_status}. Pending tasks: {len(self.next_tasks)}"
    
    def get_recommendations(self, context: str = "general") -> List[str]:
        """
        Get intelligent recommendations based on QUARK's current state.
        
        Args:
            context: The context for recommendations (e.g., "development", "testing", "evolution")
        """
        # Base recs
        recommendations = []

        # Integrate roadmap phase status
        status_map = ROADMAP_CTRL.get_roadmap_status_map()
        phase4_status = status_map.get('HIGH_LEVEL_ROADMAP', 'Unknown')

        if context in {"general", "development", "roadmap"}:
            recommendations.append(
                f"🗺️  Phase-4 overall status: {phase4_status} (see Master Roadmap)"
            )
        
        # Legacy placeholders (detailed stage/recent_status data now comes directly
        # from roadmap/tasks; retain empty vars for any downstream custom hooks).
        current_stage = ""
        recent_status = ""
        
        if context == "development":
            if "IN PROGRESS" in recent_status:
                recommendations.extend([
                    "🔄 Continue current development work to completion",
                    "🧪 Test the implemented features thoroughly",
                    "📝 Update the state file with progress made"
                ])
            elif "COMPLETE" in recent_status:
                recommendations.extend([
                    "🚀 Move to the next development phase",
                    "✅ Validate all deliverables meet acceptance criteria",
                    "📊 Update project metrics and progress tracking"
                ])
        
        elif context == "testing":
            recommendations.extend([
                "🧪 Run the test suite to validate current functionality",
                "🔍 Check for any failing tests and debug issues",
                "📈 Monitor performance metrics and identify bottlenecks"
            ])
        
        elif context == "evolution":
            if "STAGE N3 COMPLETE" in current_stage or "Phase 3" in current_stage:
                recommendations.extend([
                    "🚀 Begin Phase 4 roadmap implementation",
                    "🎯 Start with cognitive benchmark suite (4.1)",
                    "🧠 Prepare for AGI validation and testing"
                ])
            elif "READY FOR STAGE N4" in current_stage or "Phase 4" in current_stage:
                recommendations.extend([
                    "🚀 Execute Phase 4 roadmap milestones",
                    "🧪 Implement cognitive benchmark suite",
                    "🔬 Develop robustness and adaptivity testing"
                ])
        
        elif context == "roadmap":
            # Roadmap-specific recommendations
            recommendations.extend([
                "🗺️  Review Phase 4 roadmap milestones",
                "🎯 Focus on cognitive benchmark implementation",
                "🧪 Prepare for AGI validation testing",
                "📊 Track progress against roadmap deliverables"
            ])
        
        elif context == "general":
            # General recommendations based on current state
            if "Brain-to-Body Control" in recent_status:
                recommendations.extend([
                    "🧠 Test the brain-to-body control system end-to-end",
                    "⚙️  Tune motor control parameters for optimal performance",
                    "🔄 Validate fallback systems work correctly"
                ])
            
            if "STAGE N3 COMPLETE" in current_stage or "Phase 3" in current_stage:
                recommendations.extend([
                    "🎉 Celebrate Phase 3 roadmap completion",
                    "📊 Document lessons learned and best practices",
                    "🚀 Prepare for Phase 4 roadmap implementation"
                ])
            
            # Add roadmap context
            recommendations.extend([
                "🗺️  Align all development work with high-level roadmap",
                "🎯 Focus on Phase 4 milestone deliverables",
                "📈 Track progress against roadmap phases"
            ])
        
        return recommendations
    
    def get_next_priority_actions(self) -> List[str]:
        """Get the immediate next priority actions based on current state."""
        if not self.next_tasks:
            return ["📋 No pending tasks in task registry"]
        return [f"✅ {t.get('title','Unnamed Task')} (priority: {t.get('priority')})" for t in self.next_tasks]
    
    def provide_intelligent_guidance(self, user_query: str) -> str:
        """
        Provide intelligent guidance based on user query and current state.
        
        Args:
            user_query: The user's question or request for guidance
        """
        query_lower = user_query.lower()

        # Always start with a fresh snapshot so every command ("next steps", "tasks",
        # "recommendations", …) reflects the latest git workspace state.
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
"""

        # Unified keyword → context mapping
        KEYWORDS = {
            "development": ["recommend", "recommendation", "next step", "next steps", "do next", "continue", "proceed"],
            "testing": ["test", "validate", "benchmark"],
            "evolution": ["evolve", "evolution", "stage"],
            "roadmap": ["roadmap", "milestone", "phase"],
            # accept both "quark tasks" and "quarks tasks" plus plural/possessive variants
            "tasks": ["task", "tasks", "quark tasks", "quarks tasks", "todo", "todos", "todo's", "to do", "to do's", "agenda", "next steps", "quark next steps", "quarks next steps", "quark to do", "quarks to do"],
            "integrate": ["integrate", "ingest", "add resource"],
        }

        def detect_context() -> str:
            for ctx, words in KEYWORDS.items():
                if any(w in query_lower for w in words):
                    return ctx
            return "general"

        context = detect_context()

        # When the user explicitly asks for tasks, show **only the tasks freshly
        # generated from the roadmap sync** (ignoring the pre-existing registry)
        # so the list is always aligned with the latest roadmap view.
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
            # Extract the resource path/uri after the keyword 'integrate'
            import re, shlex, subprocess, sys
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
        
        guidance = f"""
🧠 QUARK INTELLIGENT GUIDANCE
{'='*50}

📋 Current Status:
{self.get_current_status_summary()}

🎯 Recommendations for {context.title()}:
"""
        for rec in recommendations:
            guidance += f"   {rec}\n"
        
        guidance += f"\n🚀 Immediate Next Actions:\n"
        for action in next_actions:
            guidance += f"   {action}\n"
        
        return guidance

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

if __name__ == "__main__":
    main()


# S3 and Cloud Integration Recommendations
def get_s3_integration_recommendations():
    """Get recommendations for S3 and cloud integration"""
    return [
        {
            "id": "s3_model_optimization",
            "priority": 0.8,
            "category": "Cloud Optimization",
            "title": "Optimize S3 Model Storage",
            "description": "Review and optimize model storage in S3 bucket",
            "action": "python quark_state_system/s3_model_manager.py",
            "estimated_time": "30 minutes"
        },
        {
            "id": "tokyo_instance_scaling",
            "priority": 0.7,
            "category": "Infrastructure",
            "title": "Consider Instance Scaling",
            "description": "Evaluate if c5.xlarge meets current computational needs",
            "action": "Monitor CPU/memory usage and consider upgrading to GPU instance",
            "estimated_time": "15 minutes"
        },
        {
            "id": "quantum_braket_integration",
            "priority": 0.9,
            "category": "Quantum Computing",
            "title": "Enhance Quantum-Brain Integration",
            "description": "Integrate Braket quantum computing with brain models",
            "action": "python brain_modules/alphagenome_integration/quantum_braket_integration.py",
            "estimated_time": "45 minutes"
        },
        {
            "id": "bedrock_ai_enhancement",
            "priority": 0.85,
            "category": "AI Integration",
            "title": "Expand Bedrock AI Capabilities",
            "description": "Use Claude and Titan models for advanced brain analysis",
            "action": "python brain_modules/alphagenome_integration/bedrock_brain_demo.py",
            "estimated_time": "30 minutes"
        }
    ]



# Quantum Computing Recommendations
def get_quantum_recommendations():
    """Get quantum computing recommendations for brain simulation"""
    from quantum_decision_engine import QuantumDecisionEngine
    
    engine = QuantumDecisionEngine()
    usage_report = engine.get_usage_report()
    
    recommendations = [
        {
            "id": "quantum_consciousness_modeling",
            "priority": 0.9,
            "category": "Quantum Computing",
            "title": "Implement Quantum Consciousness Models",
            "description": "Use quantum entanglement for global workspace theory implementation",
            "action": "route_computation_intelligently('consciousness_modeling', problem_size=100)",
            "estimated_time": "2 hours",
            "quantum_advantage": True
        },
        {
            "id": "optimize_quantum_usage",
            "priority": 0.7,
            "category": "Cost Optimization",
            "title": "Optimize Quantum Computing Usage",
            "description": f"Monitor quantum costs (current: ${usage_report['total_quantum_cost']:.2f})",
            "action": "Review quantum task routing and use simulators when possible",
            "estimated_time": "30 minutes"
        },
        {
            "id": "quantum_memory_research",
            "priority": 0.8,
            "category": "Research",
            "title": "Quantum Memory Consolidation",
            "description": "Explore quantum superposition for hippocampal memory modeling",
            "action": "route_computation_intelligently('memory_consolidation', problem_size=150)",
            "estimated_time": "1.5 hours",
            "quantum_advantage": True
        }
    ]
    
    # Add usage-specific recommendations
    if usage_report['quantum_percentage'] < 10:
        recommendations.append({
            "id": "explore_quantum_benefits",
            "priority": 0.6,
            "category": "Exploration",
            "title": "Explore Quantum Computing Benefits",
            "description": "Try quantum computing for optimization and search tasks",
            "action": "Test quantum decision engine with various brain simulation tasks",
            "estimated_time": "1 hour"
        })
    
    return recommendations
