"""
Quark Roadmap Intelligence Core

This controller is responsible for parsing all roadmap documents, unifying their objectives,
and providing a clear, prioritized list of goals for the Autonomous Agent.

It understands the structure of the project's markdown-based roadmaps and can
determine the current phase, upcoming milestones, and overall progress.
"""
import os
import re
from glob import glob
from typing import Dict, List, Any, Optional

class RoadmapController:
    """Manages and synthesizes project roadmaps."""

    def __init__(self, workspace_root: str):
        """
        Initializes the Roadmap Controller.

        Args:
            workspace_root: The absolute path to the project's root directory.
        """
        self.workspace_root = workspace_root
        self.roadmaps = self._load_and_parse_roadmaps()
        self.unified_goals = self._unify_goals()
        print(f"✅ Roadmap Intelligence Core Initialized. Found {len(self.roadmaps)} roadmaps.")

    def _load_and_parse_roadmaps(self) -> Dict[str, Any]:
        """
        Finds, loads, and parses all known roadmap files.
        """
        roadmap_files = [
            'quark_state_system/QUARK_ROADMAP.md',
            'tasks/HIGH_LEVEL_ROADMAP.md',
            'documentation/docs/AGI_INTEGRATED_ROADMAP.md',
            'documentation/docs/roadmap.md', # Biological AGI Roadmap
            'management/rules/roadmap/UNIFIED_AGI_MASTER_ROADMAP_FINAL.md'
        ]
        
        parsed_data = {}
        for rel_path in roadmap_files:
            path = os.path.join(self.workspace_root, rel_path)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    content = f.read()
                    parsed_data[rel_path] = self._parse_markdown_roadmap(content)
        return parsed_data

    def _parse_markdown_roadmap(self, content: str) -> Dict[str, Any]:
        """

        A simple parser to extract phases and tasks from roadmap markdown files.
        This looks for headings like '## Phase X' or '### PILLAR Y'.
        """
        data = {"phases": [], "raw_content": content}
        current_phase = None

        for line in content.splitlines():
            phase_match = re.match(r'^##\s+(Phase\s+\d+:.*|PILLAR\s+\d+:.*)', line, re.IGNORECASE)
            if phase_match:
                if current_phase:
                    data["phases"].append(current_phase)
                current_phase = {"title": phase_match.group(1).strip(), "tasks": []}
            elif current_phase and (line.strip().startswith('-') or line.strip().startswith('*')):
                task = line.strip().lstrip('-* ').strip()
                status = "pending"
                if "✅" in task or "COMPLETED" in task:
                    status = "completed"
                elif "🚧" in task or "IN PROGRESS" in task:
                    status = "in_progress"
                current_phase["tasks"].append({"description": task, "status": status})
        
        if current_phase:
            data["phases"].append(current_phase)
            
        return data

    def _unify_goals(self) -> List[Dict[str, Any]]:
        """
        Synthesizes goals from all roadmaps into a single prioritized list.
        For now, it simply concatenates all pending tasks. A more sophisticated
        prioritization logic can be added later.
        """
        unified_list = []
        for roadmap_name, data in self.roadmaps.items():
            for phase in data.get('phases', []):
                for task in phase.get('tasks', []):
                    if task['status'] == 'pending':
                        unified_list.append({
                            "source_roadmap": roadmap_name,
                            "phase": phase['title'],
                            "task": task['description']
                        })
        return unified_list

    def get_next_actionable_goal(self) -> Optional[Dict[str, Any]]:
        """
        Determines the most immediate, actionable goal from the unified list.

        Returns:
            A dictionary representing the next goal, or None if all goals are complete.
        """
        if not self.unified_goals:
            print("🎉 All roadmap goals appear to be complete!")
            return None
        
        # Simple FIFO logic for now. Returns the first pending task.
        next_goal = self.unified_goals[0]
        print(f"🎯 Next Actionable Goal: {next_goal['task']} (from {next_goal['source_roadmap']})")
        return next_goal

    def report_progress(self, completed_goal: Dict[str, Any]):
        """
        Marks a goal as complete and refreshes the unified goal list.
        Note: This does not yet modify the original markdown files.
        """
        # Find and remove the completed goal from the list
        self.unified_goals = [g for g in self.unified_goals if g['task'] != completed_goal['task']]
        print(f"📊 Progress Reported: Completed '{completed_goal['task']}'. {len(self.unified_goals)} goals remaining.")


# Example Usage
if __name__ == '__main__':
    # This assumes you run this script from the workspace root for the paths to work.
    # In the actual implementation, the Autonomous Agent will pass the correct root path.
    try:
        controller = RoadmapController(os.getcwd())
        
        print("\n--- Unified Pending Goals ---")
        for i, goal in enumerate(controller.unified_goals[:5]): # Print top 5
            print(f"{i+1}. {goal['task']}")
            
        print("\n--- Getting Next Action ---")
        next_action = controller.get_next_actionable_goal()

        if next_action:
            print("\n--- Reporting Progress ---")
            controller.report_progress(next_action)
            
            print("\n--- Getting Final Action ---")
            controller.get_next_actionable_goal()

    except Exception as e:
        print(f"Error during example run: {e}")
        print("Please ensure you are running this script from the project's root directory ('/Users/camdouglas/quark').")
