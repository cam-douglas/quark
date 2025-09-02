"""Task Bridge – lightweight wrapper around BiologicalBrainTaskIntegration.

Import this from simulators to fetch roadmap/chat tasks and mark them done.
"""
from pathlib import Path
import logging

# Optional import – module may be removed after refactor
try:
    from state.tasks.integrations.biological_brain_task_integration import (
        BiologicalBrainTaskIntegration,
    )
except ModuleNotFoundError:
    # fallback minimal integration
    class BiologicalBrainTaskIntegration:  # type: ignore
        def __init__(self):
            self.central_tasks = {}

        def sync_from_roadmap(self):
            pass

# Removed top-level imports to avoid circular dependencies
# These will be imported lazily when needed

logger = logging.getLogger(__name__)

class _TaskBridge:
    def __init__(self):
        self._integration = BiologicalBrainTaskIntegration()
        self.sync()

    # ------------------------------------------------------------------
    def sync(self):
        """Sync tasks from roadmap YAML + recommendations."""
        self._integration.sync_from_roadmap()

    # ------------------------------------------------------------------
    def get_pending_tasks(self):
        tasks = []
        for t, meta in self._integration.central_tasks.items():
            if meta["status"] != "pending":
                continue
            # If the task is long, break it down with the LLM-based planner
            if len(t.split()) > 12:  # heuristic threshold
                try:
                    # Lazy import to avoid circular dependency
                    from state.quark_state_system.advanced_planner import plan as llm_plan
                    subtasks = [s["title"] for s in llm_plan(t)]
                    tasks.extend(subtasks if subtasks else [t])
                except Exception as exc:  # fallback if model unavailable
                    logger.warning("Advanced planner failed (%s); using original task", exc)
                    tasks.append(t)
            else:
                tasks.append(t)
        return tasks

    # ------------------------------------------------------------------
    def mark_done(self, task: str):
        if task in self._integration.central_tasks:
            self._integration.central_tasks[task]["status"] = "done"
            # If roadmap task, append DONE in its roadmap file
            if self._integration.central_tasks[task].get("source") == "roadmap":
                self._append_done_to_roadmap(task)
            try:
                # Lazy import to avoid circular dependency
                from QUARK_STATE_SYSTEM import update_roadmap_statuses
                update_roadmap_statuses()
            except Exception as exc:
                logger.warning("Roadmap status update failed: %s", exc)

    def _append_done_to_roadmap(self, task_line: str):
        """Search roadmap markdown files for the bullet containing task_line and append DONE."""
        from management.rules.roadmap import roadmap_controller
        import re
        for meta in roadmap_controller.get_all_roadmaps():
            if meta["format"] != "markdown":
                continue
            path = Path(meta["path"])
            text = path.read_text(encoding="utf-8")
            pattern = re.escape(task_line)
            match = re.search(pattern, text)
            if match:
                if "DONE" not in text[match.start(): match.end()+10]:
                    new_text = text.replace(task_line, f"{task_line} DONE")
                    path.write_text(new_text, encoding="utf-8")
                break

TASK_BRIDGE = _TaskBridge()
