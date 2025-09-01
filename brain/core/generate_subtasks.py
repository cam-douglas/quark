
    # ------------------------------------------------------------------
    # Task Planning API (LLM-powered)
    # ------------------------------------------------------------------
    def generate_subtasks(self, bullet: str, priority: str = "medium") -> None:
        """Generate subtasks for *bullet* and broadcast them.

        The tasks are sent to:
        • GlobalWorkspace.broadcast({"tasks": tasks})
        • MetaController.ingest_tasks(tasks)
        """
        if not self._planner_fn:
            print("⚠️ Advanced Planner unavailable – cannot generate subtasks.")
            return
        tasks = self._planner_fn(bullet)
        if not tasks:
            print("⚠️ Planner returned no tasks.")
            return
        # Stamp priority if provided
        for t in tasks:
            t.setdefault("priority", priority)
        # Broadcast to global workspace
        if hasattr(self, "global_workspace"):
            self.global_workspace.broadcast({"tasks": tasks, "source": "advanced_planner"})
        # Feed meta-controller
        if hasattr(self, "meta_controller") and hasattr(self.meta_controller, "ingest_tasks"):
            self.meta_controller.ingest_tasks(tasks)
        print(f"✅ Generated {len(tasks)} tasks for bullet: '{bullet[:40]}...' and broadcasted.")
