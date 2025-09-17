
    @property
    def llm_manipulation_planner(self):
        """Lazy loader for LLMManipulationPlanner."""
        if self._llm_manipulation_planner is None:
            print("🤖 Lazily loading LLMManipulationPlanner model...")
            self._llm_manipulation_planner = LLMManipulationPlanner()
            print("✅ LLMManipulationPlanner loaded.")
        return self._llm_manipulation_planner
