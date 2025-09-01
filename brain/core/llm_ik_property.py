
    @property
    def llm_ik(self):
        """Lazy loader for LLMInverseKinematics."""
        if self._llm_ik is None:
            print("🤖 Lazily loading LLMInverseKinematics model...")
            self._llm_ik = LLMInverseKinematics()
            print("✅ LLMInverseKinematics loaded.")
        return self._llm_ik
