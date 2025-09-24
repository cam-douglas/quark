
    # ------------------------------------------------------------------
    # External QA Facade
    # ------------------------------------------------------------------
def ask(self, question: str, *, top_k: int = 3) -> str:
        """Answer a natural-language *question* using memory and LLM fallback."""
        if not question or not isinstance(question, str):
            return "I need a question to answer."

        try:
            result = self.knowledge_hub.retrieve(
                question, episodic_memory=self.hippocampus, top_k=top_k
            )
            episodes = result.get("episodes", [])
            if episodes:
                # Return the content of the top episode (simple heuristic)
                content = episodes[0].content
                if isinstance(content, dict) and "text" in content:
                    return content["text"]
                return str(content)

            llm_ans = result.get("llm_answer")
            if llm_ans:
                return llm_ans
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(f"BrainSimulator.ask failed: {e}")
        return "I'm not sure about that yet."
