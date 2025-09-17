"""Self-Learning Orchestrator - The executive for knowledge acquisition.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

import logging
from typing import Dict, Any

from brain.architecture.tools.internet_scraper import InternetScraper
from brain.architecture.tools.kaggle_connector import KaggleConnector
from brain.architecture.tools.github_connector import GitHubConnector
from brain.architecture.tools.huggingface_connector import HuggingFaceConnector
from brain.architecture.tools.academic_connector import AcademicConnector
from brain.architecture.neural_core.cognitive_systems.knowledge_hub import KnowledgeHub, KnowledgeObject

logger = logging.getLogger(__name__)

class SelfLearningOrchestrator:
    """
    Identifies knowledge gaps, seeks information using various tools,
    and passes it to the KnowledgeHub for processing.
    """
    from collections import deque

    _WINDOW = 5  # epochs
    _PLATEAU_TOL = 0.002  # loss improvement threshold

    def __init__(self, knowledge_hub: KnowledgeHub):
        """Initializes the orchestrator and its tools."""
        self.khub = knowledge_hub
        self.academic_connector = AcademicConnector()
        self.recently_researched = set()
        # Instantiate all our data gathering tools
        self.scraper = InternetScraper()
        self.kaggle = KaggleConnector()
        self.github = GitHubConnector()
        self.huggingface = HuggingFaceConnector()
        self.academic = AcademicConnector()

    # ------------------------------------------------------------------
    # Meta-learning trigger
    # ------------------------------------------------------------------
    def maybe_trigger_self_training(self, metrics: dict[str, float]) -> None:
        """Track loss and invoke fine-tune when learning plateaus.

        Rule: maintain a moving window of the last `_WINDOW` epoch losses; if
        the best loss in the window is not at least `_PLATEAU_TOL` lower than
        the earliest loss, and we have seen ≥10 k samples, trigger fine-tune.
        """
        loss = metrics.get("loss", 0.0)
        seen = metrics.get("samples_seen", 0)

        if not hasattr(self, "_loss_hist"):
            self._loss_hist = self.__class__.deque(maxlen=self._WINDOW)  # type: ignore[attr-defined]

        self._loss_hist.append(loss)
        if len(self._loss_hist) < self._WINDOW:
            return  # need more data

        earliest = self._loss_hist[0]
        best = min(self._loss_hist)
        if earliest - best < self._PLATEAU_TOL and seen >= 10_000:
            self.khub.handle_command("fine-tune quark")

    def seek_and_assimilate(self, brain_modules: Dict[str, Any], brain_status: Dict[str, Any], topic_hint: str = None):
        """
        The main loop for the self-learning process.
        """
        knowledge_gap = self._identify_knowledge_gap(brain_status, topic_hint)

        if not knowledge_gap or knowledge_gap in self.recently_researched:
            logger.info(f"Skipping redundant or empty knowledge gap: '{knowledge_gap}'")
            return

        logger.info(f"Identified knowledge gap. Seeking info on: '{knowledge_gap}'")
        self.recently_researched.add(knowledge_gap)
        if len(self.recently_researched) > 20: # Keep the set from growing indefinitely
            self.recently_researched.pop()

        raw_data = self._seek_knowledge(knowledge_gap)

        if raw_data:
            # raw_data is a dict with keys: source, content, type, citation (optional)
            try:
                k_objects = self.khub.assimilate(
                    raw_data["content"],
                    source=raw_data.get("source", "unknown"),
                    citation=raw_data.get("citation", ""),
                )
            except Exception as e:
                logger.error(f"KnowledgeHub.assimilate failed: {e}")
                return

            for k_obj in k_objects:
                self._inject_knowledge(brain_modules, k_obj)
        else:
            logger.warning(f"No information found for knowledge gap: {knowledge_gap}")

    def _identify_knowledge_gap(self, brain_status: Dict[str, Any], topic_hint: str = None) -> str:
        """
        Identifies a knowledge gap based on the brain's current state.
        This is a simplified heuristic. A real system would have a more
        complex model of its own knowledge.
        Args:
            brain_status: The current status of the BrainSimulator.
        Returns:
            A search query representing the most pressing knowledge gap.
        """
        # Make the knowledge gap identification dynamic and context-aware
        if topic_hint:
            # Formulate a research query based on the user's input
            return f"linguistic analysis of '{topic_hint}'"

        # Fallback to a more generic, but still useful, knowledge gap
        # This can be improved with more sophisticated logic based on brain_status
        return "improving reinforcement learning sample efficiency"

    def _seek_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Uses various tools to find information about a given query.
        """
        # For now, we prioritize academic sources for grounding.
        try:
            results = self.academic_connector.search_pubmed(query, max_results=1)
            if results:
                # Assuming the first result is the most relevant
                best_result = results[0]
                logger.info(f"Assimilated knowledge from PubMed article: https://pubmed.ncbi.nlm.nih.gov/{best_result['uid']}")
                return {
                    "source": f"PubMed:{best_result['uid']}",
                    "content": best_result['abstract'] or best_result['title'],
                    "type": "declarative"
                }
        except Exception as e:
            logger.error(f"Error seeking knowledge from PubMed: {e}")
            return None
        return None

    def _inject_knowledge(self, brain_modules: Dict[str, Any], knowledge: "KnowledgeObject"):
        """
        Injects assimilated knowledge into the appropriate brain modules.
        """
        if knowledge.k_type == "declarative" or getattr(knowledge.k_type, "value", None) == "declarative":
            if "hippocampus" in brain_modules:
                brain_modules["hippocampus"].inject_knowledge(knowledge)
                logger.info("Injected declarative knowledge into Hippocampus.")
        elif knowledge.k_type == "procedural" or getattr(knowledge.k_type, "value", None) == "procedural":
            if "rl_agent" in brain_modules:
                brain_modules["rl_agent"].inject_knowledge(knowledge)
                logger.info("Injected procedural knowledge into RL Agent.")
        else:
            logger.warning(
                f"Could not inject knowledge of type '{knowledge.k_type}'. No suitable module found."
            )
