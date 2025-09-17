"""Knowledge Hub - The central processing unit for external information.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""

import time
from typing import Dict, Any, List, Optional
from enum import Enum

# Import retriever and LLM fallback lazily to avoid heavy deps at import time
from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
from brain.architecture.neural_core.cognitive_systems.callback_hub import hub


class KnowledgeType(Enum):
    DECLARATIVE = "declarative"  # Facts, concepts
    PROCEDURAL = "procedural"    # Skills, how-to knowledge
    EPISODIC = "episodic"        # Events, experiences


class KnowledgeObject:
    """
    A standardized representation of a piece of knowledge that can be
    injected into various brain modules.

    New fields added (2025-08-30):
        embedding: Optional[list[float]] -- vector representation for fast similarity search
        created_at: float               -- unix timestamp of object creation

    Back-compatibility: the existing call sites in SelfLearningOrchestrator and
    others still supply the original 4 positional arguments, so the new
    parameters must remain optional **after** those required ones.
    """

    def __init__(
        self,
        k_type: KnowledgeType,
        content: Any,
        source: str,
        citation: str,
        embedding: Optional[List[float]] = None,
        created_at: Optional[float] = None,
    ):
        self.k_type = k_type
        self.content = content
        self.source = source
        self.citation = citation
        self.embedding = embedding  # May be filled later by retriever/encoder
        self.created_at = created_at if created_at is not None else time.time()

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize object to plain-python dict for persistence/logging."""
        return {
            "k_type": self.k_type.value if isinstance(self.k_type, KnowledgeType) else self.k_type,
            "content": self.content,
            "source": self.source,
            "citation": self.citation,
            "embedding": self.embedding,
            "created_at": self.created_at,
        }


class KnowledgeHub:
    """
    Processes raw data from various sources and transforms it into
    standardized KnowledgeObjects for assimilation by the brain.
    """
    def __init__(self, memory_provider: Optional["EpisodicMemory"] = None):
        # Memory provider is optional; many pipelines supply episodic memory
        # at call time (e.g., retrieve) rather than construction time.
        self.memory = memory_provider

        # If the provider is E8MemoryAdapter-aware keep reference for mood/drive plumbing
        self._maybe_adapter = getattr(memory_provider, "_e8_adapter", None) if memory_provider else None

        # Retriever is initialised lazily because it needs access to an
        # EpisodicMemory instance to build its index.
        self._retriever = None  # type: Optional["KnowledgeRetriever"]
        self.resource_manager = ResourceManager(auto_scan=False)
        hub.register(self._on_resource_event)
        self._resource_map = {}
        # Rebuild resource map from persisted registry
        for rid, meta in self.resource_manager.registry.items():
            if path := meta.get("integrated_path"):
                self._resource_map[rid] = path
        # Start consumer to auto-assimilate resources
        from brain.architecture.neural_core.cognitive_systems.resource_consumer import ResourceConsumer
        self._consumer = ResourceConsumer(self)

    def _on_resource_event(self, event: str, data):
        """Store integrated paths for quick lookup by other modules."""
        if event == "resource_integrated":
            self._resource_map[data.get("id")] = data.get("path")

    # New API
    def ensure_resource(self, path: str, *, force: bool = False) -> str | None:
        """Register a local resource and return integrated path once available."""
        rid = self.resource_manager.register_resource(path, metadata={"force": force})
        meta = self.resource_manager.registry.get(rid, {})
        return meta.get("integrated_path")

    def process_text(self, text: str, source: str, citation: str) -> List[KnowledgeObject]:
        """
        Processes raw text into declarative and episodic knowledge.
        """
        # This is a simplified implementation. A real system would use
        # advanced NLP for entity recognition, summarization, and classification.

        # For now, we'll treat each paragraph as a declarative fact.
        knowledge_objects = []
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        for para in paragraphs:
            # Simple heuristic to classify knowledge type
            k_type = KnowledgeType.PROCEDURAL if "how to" in para.lower() or "step-by-step" in para.lower() else KnowledgeType.DECLARATIVE

            knowledge_objects.append(
                KnowledgeObject(
                    k_type=k_type,
                    content={"text": para},
                    source=source,
                    citation=citation
                )
            )
        return knowledge_objects

    def process_dataset(self, dataset: List[Dict], source: str, citation: str) -> List[KnowledgeObject]:
        """
        Processes structured data from a dataset into procedural or declarative knowledge.
        """
        # For now, we'll assume the dataset is for imitation learning (procedural)
        return [
            KnowledgeObject(
                k_type=KnowledgeType.PROCEDURAL,
                content={"dataset": dataset},
                source=source,
                citation=citation
            )
        ]

    # ------------------------------------------------------------------
    # Retrieval wrapper
    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        episodic_memory: "EpisodicMemory",
        top_k: int = 5,
        use_llm_fallback: bool = True,
    ) -> Dict[str, Any]:
        """High-level retrieval pipeline.

        Parameters
        ----------
        question : str
            User question.
        episodic_memory : EpisodicMemory
            Memory instance to query.
        top_k : int, default 5
            How many episodes to return.
        use_llm_fallback : bool, default True
            Whether to call LLM fallback if no relevant episodes found.

        Returns
        -------
        dict with keys:
            episodes : list[MemoryEpisode]
            llm_answer : Optional[str]
        """

        # Lazy import to avoid circular deps at module load
        from brain.architecture.neural_core.cognitive_systems.knowledge_retriever import (
            KnowledgeRetriever,
        )
        from brain.architecture.neural_core.cognitive_systems.llm_fallback import (
            answer_with_llm,
        )

        if self._retriever is None or self._retriever.memory is not episodic_memory:
            self._retriever = KnowledgeRetriever(episodic_memory)

        try:
            ranked = self._retriever.retrieve(question, top_k=top_k)
            episodes = [ep for ep, _ in ranked]
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(f"KnowledgeHub.retrieve error: {e}")
            episodes = []

        llm_ans: Optional[str] = None
        if use_llm_fallback and not episodes:
            llm_ans = answer_with_llm(question)

        return {"episodes": episodes, "llm_answer": llm_ans}

    # ------------------------------------------------------------------
    # Natural-language command router (streaming training/fine-tune)
    # ------------------------------------------------------------------

    def handle_command(self, text: str) -> Optional[str]:
        """Parse natural language commands like *train quark* or *fine-tune quark*.

        Returns a human-readable status string when a command was executed,
        otherwise *None* to indicate the hub did not handle the text.
        """
        import re

        t = text.lower().strip()

        m_train = re.search(r"\btrain\s+quark(?:\s+with\s+(?P<path>\S+))?", t)
        if m_train:
            local_path = m_train.group("path") if m_train.group("path") else None
            overrides = {"data_mode": "streaming", "bucket": "quark-main-tokyo-bucket"}
            rc = self.resource_manager.run_training_job(
                "train",
                overrides=overrides,
                dataset_local_path=local_path,
            )
            return f"Training launched (exit code {rc})"

        m_ft = re.search(r"\b(?:fine[- ]?tune|finetune)\s+quark(?:\s+with\s+(?P<path>\S+))?", t)
        if m_ft:
            local_path = m_ft.group("path") if m_ft.group("path") else None
            overrides = {"data_mode": "streaming", "bucket": "quark-main-tokyo-bucket"}
            rc = self.resource_manager.run_training_job(
                "fine_tune",
                overrides=overrides,
                dataset_local_path=local_path,
            )
            return f"Fine-tuning launched (exit code {rc})"

        return None

    # ------------------------------------------------------------------
    # Unified public API
    # ------------------------------------------------------------------
    def assimilate(self, raw: Any, *, source: str = "unknown", citation: str = "") -> List[KnowledgeObject]:
        """Convert *raw* input (text str or dataset list[dict]) into KnowledgeObjects.

        Args:
            raw: Either a string (free-form text) or an iterable of dicts
                 representing a structured dataset.
            source: Human-readable origin label (URL, file path, etc.).
            citation: Formal citation if available.

        Returns
        -------
        list[KnowledgeObject]
            One or more standardized objects ready for injection.
        """

        from collections.abc import Iterable
        from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
        rm = ResourceManager.get_default()
        if isinstance(raw, str):
            if rm:
                with rm.request_resources("nlp"):
                    return self.process_text(raw, source, citation)
            return self.process_text(raw, source, citation)

        # Heuristic: dataset must be iterable of mapping objects
        if isinstance(raw, Iterable):
            raw_list = list(raw)
            if raw_list and isinstance(raw_list[0], dict):
                if rm:
                    with rm.request_resources("nlp"):
                        return self.process_dataset(raw_list, source, citation)
                return self.process_dataset(raw_list, source, citation)

        # Unsupported type → raise descriptive error
        raise TypeError(
            "KnowledgeHub.assimilate received unsupported type "
            f"{type(raw).__name__}. Expected str or iterable of dicts."
        )

    # ------------------------------------------------------------------
    # Kaleidescope mood / drive pass-through (optional)
    # ------------------------------------------------------------------

    def set_mood(self, mood: Dict[str, float]):  # noqa: D401 – simple setter
        """Forward *mood* vector to the underlying memory adapter if supported."""
        if self._maybe_adapter and hasattr(self._maybe_adapter, "set_mood"):
            self._maybe_adapter.set_mood(mood)

    def set_drives(self, drives: Dict[str, float]):
        if self._maybe_adapter and hasattr(self._maybe_adapter, "set_drives"):
            self._maybe_adapter.set_drives(drives)
