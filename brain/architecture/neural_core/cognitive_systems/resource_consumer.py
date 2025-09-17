"""ResourceConsumer – reacts to resource_integrated events and feeds them into KnowledgeHub.
Currently handles plain-text files (.txt, .md) by assimilating their contents
into the brain's KnowledgeHub.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
from pathlib import Path
from typing import Dict, Any

from brain.architecture.neural_core.cognitive_systems.callback_hub import hub
from brain.architecture.neural_core.cognitive_systems.knowledge_hub import KnowledgeHub

class ResourceConsumer:
    def __init__(self, kh: KnowledgeHub):
        self.kh = kh
        hub.register(self._on_event)

    # --------------------------------------------------
    def _on_event(self, event: str, data: Dict[str, Any]):
        if event != "resource_integrated":
            return
        path = Path(data.get("path", ""))
        if not path.exists():
            return
        if path.suffix in {".txt", ".md"}:
            try:
                text = path.read_text(errors="ignore")
                objs = self.kh.assimilate(text, source=str(path), citation="auto-import")
                # For now just log number of objects assimilated
                import logging
                logging.getLogger("quark.resource_consumer").info("Assimilated %d objects from %s", len(objs), path)
            except Exception as e:
                import logging
                logging.getLogger("quark.resource_consumer").warning("Failed to assimilate %s: %s", path, e)
