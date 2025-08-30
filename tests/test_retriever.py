from brain.architecture.neural_core.cognitive_systems.knowledge_retriever import KnowledgeRetriever
from brain.architecture.neural_core.hippocampus.episodic_memory import EpisodicMemory


def _store(mem: EpisodicMemory, text: str):
    mem.store_episode(content={"text": text}, context={}, importance=0.9)


def test_retriever_rank():
    mem = EpisodicMemory(max_episodes=5)
    _store(mem, "The sky is blue.")
    _store(mem, "Grass is green.")
    _store(mem, "Roses are red.")

    retriever = KnowledgeRetriever(mem)
    retriever.build_index()

    results = retriever.retrieve("What color is the sky?", top_k=2)
    assert results, "Expected at least one result"
    top_episode, score = results[0]
    assert "sky" in top_episode.content["text"].lower()
