from brain.architecture.neural_core.hippocampus.episodic_memory import EpisodicMemory
from brain.architecture.neural_core.memory.episodic_store import EpisodicMemoryStore


def test_episode_roundtrip(tmp_path):
    mem = EpisodicMemory(max_episodes=10)
    store = EpisodicMemoryStore(mem)

    ep_id = store.store_episode({"text": "Hello world"}, {"src": "unit"})
    assert ep_id
    assert store.get_stats()["total_episodes"] == 1

    # Persist & reload
    p = tmp_path / "epis.json.gz"
    store.save(str(p))

    new_mem = EpisodicMemory(max_episodes=10)
    new_store = EpisodicMemoryStore(new_mem)
    new_store.load(str(p))
    assert new_store.get_stats()["total_episodes"] == 1

    res = new_store.retrieve_episode({"text": "Hello"}, max_results=1)
    assert res and res[0].content["text"].startswith("Hello")
