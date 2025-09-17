import os

from brain.architecture.neural_core.hippocampus.episodic_memory import EpisodicMemory


def test_store_retrieve_with_e8(monkeypatch):
    os.environ["USE_E8_MEMORY"] = "true"
    mem = EpisodicMemory(max_episodes=10)

    ep_id = mem.store_episode({"text": "hello world"}, context={})
    results = mem.retrieve_episode("hello world", max_results=1)
    assert results and results[0].episode_id == ep_id
