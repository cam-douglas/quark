from brain.architecture.neural_core.hippocampus.episodic_memory import EpisodicMemory
from brain.architecture.neural_core.learning.long_term_memory import LongTermMemory
from brain.architecture.neural_core.memory.episodic_store import EpisodicMemoryStore
from brain.architecture.neural_core.memory.longterm_store import LongTermMemoryStore
from brain.architecture.neural_core.memory.persistence_manager import MemoryPersistenceManager


def test_persistence_roundtrip(tmp_path, monkeypatch):
    # Redirect state/memory to temp dir
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    # Build stores and manager
    epi_store = EpisodicMemoryStore(EpisodicMemory(max_episodes=10))
    ltm_store = LongTermMemoryStore(LongTermMemory())
    mgr = MemoryPersistenceManager({"epi": epi_store, "ltm": ltm_store}, auto_register_atexit=False)

    # Store episode and sync counts
    eid = epi_store.store_episode({"text": "Persistence test"}, {})
    # Save
    mgr.save_all()

    # Fresh stores and manager
    epi2 = EpisodicMemoryStore(EpisodicMemory(max_episodes=10))
    ltm2 = LongTermMemoryStore(LongTermMemory())
    mgr2 = MemoryPersistenceManager({"epi": epi2, "ltm": ltm2}, auto_register_atexit=False)
    mgr2.load_all()

    # Episode should exist after load
    episodes = epi2.retrieve_episode({"text": "Persistence"}, max_results=1)
    assert episodes, "Loaded store should contain the saved episode"
    assert "Persistence test" in episodes[0].content["text"]
