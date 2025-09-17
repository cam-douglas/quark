from brain.architecture.neural_core.hippocampus.episodic_memory import EpisodicMemory
from brain.architecture.neural_core.learning.long_term_memory import LongTermMemory
from brain.architecture.neural_core.memory.episodic_store import EpisodicMemoryStore
from brain.architecture.neural_core.memory.longterm_store import LongTermMemoryStore
from brain.architecture.neural_core.memory.memory_synchronizer import MemorySynchronizer


def test_sync_forwarding():
    epi_store = EpisodicMemoryStore(EpisodicMemory(max_episodes=10))
    ltm_store = LongTermMemoryStore(LongTermMemory())
    sync = MemorySynchronizer(epi_store, ltm_store)

    # initial stats
    s0 = sync.sync()
    assert s0["forwarded"] == 0

    # add episode then sync
    eid = epi_store.store_episode({"text": "Running test."}, {})
    s1 = sync.sync()
    assert s1["forwarded"] == 1
    # second sync idempotent
    s2 = sync.sync()
    assert s2["forwarded"] == 0
    assert ltm_store.ltm.total_experiences == 1
