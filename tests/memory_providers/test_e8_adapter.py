import os
import time

import pytest

# Ensure feature flag off for unit tests unless explicitly enabled via env
os.environ.pop("USE_E8_MEMORY", None)

from brain.architecture.neural_core.cognitive_systems.memory_providers.e8_adapter import (
    E8MemoryAdapter,
)


@pytest.fixture()
def adapter():
    return E8MemoryAdapter()


@pytest.mark.parametrize("n_items", [10, 100])
def test_store_and_query(adapter, n_items):
    """Adapter should round-trip store/query and preserve ranking order."""
    texts = [f"sample text {i}" for i in range(n_items)]

    for txt in texts:
        adapter.store(txt, metadata={"index": txt.split()[-1]})

    # Query existing text should return itself with highest score
    item_id, score, payload = adapter.query("sample text 5", top_k=1)[0]
    assert payload["index"] == "5"
    assert 0.9 <= score <= 1.0


def test_latency_under_threshold(adapter):
    n_items = 1000
    texts = [f"doc {i}" for i in range(n_items)]
    for txt in texts:
        adapter.store(txt, {})

    start = time.perf_counter_ns()
    _ = adapter.query("search term", top_k=5)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    # Allow generous threshold on CI but target <20 ms
    assert elapsed_ms < 50, f"Query latency too high: {elapsed_ms:.2f}ms"
