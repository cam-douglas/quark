import os
import random
import string

import pytest

pytest.importorskip("pytest_benchmark")

from brain.architecture.neural_core.cognitive_systems.memory_providers.e8_adapter import (
    E8MemoryAdapter,
)

RAND = random.Random(42)


def _rand_sentence():
    words = ["".join(RAND.choices(string.ascii_lowercase, k=RAND.randint(3, 8))) for _ in range(8)]
    return " ".join(words)


@pytest.fixture(scope="module")
def adapter_and_corpus():
    os.environ["USE_E8_MEMORY"] = "false"  # run fallback path for speed
    adapter = E8MemoryAdapter()
    corpus = [_rand_sentence() for _ in range(1000)]
    for doc in corpus:
        adapter.store(doc, {})
    return adapter, corpus


def test_latency(adapter_and_corpus, benchmark):
    adapter, _ = adapter_and_corpus

    def _query():
        adapter.query("search", top_k=5)

    result = benchmark(_query)
    assert result.mean * 1000 < 20  # seconds → ms


def test_recall_top10(adapter_and_corpus):
    adapter, corpus = adapter_and_corpus
    hits = 0
    for _ in range(20):
        doc = RAND.choice(corpus)
        res = adapter.query(doc, top_k=10)
        ids = {p[2].get("text") for p in res}
        if doc in ids:
            hits += 1
    assert hits >= 15  # ≥75% recall
