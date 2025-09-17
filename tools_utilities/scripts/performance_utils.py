

"""
Utility helpers for lightweight performance profiling.

This module introduces two minimal-overhead helpers that add **wall-clock**
latency measurements around critical functions without pulling in heavy
profilers or restructuring code.

All timing uses `time.perf_counter()` which provides sub-microsecond
resolution on macOS.

Usage
-----

```python
from tools_utilities.scripts.performance_utils import profile_timing

@profile_timing("agent.execute_next_goal")
def execute_next_goal(...):
    ...
```

A global, in-memory registry (`TIMING_REGISTRY`) aggregates the cumulative
stats so callers can print a summary with `print_timing_breakdown()` after a
pipeline finishes.

The decorator intentionally adds **<100 µs** overhead per call so it’s safe to
leave enabled.
"""
from __future__ import annotations

import time
import functools
import threading
from typing import Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Internal timing registry (thread-safe via simple lock as the writes are tiny)
# ---------------------------------------------------------------------------
_TIMING_LOCK = threading.Lock()
TIMING_REGISTRY: Dict[str, List[float]] = {}


def _record(name: str, duration: float) -> None:
    """Record `duration` seconds under the key *name*."""
    with _TIMING_LOCK:
        TIMING_REGISTRY.setdefault(name, []).append(duration)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def profile_timing(name: str | None = None) -> Callable[[Callable[..., "T"]], Callable[..., "T"]]:  # type: ignore[name-defined]
    """Decorator that records wall-clock execution time of the wrapped function.

    Parameters
    ----------
    name
        Optional name to store in the registry. If *None*, the wrapped
        function’s ``__qualname__`` is used.
    """

    def decorator(func: Callable[..., "T"]):  # type: ignore[name-defined]
        label = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[override]
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                _record(label, time.perf_counter() - start)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def get_timing_stats() -> List[Tuple[str, int, float, float]]:
    """Return a list of (name, calls, total_seconds, avg_ms_per_call)."""
    with _TIMING_LOCK:
        items = list(TIMING_REGISTRY.items())
    stats: List[Tuple[str, int, float, float]] = []
    for name, durations in items:
        total = sum(durations)
        calls = len(durations)
        avg_ms = (total / calls) * 1000 if calls else 0.0
        stats.append((name, calls, total, avg_ms))
    # Sort by total time descending
    stats.sort(key=lambda x: x[2], reverse=True)
    return stats


def print_timing_breakdown(limit: int | None = None) -> None:
    """Pretty-print the timing breakdown to stdout.

    Parameters
    ----------
    limit
        If given, only print the top *limit* functions by total time.
    """
    stats = get_timing_stats()
    if limit is not None:
        stats = stats[:limit]

    print("\n=== ⏱️  Timing Breakdown ===")
    for name, calls, total, avg_ms in stats:
        print(f"{name:50s}  calls={calls:3d}  total={total:6.3f}s  avg={avg_ms:6.2f}ms")
    print("===========================\n")

# ---------------------------------------------------------------------------
# Cached file reads (Path.read_text wrapper with mtime check)
# ---------------------------------------------------------------------------

from pathlib import Path

_FILE_CACHE: Dict[tuple, str] = {}


def read_text_cached(path: str | Path, encoding: str = "utf-8") -> str:
    """Return text contents of *path* with an mtime-aware cache.

    The cache key is (absolute_path, mtime_ns). If the file changes on disk the
    mtime key changes, forcing a fresh read. Safe for multi-thread use because
    each read is atomic and text length is small.
    """
    p = Path(path).expanduser().resolve()
    try:
        mtime = p.stat().st_mtime_ns
    except FileNotFoundError:
        raise
    cache_key = (str(p), mtime)
    if cache_key in _FILE_CACHE:
        return _FILE_CACHE[cache_key]
    text = p.read_text(encoding=encoding, errors="ignore")
    _FILE_CACHE[cache_key] = text
    return text


# ---------------------------------------------------------------------------
# Lightweight in-memory memoization (function-level)
# ---------------------------------------------------------------------------

def memoize(maxsize: int | None = 1024):
    """Very small LFU-ish memoization decorator for pure functions.

    Designed for caching remote/model calls where identical inputs repeat
    within a session. Evicts the least-recently used entry when the cache
    exceeds *maxsize* (if provided).
    """

    def decorator(func: Callable[..., "T"]):  # type: ignore[name-defined]
        cache: Dict[tuple, "T"] = {}
        access_order: List[tuple] = []  # simple LRU tracking

        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[override]
            key = args + tuple(sorted(kwargs.items()))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            access_order.append(key)
            # Evict if needed
            if maxsize is not None and len(cache) > maxsize:
                oldest = access_order.pop(0)
                cache.pop(oldest, None)
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Maintenance helpers
# ---------------------------------------------------------------------------


def reset_timing_registry() -> None:
    """Clear timing stats (useful between prompts to avoid cross-contamination)."""
    with _TIMING_LOCK:
        TIMING_REGISTRY.clear()
