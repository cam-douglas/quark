"""
Prometheus metrics for brainstem segmentation.

Exports latency, run counters, and Dice drift gauges.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server


_SERVER_STARTED = False
_LOCK = threading.Lock()


# Metrics
SEGMENTATION_RUNS = Counter(
    "brainstem_segmentation_runs_total",
    "Total number of brainstem segmentation runs",
)

SEGMENTATION_SUCCESS = Counter(
    "brainstem_segmentation_success_total",
    "Number of successful brainstem segmentation runs",
)

SEGMENTATION_LATENCY = Histogram(
    "brainstem_segmentation_latency_seconds",
    "Latency of brainstem segmentation inference in seconds",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

OVERALL_DICE = Gauge(
    "brainstem_segmentation_overall_dice",
    "Measured overall Dice against manual spot-check labels",
)

DICE_DRIFT = Gauge(
    "brainstem_segmentation_dice_drift",
    "Positive drift = target_overall_dice - measured_overall_dice",
)


def init_metrics_server(port: Optional[int] = None) -> None:
    """Start Prometheus metrics HTTP server if not already started.

    Port precedence: arg > env `BRAINSTEM_METRICS_PORT` > 9109.
    """
    global _SERVER_STARTED
    if _SERVER_STARTED:
        return
    with _LOCK:
        if _SERVER_STARTED:
            return
        resolved_port = port or int(os.environ.get("BRAINSTEM_METRICS_PORT", "9109"))
        start_http_server(resolved_port)
        _SERVER_STARTED = True


def record_inference(latency_seconds: float, success: bool) -> None:
    """Record a segmentation inference event."""
    SEGMENTATION_RUNS.inc()
    SEGMENTATION_LATENCY.observe(latency_seconds)
    if success:
        SEGMENTATION_SUCCESS.inc()


def record_dice(measured_overall: float, target_overall: float) -> None:
    """Publish Dice metrics and drift."""
    OVERALL_DICE.set(measured_overall)
    drift = max(0.0, target_overall - measured_overall)
    DICE_DRIFT.set(drift)


