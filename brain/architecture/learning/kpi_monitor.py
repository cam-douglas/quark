"""kpi_monitor.py – Lightweight runtime KPI monitor for Quark.

Purpose
-------
Track rolling metrics during simulation and decide *when* the system should
launch a training or fine-tuning job.  The actual launch is delegated to
`ResourceManager.run_training_job()` in Phase-2 wiring.

Design goals
============
* Zero heavy dependencies – pure Python + PyYAML only.
* Non-blocking – computations are O(1) per update.
* Config-driven – thresholds live in `training_pipeline.yaml`.
* Thread-safe singleton – accessed from multiple simulation threads.

API
---
>>> from brain.architecture.learning.kpi_monitor import kpi_monitor
>>> kpi_monitor.update({"reward": 0.7, "loss": 1.23})
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]  # /quark
_PIPELINE_YAML = (
    _REPO_ROOT / "management" / "configurations" / "project" / "training_pipeline.yaml"
)

_DEFAULT_THRESHOLDS = {
    "reward": 0.9,      # high reward = good; below triggers training
    "loss": 0.5,        # low loss good; above triggers training
}


class _KpiMonitor:  # noqa: D401 – internal singleton
    """Rolling KPI monitor with simple threshold checking."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Deque[float]] = {}
        self._window: int = 1000  # last N steps
        self._lock = Lock()
        self.thresholds = _DEFAULT_THRESHOLDS.copy()
        self._load_pipeline_thresholds()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def update(self, metrics: Dict[str, float]) -> Optional[str]:  # noqa: D401
        """Update rolling metrics.

        Parameters
        ----------
        metrics : dict
            e.g. {"reward": 0.8, "loss": 1.2}

        Returns
        -------
        Optional[str]
            "train" / "finetune" if action recommended, else ``None``.
        """
        with self._lock:
            for k, v in metrics.items():
                dq = self._metrics.setdefault(k, deque(maxlen=self._window))
                dq.append(float(v))

            # simple heuristic: if avg reward < threshold or avg loss > threshold
            avg_reward = self._avg("reward")
            avg_loss = self._avg("loss")

            if avg_reward is not None and avg_reward < self.thresholds["reward"]:
                return "train"
            if avg_loss is not None and avg_loss > self.thresholds["loss"]:
                return "finetune"
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _avg(self, key: str) -> Optional[float]:
        dq = self._metrics.get(key)
        if not dq:
            return None
        return sum(dq) / len(dq)

    def _load_pipeline_thresholds(self) -> None:  # noqa: D401
        if not _PIPELINE_YAML.exists():
            return
        try:
            data = yaml.safe_load(_PIPELINE_YAML.read_text()) or {}
            tp = data.get("training_pipeline", {})
            training_triggers = tp.get("training_triggers", {})
            auto = training_triggers.get("automatic", {})
            thr = auto.get("thresholds", {})
            if "quality_threshold" in thr:
                # map quality_threshold to reward threshold inversely
                self.thresholds["reward"] = float(thr["quality_threshold"])
            if "performance_threshold" in thr:
                self.thresholds["loss"] = float(1.0 - thr["performance_threshold"])
        except Exception:  # pragma: no cover – safe default
            pass


# Global singleton instance
kpi_monitor = _KpiMonitor()
