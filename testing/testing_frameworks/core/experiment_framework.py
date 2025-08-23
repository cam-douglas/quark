#!/usr/bin/env python3
"""
Core Experiment Framework for QUARK testing suite.

Purpose: Provide minimal, stable interfaces for defining and running experiments
within the testing framework, so downstream scripts can import without fallbacks.

Inputs: ExperimentConfig dataclass instances
Outputs: ExperimentResult dataclass instances with metrics and artifacts
Seeds: None (determinism handled by callers)
Deps: Standard library only; keep this module lightweight
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import time


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run.

    Includes FAIR-style metadata so every run is *findable* and *re-usable*.
    """

    # Core identifiers
    experiment_id: str  # human-readable ID
    version: str = "0.1.0"  # semantic version for config schema

    # Descriptive metadata
    description: str = ""
    authors: List[str] = field(default_factory=list)
    created_by: str = "quark-ci"

    # Runtime parameters & tags
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    # Reproducibility helpers
    git_commit: str | None = None  # populated at runtime
    param_hash: str | None = None  # SHA-256 hash of params for quick lookup

    def finalise(self) -> None:
        """Populate *git_commit* and *param_hash* just before execution."""
        import hashlib, subprocess, json, os

        # Get current git commit (if repo present)
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd())
                .decode()
                .strip()
            )
            self.git_commit = commit
        except Exception:
            self.git_commit = "unknown"

        # Stable hash of params
        param_bytes = json.dumps(self.params, sort_keys=True).encode()
        self.param_hash = hashlib.sha256(param_bytes).hexdigest()[:12]


@dataclass
class PerformanceMetrics:
    """Lightweight performance metrics container."""
    metrics: Dict[str, float] = field(default_factory=dict)

    def record(self, name: str, value: float) -> None:
        self.metrics[name] = float(value)


@dataclass
class ExperimentResult:
    """Canonical experiment result payload."""
    experiment_id: str
    started_at_s: float
    finished_at_s: float
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class BaseExperiment:
    """Base class for QUARK experiments.

    Subclasses should override run() and may use _now() for timing.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.performance = PerformanceMetrics()
        self._started_at_s: Optional[float] = None
        self._finished_at_s: Optional[float] = None

    def _now(self) -> float:
        return time.time()

    def run(self) -> ExperimentResult:
        """Run the experiment. Subclasses must implement."""
        raise NotImplementedError


class HybridSLMLLMExperiment(BaseExperiment):
    """Minimal hybrid SLM+LLM experiment placeholder.

    This is a stub to satisfy import requirements. It measures trivial
    timings and returns a successful result; real logic should be
    implemented in domain-specific experiments.
    """

    def run(self) -> ExperimentResult:
        self._started_at_s = self._now()

        # Minimal timing to populate metrics
        t0 = self._now()
        _ = sum(range(10_000))  # lightweight CPU work
        t1 = self._now()
        self.performance.record("cpu_warmup_s", t1 - t0)

        # Compose result
        self._finished_at_s = self._now()
        return ExperimentResult(
            experiment_id=self.config.experiment_id,
            started_at_s=self._started_at_s,
            finished_at_s=self._finished_at_s,
            success=True,
            metrics=self.performance.metrics,
            artifacts={},
            notes="HybridSLMLLMExperiment stub completed successfully",
        )


class ExperimentManager:
    """Tiny helper to run experiments with a unified API."""

    def __init__(self):
        self.history: List[ExperimentResult] = []
        self._live_server_started = False

    def _ensure_live_server(self):
        """Use existing live streaming server instead of starting a new one."""
        if self._live_server_started:
            return
            
        try:
            # Import live streaming components
            import sys
            import os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
            
            from testing.visualizations.visual_utils import live_series
            
            # Don't start a new server, just use the existing one
            self._live_server_started = True
            
            # Stream experiment manager start
            live_series("experiment_manager", "started", 0)
            
        except Exception as e:
            print(f"Warning: Live streaming not available: {e}")
            self._live_server_started = False

    def _stream_metric(self, metric_name: str, value, step: int = 0):
        """Stream a metric to the live dashboard."""
        if not self._live_server_started:
            return
            
        try:
            import sys
            import os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
            
            from testing.visualizations.visual_utils import live_series
            live_series(f"exp_{metric_name}", value, step)
            
        except Exception:
            pass  # Non-fatal

    def run(self, experiment: BaseExperiment) -> ExperimentResult:
        # Start live streaming server
        self._ensure_live_server()
        
        # Stream experiment start
        self._stream_metric("start", experiment.config.experiment_id, 0)
        self._stream_metric("status", "running", 0)
        
        # Finalise config with FAIR metadata if attribute exists
        if hasattr(experiment.config, "finalise"):
            experiment.config.finalise()  # type: ignore[attr-defined]

        # Stream experiment parameters
        for key, value in experiment.config.params.items():
            self._stream_metric(f"param_{key}", value, 0)

        result = experiment.run()
        self.history.append(result)

        # Stream experiment completion
        self._stream_metric("status", "completed", 0)
        self._stream_metric("success", float(result.success), 0)
        self._stream_metric("duration", result.finished_at_s - result.started_at_s, 0)
        
        # Stream all result metrics
        for metric_name, metric_value in result.metrics.items():
            self._stream_metric(f"result_{metric_name}", metric_value, 0)

        # --- MLflow logging (best-effort, non-fatal) ---
        try:
            import mlflow

            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("Quark Experiments")

            with mlflow.start_run(run_name=experiment.config.experiment_id):
                # params
                mlflow.log_params(experiment.config.params)
                mlflow.log_param("version", experiment.config.version)
                if experiment.config.git_commit:
                    mlflow.log_param("git_commit", experiment.config.git_commit)
                if experiment.config.param_hash:
                    mlflow.log_param("param_hash", experiment.config.param_hash)

                # metrics
                for k, v in result.metrics.items():
                    mlflow.log_metric(k, v)
                mlflow.log_metric("success", float(result.success))

                # artifact – dump result JSON
                import json, tempfile, pathlib

                tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
                json.dump(result.__dict__, tmp)
                tmp.close()
                mlflow.log_artifact(tmp.name, artifact_path="result")
                pathlib.Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            # Do not fail experiment if MLflow unavailable
            pass

        return result

    def __del__(self):
        """Cleanup live streaming on destruction."""
        # No server to cleanup since we use the existing one
        pass


