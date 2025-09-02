#!/usr/bin/env python
"""pipeline_orchestrator.py – Bridge between policy YAML and job scripts.

Usage (called by agent or other modules)::

    python pipeline_orchestrator.py train cloud   # or "train local"
    python pipeline_orchestrator.py finetune cloud --checkpoint runs/20250902-120000/last.pt

Natural-language synonyms (locally, streaming, etc.) are normalised the same
way the downstream job scripts expect.

Logic:
1. Load `management/configurations/project/training_pipeline.yaml`.
2. Check top-level flags like `cloud_training.enabled` or `local_training.enabled`.
3. Decide whether the requested action is allowed under current policy.
4. Spawn the corresponding entry-point script with the low-level job YAML.

This is intentionally lightweight; full KPI/feedback evaluation belongs to a
future orchestration service.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Final

import yaml

PIPELINE_YAML: Final = Path(__file__).parent.parent / "management" / "configurations" / "project" / "training_pipeline.yaml"
JOB_CONFIGS: Final = {
    "train": Path(__file__).parent.parent / "management" / "configurations" / "project" / "training_config.yaml",
    "finetune": Path(__file__).parent.parent / "management" / "configurations" / "project" / "finetune_config.yaml",
}
JOB_SCRIPTS: Final = {
    "train": Path(__file__).with_name("train_streaming.py"),
    "finetune": Path(__file__).with_name("finetune_streaming.py"),
}

_LOCAL = {"local", "locally", "localhost"}
_CLOUD = {"cloud", "stream", "streaming", "sagemaker", "remote"}


def load_policy() -> dict:  # noqa: D401
    """Load the training pipeline YAML if present, else return empty dict."""
    if PIPELINE_YAML.exists():
        with PIPELINE_YAML.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def policy_allows(action: str, backend: str) -> bool:  # noqa: D401
    """Very simple gate: honour cloud/local `enabled` flags."""
    policy = load_policy().get("training_pipeline", {})
    if backend == "cloud":
        return policy.get("cloud_training", {}).get("enabled", True)
    return policy.get("local_training", {}).get("enabled", True)


def main():  # noqa: D401
    p = argparse.ArgumentParser(description="Quark Training Pipeline Orchestrator")
    p.add_argument("action", choices=["train", "finetune"], help="Which job to launch")
    p.add_argument("backend", help="local / cloud (natural language accepted)")
    p.add_argument("--checkpoint", type=Path, help="Checkpoint path for finetune")
    p.add_argument("--deploy", action="store_true", help="Deploy endpoint after training")
    p.add_argument("--deployment", dest="deploy", action="store_true", help=argparse.SUPPRESS)
    args, extra = p.parse_known_args()

    backend_norm = args.backend.lower()
    if backend_norm in _LOCAL:
        backend = "local"
    elif backend_norm in _CLOUD:
        backend = "cloud"
    else:
        sys.exit(f"[orchestrator] Unknown backend '{args.backend}'.")

    if not policy_allows(args.action, backend):
        sys.exit(f"[orchestrator] Policy blocks {backend} {args.action} at this time.")

    script = JOB_SCRIPTS[args.action]
    cfg = JOB_CONFIGS[args.action]

    cmd = [sys.executable, str(script), "--config", str(cfg), "--backend", backend]
    if args.deploy:
        cmd.append("--deploy")
    if args.action == "finetune" and args.checkpoint:
        cmd.extend(["--checkpoint", str(args.checkpoint)])
    cmd.extend(extra)  # forward any additional overrides

    print("[orchestrator] launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":  # pragma: no cover
    main()
