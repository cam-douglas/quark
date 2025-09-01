#!/usr/bin/env python
"""train_streaming.py – Minimal training entry-point for Quark.

Usage
-----
$ python tools_utilities/scripts/train_streaming.py \
        --config management/configurations/project/training_config.yaml \
        --override data_mode=streaming train_prefix=datasets/myset/train-

The script:
1. Loads the YAML config (OmegaConf) and key-value CLI overrides.
2. Builds a DataLoader via `brain.ml.dataset_shards.data_loader_factory.build_dataloader`.
3. Runs a toy training loop that prints batch shapes – replace with real model.
"""
from __future__ import annotations

import argparse, sys, logging, time, subprocess, json
from pathlib import Path
from typing import List

from omegaconf import OmegaConf
import torch

from brain.ml.dataset_shards.data_loader_factory import build_dataloader

# helper for progress
from tools_utilities.scripts.dataset_discovery import discover
from tqdm import tqdm


def parse_overrides(kvs: List[str]):
    """Parse key=value CLI pairs into a dict."""
    cfg = {}
    for kv in kvs:
        if "=" not in kv:
            raise ValueError(f"Override '{kv}' must be key=value")
        k, v = kv.split("=", 1)
        # attempt numeric cast
        if v.isdigit():
            v = int(v)
        cfg[k] = v
    return cfg


def main():
    p = argparse.ArgumentParser(description="Quark streaming training script")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    p.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = p.parse_args()

    base_cfg = OmegaConf.load(args.config)
    overrides = parse_overrides(args.override)
    cfg = OmegaConf.merge(base_cfg, overrides)

    loader = build_dataloader(cfg)

    total_bytes = None
    if cfg.data_mode == "streaming":
        stats = discover(cfg.bucket, depth=len(cfg.train_prefix.strip("/").split("/")))
        if cfg.train_prefix in stats:
            total_bytes = stats[cfg.train_prefix]["bytes"]

    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Streaming train") if total_bytes else None

    logger = logging.getLogger(__name__)

    try:
        for epoch in range(1):
            for batch, shard_bytes in loader:
                if pbar is not None and shard_bytes:
                    pbar.update(shard_bytes)
                # TODO: real training step
            break  # smoke-test 1 epoch
    except Exception as e:
        logger.exception("Training failed: %s", e)
        from brain.architecture.neural_core.cognitive_systems.knowledge_hub import KnowledgeHub
        KnowledgeHub(None).handle_command(f"training failed: {e}")
        raise

    if pbar is not None:
        pbar.close()

    # ---- run evaluation on val split (if available) ----
    try:
        val_prefix = cfg.get("train_prefix", "").replace("train", "val")
        eval_script = Path(__file__).with_name("eval_streaming.py")
        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--config",
            str(args.config),
            "--override",
            f"bucket={cfg.bucket}",
            "--override",
            f"train_prefix={val_prefix}",
            "--override",
            "data_mode=streaming",
        ]
        proc = subprocess.run(eval_cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            metrics = json.loads(proc.stdout.strip())
            logger.info("Evaluation metrics: %s", metrics)
        else:
            logger.warning("Evaluator failed: %s", proc.stderr)
    except Exception as e:  # noqa: BLE001
        logger.warning("Evaluator exception: %s", e)

    # ---- persist checkpoint & register ----
    run_dir = Path.cwd() / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last.pt"
    import torch  # local import avoids heavy dep at top-level; 'time' already imported globally
    torch.save({}, ckpt_path)  # TODO: replace {} with model.state_dict()
    from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
    if getattr(ResourceManager, "_DEFAULT", None):
        ResourceManager._DEFAULT.register_model_checkpoint(ckpt_path)

    print("Streaming training script completed, checkpoint stored.")


if __name__ == "__main__":
    sys.exit(main())
