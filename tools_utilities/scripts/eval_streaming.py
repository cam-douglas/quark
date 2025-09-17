#!/usr/bin/env python
"""eval_streaming.py – lightweight evaluation entry-point.

This mirrors *train_streaming.py* but runs without gradients and returns a JSON
metrics blob on stdout so the caller can capture it.

Typical invocation (from train_streaming.py):

    subprocess.run([
        sys.executable, "eval_streaming.py", "--config", cfg_path,
        "--override", f"bucket={bucket}",
        "--override", f"train_prefix={val_prefix}",
        "--override", "data_mode=streaming",
    ], capture_output=True)

The script prints a one-line JSON dict, e.g.:
    {"eval_loss": 0.1234, "num_batches": 42}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import List

import torch
from omegaconf import OmegaConf

from brain.ml.dataset_shards.data_loader_factory import build_dataloader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_overrides(kvs: List[str]):
    cfg = {}
    for kv in kvs:
        if "=" not in kv:
            raise ValueError(f"Override '{kv}' must be key=value")
        k, v = kv.split("=", 1)
        if v.isdigit():
            v = int(v)
        cfg[k] = v
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(description="Quark streaming evaluation script")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = p.parse_args()

    base_cfg = OmegaConf.load(args.config)
    overrides = parse_overrides(args.override)
    cfg = OmegaConf.merge(base_cfg, overrides)

    loader = build_dataloader(cfg)
    loader_iter = iter(loader)

    torch.set_grad_enabled(False)
    total_loss = 0.0
    batches = 0

    start = time.time()
    for batch, _ in loader_iter:
        # Placeholder loss: mean squared value against zeros
        if isinstance(batch, torch.Tensor):
            loss = torch.mean(batch.float() ** 2).item()
        else:
            # Fallback: use length of batch list
            loss = len(batch)
        total_loss += loss
        batches += 1
    dur = time.time() - start

    avg_loss = total_loss / max(1, batches)
    metrics = {"eval_loss": avg_loss, "num_batches": batches, "duration_sec": dur}
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
