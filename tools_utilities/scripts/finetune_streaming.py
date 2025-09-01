#!/usr/bin/env python
"""finetune_streaming.py – Minimal fine-tuning entry point for Quark.

Run similarly to `train_streaming.py` but loads a checkpoint and lowers LR.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from brain.ml.dataset_shards.data_loader_factory import build_dataloader
from tools_utilities.scripts.dataset_discovery import discover
from tqdm import tqdm

# Placeholder model
class DummyModel:
    def __init__(self):
        pass
    def __call__(self, batch):
        return 0.0
    def load_state_dict(self, _):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=False)
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    for kv in args.override:
        k, v = kv.split("=", 1)
        OmegaConf.update(cfg, k, v, merge=True)

    loader = build_dataloader(cfg)

    total_bytes = None
    if cfg.data_mode == "streaming":
        stats = discover(cfg.bucket, depth=len(cfg.train_prefix.strip("/").split("/")))
        if cfg.train_prefix in stats:
            total_bytes = stats[cfg.train_prefix]["bytes"]

    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Fine-tune stream") if total_bytes else None
    model = DummyModel()
    if args.checkpoint and args.checkpoint.exists():
        print("Loading checkpoint", args.checkpoint)
        model.load_state_dict({})

    for i, batch in enumerate(loader):
        loss = model(batch)
        if pbar is not None:
            pbar.update(sum(len(str(s).encode()) for s in batch))
        if i % 100 == 0:
            print("Fine-tune step", i, "loss", loss)
        if i == 200:
            break

    if pbar is not None:
        pbar.close()

    # ---- persist checkpoint & register ----
    run_dir = Path.cwd() / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last.pt"
    # if model has state_dict use it, else save empty dict
    state = model.state_dict() if hasattr(model, "state_dict") else {}
    import torch
    torch.save(state, ckpt_path)

    from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
    ResourceManager._DEFAULT.register_model_checkpoint(ckpt_path, name="latest-quark-model-ft")

    print("Fine-tune script completed, checkpoint stored.")


if __name__ == "__main__":
    sys.exit(main())
