"""data_loader_factory – creates a PyTorch DataLoader based on `cfg.data_mode`.

If `cfg.data_mode == "streaming"`, it builds a `StreamDataset` that streams
shards from S3.  Otherwise it expects a local dataset directory compatible
with `torchvision.datasets.ImageFolder` or any `torch.utils.data.Dataset`
provided via `cfg.local_dataset_cls`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib

from torch.utils.data import DataLoader

from tools_utilities.scripts.s3_streaming_manager import StreamingManager
from brain.ml.dataset_shards.stream_dataset import StreamDataset


def _resolve(name: str):
    """Import dotted path like ``module.sub:Class`` or ``module.sub.Class``."""
    if ":" in name:
        module, attr = name.split(":", 1)
    else:
        module, attr = name.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)


def build_dataloader(cfg: Any) -> DataLoader:
    """Return DataLoader according to cfg.

    Expected cfg attributes / keys:
    - data_mode: "local" | "streaming"
    - batch_size, num_workers
    - For streaming:
        • bucket, train_prefix, deserialize_fn (dotted path)
    - For local:
        • local_data_root, local_dataset_cls (dotted path, default torchvision.datasets.ImageFolder)
    """
    if isinstance(cfg, dict):
        get = cfg.get
    else:
        get = lambda k, d=None: getattr(cfg, k, d)

    batch_size = get("batch_size", 32)
    num_workers = get("num_workers", 4)

    if get("data_mode") == "streaming":
        bucket = get("bucket", "quark-main-tokyo-bucket")
        prefix = get("train_prefix", "datasets/")
        deserialize_path = get("deserialize_fn", "json:loads")
        deserialize_fn = _resolve(deserialize_path)
        sm = StreamingManager(bucket=bucket)
        ds = StreamDataset(sm, prefix=prefix, deserialize=deserialize_fn)
        shuffle = False  # IterableDataset cannot be shuffled by DataLoader
        num_workers = 0  # avoid pickling StreamingManager / boto3 client
    else:
        root = Path(get("local_data_root", "data/"))
        dataset_cls_path = get("local_dataset_cls", "torchvision.datasets:ImageFolder")
        dataset_cls = _resolve(dataset_cls_path)
        ds = dataset_cls(root)
        shuffle = True

    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
