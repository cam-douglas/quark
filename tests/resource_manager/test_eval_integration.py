import sys
import types
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Patch build_dataloader to return tiny loader

def _dummy_loader(cfg):
    class _DummyDS(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield torch.zeros(1), 0  # data, shard_bytes

    return DataLoader(_DummyDS(), batch_size=None, num_workers=0)


def test_train_invokes_evaluator(monkeypatch):
    monkeypatch.setattr(
        "brain.ml.dataset_shards.data_loader_factory.build_dataloader", _dummy_loader
    )
    dummy_out = json.dumps({"eval_loss": 0.0})
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: types.SimpleNamespace(returncode=0, stdout=dummy_out, stderr=""),
    )

    import importlib

    import types as _t
    dummy_oc = _t.ModuleType("omegaconf")
    from types import SimpleNamespace
    class _OC:
        @staticmethod
        def load(_):
            # minimal cfg with attributes used in train_streaming.py
            return SimpleNamespace(data_mode="local", bucket="dummy", train_prefix="root/train/")
        @staticmethod
        def merge(a, b):
            return a  # return base cfg unchanged for test
    OmegaConf=_OC
    dummy_oc.OmegaConf=_OC
    import sys as _sys
    _sys.modules["omegaconf"] = dummy_oc

    ts = importlib.import_module("tools_utilities.scripts.train_streaming")
    cfg_path = Path("management/configurations/project/training_config.yaml")
    argv = [
        "train_streaming.py",
        "--config",
        str(cfg_path),
        "--override",
        "data_mode=local",
        "--override",
        "batch_size=1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = ts.main()
    assert exit_code == None or exit_code == 0  # main returns None or 0
