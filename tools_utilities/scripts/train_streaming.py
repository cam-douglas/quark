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

import argparse
import sys
import os
import logging
import time
import subprocess
import json
from pathlib import Path
from typing import List

from omegaconf import OmegaConf
import torch

# Quark AWS helpers
from tools_utilities.scripts.aws_utils import ensure_iam_role, ensure_ecr_image, default_image_uri

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
    # New: backend selection flag accepting natural-language synonyms.
    p.add_argument(
        "--backend",
        type=str,
        default="local",
        help=(
            "Execution backend: accepts synonyms such as 'local', 'locally' for in-process "
            "execution, or 'cloud', 'stream', 'streaming', 'sagemaker' for remote SageMaker "
            "dispatch."
        ),
    )
    p.add_argument(
        "--role-arn",
        type=str,
        default=os.getenv("QUARK_SAGEMAKER_ROLE_ARN"),
        help="IAM role ARN for SageMaker jobs (required when backend is cloud).",
    )
    p.add_argument(
        "--deploy-endpoint",
        action="store_true",
        help="Deploy inference endpoint after training. Natural-language synonyms allowed via --deploy flag.",
    )
    args = p.parse_args()

    # Natural-language toggle for deploy flag ("deploy", "deployment")
    if any(t in sys.argv for t in ["--deploy", "--deployment"]):
        args.deploy_endpoint = True

    # Normalise backend synonyms → {"local"|"cloud"}
    _LOCAL = {"local", "locally", "localhost"}
    _CLOUD = {"cloud", "stream", "streaming", "sagemaker", "remote"}
    backend_norm = args.backend.lower()
    if backend_norm in _LOCAL:
        backend = "local"
    elif backend_norm in _CLOUD:
        backend = "cloud"
    else:
        raise ValueError(
            f"Unrecognised --backend '{args.backend}'. Use one of: {sorted(_LOCAL | _CLOUD)}"
        )

    base_cfg = OmegaConf.load(args.config)
    overrides = parse_overrides(args.override)
    cfg = OmegaConf.merge(base_cfg, overrides)

    # ------------------------------------------------------------------
    # Auto-discover train_prefix when left blank in YAML
    # ------------------------------------------------------------------
    if not cfg.get("train_prefix"):
        if cfg.data_mode == "streaming":
            from tools_utilities.scripts.dataset_discovery import discover_shard_groups  # noqa: WPS433

            stats = discover_shard_groups(cfg.bucket, root_prefix="", min_objects=4)
            candidate = next((p for p in stats if p.endswith("train-/")), None)
            if candidate is None and stats:
                candidate = sorted(stats.keys())[0]
            if candidate:
                cfg.train_prefix = candidate
                print(f"[train_streaming] Auto-discovered train_prefix={candidate}")
            else:
                raise RuntimeError("Could not auto-discover train_prefix in bucket")
        else:
            # local filesystem search
            import glob

            matches = glob.glob("data/**/train-*", recursive=True)
            if matches:
                # pick directory part ending with 'train-'
                cfg.train_prefix = os.path.dirname(matches[0]) + "/"
                print(f"[train_streaming] Auto-discovered local train_prefix={cfg.train_prefix}")
            else:
                raise RuntimeError("No local train- shards found under data/**")

    if backend == "cloud":
        run_sagemaker_training(args, cfg)
        return

    # ---- Local backend (default) ----
    # Adaptive batch size for local Mac (M2 Max) to fit RAM/VRAM
    try:
        import psutil  # noqa: WPS433
    except ImportError:
        psutil = None  # type: ignore

    def _adjust_batch(cfg):  # noqa: D401, ANN001
        if backend != "local":
            return cfg  # cloud nodes fixed
        avail_bytes: int
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            avail_bytes = int(props.total_memory * 0.8)  # 80 % of VRAM
        elif psutil:
            avail_bytes = int(psutil.virtual_memory().available * 0.8)
        else:
            return cfg  # cannot introspect; keep as is

        seq_len = int(cfg.get("seq_len", 1024))
        hid = int(cfg.get("hidden_size", 768))
        bsz = int(cfg.get("batch_size", 16))

        est_bytes = bsz * seq_len * hid * 4  # float32 activations
        while est_bytes > avail_bytes and bsz > 1:
            bsz //= 2
            est_bytes = bsz * seq_len * hid * 4
        if bsz != cfg.batch_size:
            print(f"[train_streaming] Reducing batch_size to {bsz} to fit memory budget")
            cfg.batch_size = bsz
        return cfg

    cfg = _adjust_batch(cfg)

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
    torch.save({}, ckpt_path)  # TODO: replace {} with model.state_dict()
    from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
    if getattr(ResourceManager, "_DEFAULT", None):
        ResourceManager._DEFAULT.register_model_checkpoint(ckpt_path)

    print("Streaming training script completed, checkpoint stored.")


if __name__ == "__main__":
    sys.exit(main())

# ---------------------------------------------------------------------------
# Cloud-backend helpers
# ---------------------------------------------------------------------------

def run_sagemaker_training(args: argparse.Namespace, cfg):  # noqa: D401, ANN001
    """Dispatch a SageMaker training job using the provided config.

    This function keeps a minimal dependency footprint: it only imports boto3/
    sagemaker inside the body so local users without those packages aren't affected.
    """
    try:
        import time, json, boto3  # noqa: WPS433, E401  # pylint: disable=import-error
        from sagemaker.estimator import Estimator  # type: ignore
        import sagemaker  # noqa: WPS433
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Cloud backend requires 'boto3' and 'sagemaker' packages. "
            "Install them via `pip install boto3 sagemaker` or choose --backend local."
        ) from exc

    # Auto create role if not provided
    role_arn = args.role_arn or ensure_iam_role()

    sess = sagemaker.Session()

    # Build image URI – assume caller has already pushed container, else fall back to
    # SageMaker's PyTorch image matching the Python & CUDA versions.
    region = sess.boto_region_name
    account = boto3.client("sts").get_caller_identity()["Account"]
    image_uri = cfg.get("image_uri") or ensure_ecr_image() or default_image_uri()

    est = Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_count=cfg.get("sm_instances", 1),
        instance_type=cfg.get("sm_instance_type", "ml.g5.2xlarge"),
        max_run=int(cfg.get("sm_max_run", 60 * 60 * 8)),  # seconds
        hyperparameters=cfg.get("hyperparameters", {}),
        output_path=cfg.get("sm_output", f"s3://quark-artifacts/{int(time.time())}"),
        sagemaker_session=sess,
    )

    est.fit(job_name=f"quark-train-{int(time.time())}", wait=False)

    print(
        json.dumps(
            {
                "status": "submitted",
                "training_job_name": est.latest_training_job.name,
                "region": region,
            },
            indent=2,
        )
    )
    print("Use AWS Console or `aws sagemaker describe-training-job` to monitor progress.")

    # Optionally deploy endpoint when job completes (fire-and-forget)
    if args.deploy_endpoint:
        from sagemaker.model import Model  # type: ignore

        model_artifact = est.output_path + "/output/model.tar.gz"
        mdl = Model(image_uri=image_uri, model_data=model_artifact, role=role_arn, sagemaker_session=sess)
        mdl.deploy(initial_instance_count=1, instance_type="ml.g5.xlarge", endpoint_name="quark-brain-endpoint")
        print("Endpoint 'quark-brain-endpoint' deployed.")
