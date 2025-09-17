#!/usr/bin/env python
"""quark_cli.py – Natural-language wrapper to run training/fine-tuning.

Examples
--------
$ python tools_utilities/scripts/quark_cli.py "train quark" \
        --config management/configurations/project/training_config.yaml \
        --override data_mode=streaming train_prefix=datasets/myset/train-

$ python tools_utilities/scripts/quark_cli.py "finetune quark" \
        --config configs/finetune.yaml

Any phrase containing **train quark** launches the streaming trainer, while
phrases containing **finetune quark** launch the fine-tuning script.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
TRAIN_SCRIPT = ROOT / "tools_utilities/scripts/train_streaming.py"
FINETUNE_SCRIPT = ROOT / "tools_utilities/scripts/finetune_streaming.py"

# simple alias table – extend as needed
BUCKET_ALIAS = {
    "tokyo bucket": "quark-main-tokyo-bucket",
    "main bucket": "quark-main-tokyo-bucket",
}

def resolve_bucket(name: str) -> str:
    key = name.lower().strip()
    return BUCKET_ALIAS.get(key, key)

def _ensure_dataset_overrides(forwarded):
    """If bucket or train_prefix missing, prompt user via stdin."""
    kv = dict(arg.split("=", 1) for arg in forwarded if "=" in arg)
    if "bucket" not in kv:
        bucket_alias = input("Which bucket? (e.g. 'tokyo bucket'): ").strip()
        kv["bucket"] = resolve_bucket(bucket_alias)
    if "train_prefix" not in kv:
        from tools_utilities.scripts.dataset_discovery import discover_shard_groups, discover
        root = kv.get("train_prefix", "")
        # Auto-detect shard groups when root path ends with slash (directory) and not specific shard
        shard_stats = discover_shard_groups(kv["bucket"], root, min_size_kb=1) if root else {}
        if shard_stats:
            stats = shard_stats
        else:
            stats = discover(kv["bucket"], depth=3, root_prefix=root)
        if not stats:
            raise SystemExit("No datasets found in bucket")
        print("\nAvailable prefixes:")
        for i, p in enumerate(sorted(stats)):
            size_mb = stats[p]["bytes"] / 1_048_576
            print(f"[{i}] {p}  ({size_mb:.1f} MB, {stats[p]['objects']} shards)")
        choice = input("Select dataset index: ").strip()
        try:
            idx = int(choice)
            kv["train_prefix"] = list(sorted(stats))[idx]
        except Exception:
            raise SystemExit("Invalid selection")
    # rebuild forwarded list
    cleaned = [f"{k}={v}" for k, v in kv.items()] + [a for a in forwarded if "=" not in a]
    return cleaned


def main():
    p = argparse.ArgumentParser(description="Natural-language Quark CLI")
    p.add_argument("phrase", type=str, help="Command phrase, e.g. 'train quark'")
    p.add_argument("remain", nargs=argparse.REMAINDER, help="Args forwarded to underlying script")
    args = p.parse_args()

    phrase_lower = args.phrase.lower()
    forwarded = args.remain
    forwarded = _ensure_dataset_overrides(forwarded)

    # collapse multiple --override flags into one
    if forwarded.count("--override") > 1:
        kv_pairs = [arg for arg in forwarded if arg != "--override"]
        forwarded = ["--override"] + kv_pairs

    # inject default config yaml if none supplied
    if "--config" not in forwarded:
        default_cfg = str(ROOT / "management/configurations/project/training_config.yaml")
        forwarded = ["--config", default_cfg] + forwarded

    if re.search(r"\btrain\s+quark\b", phrase_lower):
        _run(TRAIN_SCRIPT, forwarded)
    elif re.search(r"\bfinetune\s+quark\b", phrase_lower):
        _run(FINETUNE_SCRIPT, forwarded)
    elif re.search(r"\bstate\b", phrase_lower):
        from tools_utilities.scripts import check_quark_state as _cqs  # lazy import
        _cqs.extract_state_info()
    elif re.search(r"\brecommendations?\b", phrase_lower):
        from state.quark_state_system.quark_recommendations import QuarkRecommendationsEngine  # noqa: WPS433
        eng = QuarkRecommendationsEngine()
        eng._refresh_state()
        for t in eng.next_tasks:
            print("•", t["title"])
    else:
        sys.exit("Unrecognised command phrase. Use 'train quark', 'finetune quark', 'state', or 'recommendations'.")


def _run(script: Path, forwarded):
    cmd = [sys.executable, str(script)] + forwarded
    print("▶", " ".join(shlex.quote(str(c)) for c in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
