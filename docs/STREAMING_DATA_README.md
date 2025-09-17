# Streaming Dataset Integration

This document describes how Quark streams lightweight dataset **shards** directly
from S3 during training and fine-tuning.

---
## 1. Architecture Overview

```
S3 bucket ───► StreamingManager (download + LRU cache) ───► StreamDataset ───► DataLoader ───► Trainer
```
* **StreamingManager** (`utilities/s3_streaming_manager.py`)
  * Synchronous download with on-disk LRU cache (default 20 GB).
  * Optional async *prefetch* to overlap network with GPU.
* **StreamDataset** (`brain/ml/dataset_shards/stream_dataset.py`)
  * PyTorch `IterableDataset` that lists shard keys under a prefix and yields
    deserialized samples.
  * Config: `shuffle`, `prefetch`, `deserialize`.
* **Training script** chooses between `local` vs `streaming` via a single
  config value (`data_mode`).

---
## 2. Operational Tips

| Tip | Details |
|-----|---------|
| **IAM policy** | Attach a policy granting at least `s3:GetObject` on the dataset prefix, e.g.<br/>```json
{
  "Effect": "Allow",
  "Action": "s3:GetObject",
  "Resource": "arn:aws:s3:::quark-main-tokyo-bucket/datasets/*"
}
``` |
| **Local cache on fast SSD** | Set `CACHE_DIR=/mnt/nvme/quark_cache` or similar. Avoid NFS or network shares—latency defeats prefetch. |
| **Monitor request rate** | CloudWatch → *S3 | Requests | GetObject*. If TPS spikes (≫3 000/s) increase shard size (e.g. 64 MB → 128 MB) to reduce request count. |
| **Hugging Face Datasets** | Many tasks can skip our custom dataset: simply pass the S3 URI and `streaming=True`:<br/>```python
from datasets import load_dataset
train_ds = load_dataset('json', data_files='s3://quark-main-tokyo-bucket/datasets/train-*.json', streaming=True)
``` |

---
## 3. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_PROFILE` / `AWS_ACCESS_KEY_ID` | – | Credentials with `s3:GetObject`. |
| `CACHE_DIR` | `~/.cache/quark_stream` | Location of on-disk LRU cache. |
| `CACHE_GB` | `20` | Cache size limit in GB. |

---
## 4. Quick-start Example

```bash
pip install boto3 aiofiles aiohttp torch  # runtime deps
pip install moto[all]                     # optional, enables test_streaming.py
export AWS_PROFILE=quark
export CACHE_DIR=~/.cache/quark_stream
python - <<'PY'
from tools_utilities.scripts.s3_streaming_manager import StreamingManager
from brain.ml.dataset_shards.stream_dataset import StreamDataset
import json, torch

sm = StreamingManager(bucket='quark-main-tokyo-bucket')
train_ds = StreamDataset(sm, prefix='datasets/myset/train-', deserialize=json.loads)
loader = torch.utils.data.DataLoader(train_ds, batch_size=32)
print(next(iter(loader)))
PY
```

---
## 5. Troubleshooting

* **`AccessDenied`** → confirm IAM allows `s3:GetObject` on the prefix.
* **Slow throughput** → verify local SSD cache, increase `AWS_MAX_CONCURRENCY` (AWS CLI) or shard size.
* **`too many open files`** → raise ulimit or reduce `prefetch`.

---
© Quark Project – Streaming Data Infrastructure

## Recursive dataset discovery (2025-09 update)

You can now point Quark to *any* local dataset folder inside the repo and the training
launcher will automatically locate the matching shard prefix in S3.

Example natural-language trigger:

```
train quark with /Users/camdouglas/quark/data/datasets/alphagenome
```

Execution flow:

1. KnowledgeHub parses the command and forwards `dataset_local_path` to
   `ResourceManager.run_training_job()`.
2. `ResourceManager` converts the path into an S3-relative prefix
   (`data/datasets/alphagenome/`) and injects it as a `train_prefix` override.
3. `quark_cli.py` calls `dataset_discovery.discover(bucket, depth=3, root_prefix="data/datasets/alphagenome/")`
   which walks all sub-directories until it finds shard files (e.g. `train/`, `val/`).
4. CLI presents the discovered prefixes to the user for confirmation; training
   then streams shards directly from those prefixes.

CLI equivalent:

```
python tools_utilities/scripts/quark_cli.py "train quark" \
  --override bucket=quark-main-tokyo-bucket \
  --override train_prefix=data/datasets/alphagenome/ \
  --override data_mode=streaming
```

Notes:
• Discovery depth defaults to 3 to capture prefixes like `datasets/name/split-`.  
• Pass `--depth N` to `dataset_discovery.py` for manual inspection.

### Auto-shard discovery (2025-09 Batch B)

When you supply only a *directory* (not a split) Quark now auto-detects shard groups
based on simple heuristics (≥8 objects, ≥1 KB median size).  Example:

```
train quark with /Users/camdouglas/quark/data/datasets/my_big_corpus
```

The flow is identical to above except `dataset_discovery.discover_shard_groups()`
recursively scores prefixes and:

* If **one** shard group is found → it is auto-selected; no prompt.
* If **multiple** groups → CLI presents a numbered list and waits for selection.
* If **none** qualify → CLI falls back to depth-based listing.

CLI manual usage:

```
python tools_utilities/scripts/dataset_discovery.py \
  --bucket quark-main-tokyo-bucket --root data/datasets/my_big_corpus --auto-shards --human
```
