#!/usr/bin/env python
"""dataset_discovery.py – List dataset prefixes and sizes in an S3 bucket.

Example
-------
$ python tools_utilities/scripts/dataset_discovery.py \
        --bucket quark-main-tokyo-bucket --depth 2 --human

Output
------
PREFIX                               SHARDS    SIZE (MB)
---------------------------------  --------  ----------
 datasets/alphagenome/train-              64      2 134
 datasets/alphagenome/val-                16        510
 datasets/myset/train-                    20        850
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Dict, Optional

import boto3
from humanize import naturalsize
from rich.console import Console
from rich.table import Table

console = Console()


def discover(bucket: str, depth: int, root_prefix: Optional[str] = None) -> Dict[str, dict]:
    """Return mapping prefix → {objects, bytes}.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    depth : int
        How many path components to keep when grouping objects.
    root_prefix : str | None
        If provided, only keys that start with this prefix are considered and the
        grouping depth is measured relative to bucket root, not the root_prefix.
    """
    if root_prefix and not root_prefix.endswith("/"):
        root_prefix += "/"

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    stats = defaultdict(lambda: {"objects": 0, "bytes": 0})

    import botocore
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=root_prefix or ""):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if root_prefix and not key.startswith(root_prefix):
                    continue
                # Build prefix at requested depth (relative to bucket root)
                prefix = "/".join(key.split("/")[:depth]) + "/"
                stats[prefix]["objects"] += 1
                stats[prefix]["bytes"] += obj["Size"]
    except botocore.exceptions.ClientError as e:
        if e.response["Error"].get("Code") == "AccessDenied":
            console.print(f"[red]Access denied to list objects in bucket {bucket}. Check IAM permissions.")
            return {}
        raise
    return stats

# ---------------------------------------------------------------------------
# Shard-group auto-discovery
# ---------------------------------------------------------------------------

def discover_shard_groups(
    bucket: str,
    root_prefix: str,
    *,
    min_objects: int = 8,
    min_size_kb: int = 64,
) -> Dict[str, dict]:
    """Return prefixes that *look* like shard groups under *root_prefix*.

    A prefix qualifies when it contains at least *min_objects* objects and the
    median object size is ≥ *min_size_kb* KB.  Directories like `.git/` or files
    beginning with a dot are ignored.
    """
    if not root_prefix.endswith("/"):
        root_prefix += "/"

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    groups: Dict[str, list[int]] = {}

    for page in paginator.paginate(Bucket=bucket, Prefix=root_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip dot-files and sentinel directories
            parts = key.split("/")
            if any(p.startswith(".") for p in parts):
                continue
            # Group by immediate parent directory of the file
            parent = "/".join(parts[:-1]) + "/"
            groups.setdefault(parent, []).append(obj["Size"])

    qualified: Dict[str, dict] = {}
    for prefix, sizes in groups.items():
        if len(sizes) < min_objects:
            continue
        sizes.sort()
        median_kb = sizes[len(sizes)//2] / 1024
        if median_kb < min_size_kb:
            continue
        qualified[prefix] = {"objects": len(sizes), "bytes": sum(sizes)}

    return qualified


def main():
    ap = argparse.ArgumentParser(description="Discover dataset prefixes in S3 bucket")
    ap.add_argument("--bucket", required=True, help="S3 bucket name")
    ap.add_argument("--depth", type=int, default=2, help="Prefix depth (default 2)")
    ap.add_argument("--root", help="Optional root prefix to restrict search")
    ap.add_argument("--auto-shards", action="store_true", help="Auto-detect shard groups under --root")
    ap.add_argument("--human", action="store_true", help="Human-readable sizes")
    args = ap.parse_args()

    if args.auto_shards and args.root:
        stats = discover_shard_groups(args.bucket, args.root)
    else:
        stats = discover(args.bucket, args.depth, root_prefix=args.root)

    if not stats:
        console.print(f"[red]No objects found in bucket {args.bucket}")
        sys.exit(1)

    tbl = Table(title=f"Datasets in {args.bucket}")
    tbl.add_column("PREFIX", justify="left")
    tbl.add_column("SHARDS", justify="right")
    tbl.add_column("SIZE", justify="right")

    for prefix, s in sorted(stats.items(), key=lambda x: x[0]):
        size = naturalsize(s["bytes"], binary=True) if args.human else str(s["bytes"])
        tbl.add_row(prefix, str(s["objects"]), size)

    console.print(tbl)


if __name__ == "__main__":
    main()
