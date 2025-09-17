#!/usr/bin/env python
"""s3_sync.py – Sync local /data directory with S3 bucket.

• Upload new/updated files
• Remove objects from bucket that no longer exist locally
• Scheduled via GitHub Actions cron at 08:00 UTC (≈19:00 Sydney)

Safety
------
Dry-run by default. Use `--yes` to perform changes.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA = ROOT / "data"
BUCKET = "quark-main-tokyo-bucket"

s3 = boto3.client("s3")

def sha256(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def list_local_files() -> Dict[str, str]:
    mapping = {}
    for fp in LOCAL_DATA.rglob("*"):
        if fp.is_file():
            key = f"data/{fp.relative_to(LOCAL_DATA).as_posix()}"
            mapping[key] = sha256(fp)
    return mapping

def list_s3_objects() -> Dict[str, str]:
    mapping = {}
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=BUCKET, Prefix="data/"):
            for obj in page.get("Contents", []):
                mapping[obj["Key"]] = obj.get("ETag", "").strip('"')
    except ClientError as e:
        print("S3 listing failed:", e)
        sys.exit(1)
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="Apply changes (default dry-run)")
    args = ap.parse_args()

    local = list_local_files()
    remote = list_s3_objects()

    uploads = [k for k, h in local.items() if remote.get(k) != h]
    deletions = [k for k in remote if k not in local]

    print(f"⬆️  {len(uploads)} objects to upload/update")
    print(f"🗑️  {len(deletions)} objects to delete from bucket")

    if not args.yes:
        print("Dry-run mode. Re-run with --yes to apply.")
        return

    for key in tqdm(uploads, desc="Uploading", unit="file"):
        src = LOCAL_DATA / Path(key).relative_to("data")
        s3.upload_file(str(src), BUCKET, key)

    for key in tqdm(deletions, desc="Deleting", unit="file"):
        s3.delete_object(Bucket=BUCKET, Key=key)

    print("Sync complete.")

if __name__ == "__main__":
    main()
