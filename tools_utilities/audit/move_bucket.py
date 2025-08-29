#!/usr/bin/env python3
"""move_bucket.py

Physically moves files/directories according to a bucket mapping CSV.
With --dry-run it copies into `.tmp_reorg/<bucket>/` and leaves originals.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TMP_ROOT = PROJECT_ROOT / ".tmp_reorg"


def load_mapping(csv_path: Path) -> List[tuple[str, str]]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["old_path"], row["new_path"]))
    return rows


def move(src: Path, dst: Path, dry_run: bool):
    if dry_run:
        # In dry-run mode we don’t materialise copies; we only log intention.
        print(f"    [dry-run] would copy {src} → {dst}")
        return

    # confirm mode – perform real move (via copy then delete)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    shutil.rmtree(src)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    rows = load_mapping(args.csv_path)
    if not args.confirm and not args.dry_run:
        print("⚠️  Use --dry-run to simulate or --confirm to move.")
        sys.exit(0)

    bucket_name = args.csv_path.stem.replace("mapping_", "")
    if args.dry_run:
        (TMP_ROOT / bucket_name).mkdir(parents=True, exist_ok=True)

    for old_rel, new_rel in rows:
        src = PROJECT_ROOT / old_rel
        dst = PROJECT_ROOT / new_rel
        if not src.exists():
            continue
        move(src, dst, dry_run=args.dry_run)

    print("Move pass finished.")


if __name__ == "__main__":
    main()
