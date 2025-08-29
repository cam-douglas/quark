#!/usr/bin/env python3
"""rewrite_bucket.py

Batch-refactor imports for a single bucket.
This is **scaffold only** – no live changes until --confirm flag is passed.

Example:
    python rewrite_bucket.py audit_outputs/mapping_bucket_A.csv --dry-run
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- helpers -----------------------------------------------------------------

def load_mapping(csv_path: Path) -> List[tuple[str, str]]:
    mapping: List[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping.append((row["old_path"], row["new_path"]))
    return mapping


def rope_rename(old: str, new: str, dry_run: bool):
    """Call rope for package rename (simple import patterns)."""
    # Rope invocation disabled in dry-run. Log intention only.
    print(f"    (dry-run) would rename package {old} → {new}")


def main():
    parser = argparse.ArgumentParser(description="Rewrite imports for a mapping bucket")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="show changes only, no writes")
    parser.add_argument("--confirm", action="store_true", help="actually write changes (unsafe)")
    args = parser.parse_args()

    mapping = load_mapping(args.csv_path)
    print(f"Loaded {len(mapping)} mapping rows from {args.csv_path}")

    if not args.confirm and not args.dry_run:
        print("⚠️  No action taken. Use --dry-run or --confirm to proceed.")
        sys.exit(0)

    # Iterate over mapping; simple rope rename for top-level dirs/packages.
    for old, new in mapping:
        if old == new:
            continue
        print(f"[rope] {old}  ➜  {new}")
        rope_rename(old, new, dry_run=args.dry_run)

    print("Rewrite pass finished – review git diff or test results.")


if __name__ == "__main__":
    main()
