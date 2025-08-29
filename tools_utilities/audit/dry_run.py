#!/usr/bin/env python3
"""dry_run.py

Clone repo to /tmp, apply rewrite+move for a bucket, run tests.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_cmd(cmd: list[str], cwd: Path):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bucket", choices=["A", "B", "C"])
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="quark_reorg_") as tmp:
        tmp_path = Path(tmp)
        print(f"Cloning repo to {tmp_path} …")
        run_cmd(["git", "clone", "--depth", "1", str(PROJECT_ROOT), str(tmp_path)], cwd=PROJECT_ROOT)

        audit_dir = tmp_path / "tools_utilities" / "audit"
        csv = PROJECT_ROOT / f"audit_outputs/mapping_bucket_{args.bucket}.csv"

        # Install core requirements and minimal extra deps for test import resolution
        run_cmd(["python", "-m", "pip", "install", "-r", "requirements.txt"], cwd=tmp_path)
        run_cmd(["python", "-m", "pip", "install", "rope", "bowler", "requests", "python-dotenv"], cwd=tmp_path)

        # Rewrite imports
        run_cmd(["python", str(audit_dir / "rewrite_bucket.py"), str(csv), "--dry-run"], cwd=tmp_path)
        # Move files (still dry-run copy)
        run_cmd(["python", str(audit_dir / "move_bucket.py"), str(csv), "--dry-run"], cwd=tmp_path)

        # Run limited smoke tests (markers or subset)
        result = subprocess.run(["pytest", "-q", "-m", "basic or smoke or sanity"], cwd=tmp_path)
        print("pytest exit code:", result.returncode)

        if result.returncode != 0:
            print("❌ Tests failed in dry-run.")
        else:
            print("✅ Dry-run successful.")


if __name__ == "__main__":
    main()
