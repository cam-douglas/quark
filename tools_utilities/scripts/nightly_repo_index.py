"""Nightly repository index generator.

This script is intended to be invoked by a cron job at 18:00 local time
(Sydney/Australia on the developer machine). It generates a full repository
index using the existing `generate_repo_index` module and stores timestamped
snapshots under `<repo_root>/repo_indexes/`.  A copy of the most recent index
is also written to `<repo_root>/repo_index.json` for convenience.

The script is self-contained: only Python standard-library imports are used.

Integration: Not simulator-integrated; repository tooling for indexing, validation, or CI.
Rationale: Executed by developers/CI to maintain repo health; not part of runtime simulator loop.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Import the helper without heavy dependencies
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools_utilities" / "scripts"))

from generate_repo_index import walk_repo  # type: ignore  # local import


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d")
    archive_dir = REPO_ROOT / "repo_indexes"
    archive_dir.mkdir(exist_ok=True)

    # Generate index
    index = walk_repo(REPO_ROOT)

    # Write timestamped snapshot
    snapshot_path = archive_dir / f"repo_index_{timestamp}.json"
    with snapshot_path.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    # Overwrite/update latest index at root
    latest_path = REPO_ROOT / "repo_index.json"
    with latest_path.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    print(f"Nightly repo index written: {snapshot_path.relative_to(REPO_ROOT)} (entries: {len(index)})")


if __name__ == "__main__":
    main()
