#!/usr/bin/env python3
"""Synchronise MASTER_ROADMAP.md with the current *master_roadmap.md*.

This utility is invoked by the *pre-push* hook to ensure the canonical
roadmap snapshot (`MASTER_ROADMAP.md`) is always up-to-date on every push.
It performs three actions:

1. Backs up any existing *MASTER_ROADMAP.md* to a timestamped copy so history
   is never lost.
2. Copies the latest *management/rules/roadmap/master_roadmap.md* over
   *MASTER_ROADMAP.md*.
3. Emits a one-line confirmation for hook logs.

Exit codes
----------
0 – success
>0 – failure (missing source file, IO error, etc.)
"""
from __future__ import annotations

import datetime as _dt
import shutil as _shutil
import sys as _sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[2]
SRC = ROOT / "management" / "rules" / "roadmap" / "master_roadmap.md"
DST = ROOT / "management" / "rules" / "roadmap" / "MASTER_ROADMAP.md"


def _timestamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def main() -> None:  # pragma: no cover
    if not SRC.exists():
        print("[update_master_roadmap] ERROR: source master_roadmap.md missing", file=_sys.stderr)
        _sys.exit(1)

    # Determine newest mtime in roadmap directory
    roadmap_dir = SRC.parent
    newest = max(p.stat().st_mtime for p in roadmap_dir.rglob("*") if p.is_file())
    dst_mtime = DST.stat().st_mtime if DST.exists() else 0

    if newest <= dst_mtime:
        print("MASTER_ROADMAP.md already up-to-date.")
        return

    # Backup old snapshot if present then copy
    if DST.exists():
        backup = DST.with_suffix(f".backup_{_timestamp()}.md")
        DST.replace(backup)

    _shutil.copy2(SRC, DST)
    print("MASTER_ROADMAP.md updated (roadmap changes detected).")


if __name__ == "__main__":
    main()
