"""Generate a flat JSON index of the repository.

This script walks the repository tree (assumed to be the directory containing
this file's grand-parent) and produces a `repo_index.json` file at the project
root.  Each file (excluding ignored patterns) is represented by a single JSON
object with the following keys:

- path: POSIX-style relative path from repo root
- type: "file"
- size: File size in bytes (integer)
- line_count: Number of lines (integer) – null for binary or if unknown
- sha1: 40-character hex digest – null for large or binary files
- last_modified: ISO-8601 timestamp (UTC)
- lang: Simple language hint derived from file extension (no dot)

Excluded directories (e.g. `data/`, `.git/`, `venv/`, etc.) are represented by
an object with keys:
- path: directory path
- type: "dir"
- excluded: true

The script respects patterns in `.gitignore` and `.cursorignore` in addition to
a built-in set of exclusions.

Integration: Not simulator-integrated; repository tooling for indexing, validation, or CI.
Rationale: Executed by developers/CI to maintain repo health; not part of runtime simulator loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

# ---- Configuration ---------------------------------------------------------------------------

BUILTIN_EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "venv",
    "__pycache__",
    "quark.egg-info",
}

# Heavy directories that should be summarised but *not* descended into.
SCOPED_DIRS_EXCLUDED = {
    "data",
}

MAX_SHA1_BYTES = 5 * 1024 * 1024  # 5 MB – skip hashing larger files to save time

# ------------------------------------------------------------------------------------------------


def read_ignore_patterns(repo_root: Path) -> List[str]:
    """Collect ignore patterns from .gitignore and .cursorignore (if present)."""
    patterns: List[str] = []
    for fname in (".gitignore", ".cursorignore"):
        fpath = repo_root / fname
        if fpath.exists():
            with fpath.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    patterns.append(line)
    return patterns


def path_is_ignored(rel_path: str, ignore_regexps: List[re.Pattern[str]]) -> bool:
    """Return True if rel_path matches any ignore regexp."""
    return any(r.search(rel_path) for r in ignore_regexps)


def compile_ignore_regexps(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    """Convert simple glob-like patterns to regex for quick matching."""
    regexps: List[re.Pattern[str]] = []
    for pat in patterns:
        # Rough glob → regex conversion; not perfect but sufficient.
        pat_regex = re.escape(pat).replace(r"\*\*", ".*?").replace(r"\*", "[^/]*?")
        regexps.append(re.compile(pat_regex))
    return regexps


def guess_lang(path: Path) -> str | None:
    """Return a short language hint based on file extension (without dot)."""
    ext = path.suffix.lower().lstrip(".")
    return ext or None


def sha1_of_file(path: Path, max_bytes: int | None = None) -> str | None:
    try:
        if max_bytes is not None and path.stat().st_size > max_bytes:
            return None
        h = hashlib.sha1()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def is_binary(path: Path) -> bool:
    """Simple heuristic: look for NUL byte in first 1024 bytes."""
    try:
        with path.open("rb") as fh:
            sample = fh.read(1024)
        return b"\0" in sample
    except Exception:
        return True


def walk_repo(repo_root: Path) -> list[dict]:
    ignore_regexps = compile_ignore_regexps(read_ignore_patterns(repo_root))

    index: list[dict] = []

    def add_excluded_dir(rel_dir: str):
        index.append({
            "path": rel_dir + "/" if not rel_dir.endswith("/") else rel_dir,
            "type": "dir",
            "excluded": True,
        })

    for dirpath, dirnames, filenames in os.walk(repo_root, topdown=True):
        rel_dir = os.path.relpath(dirpath, repo_root)
        if rel_dir == ".":
            rel_dir = ""

        # Handle directory exclusions (top-down so we can modify dirnames in-place)
        dirnames_copy = dirnames[:]
        for d in dirnames_copy:
            if d in BUILTIN_EXCLUDE_DIRS or path_is_ignored(os.path.join(rel_dir, d), ignore_regexps):
                dirnames.remove(d)
                add_excluded_dir(os.path.join(rel_dir, d))
                continue
            if d in SCOPED_DIRS_EXCLUDED:
                dirnames.remove(d)
                add_excluded_dir(os.path.join(rel_dir, d))

        for fname in filenames:
            rel_path = os.path.join(rel_dir, fname) if rel_dir else fname
            if path_is_ignored(rel_path, ignore_regexps):
                continue
            full_path = repo_root / rel_path
            if not full_path.is_file():
                continue
            try:
                stat = full_path.stat()
            except (FileNotFoundError, OSError):
                continue
            file_entry = {
                "path": rel_path.replace(os.sep, "/"),
                "type": "file",
                "size": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "lang": guess_lang(full_path),
            }
            if not is_binary(full_path):
                try:
                    with full_path.open("rb") as fh:
                        line_count = sum(1 for _ in fh)
                    file_entry["line_count"] = line_count
                except Exception:
                    file_entry["line_count"] = None
            else:
                file_entry["line_count"] = None
            file_entry["sha1"] = sha1_of_file(full_path, MAX_SHA1_BYTES)
            index.append(file_entry)
    return index


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate repo_index.json for the Quark project.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent.parent.parent), help="Repo root directory")
    parser.add_argument("--output", type=str, default="repo_index.json", help="Output JSON file path (relative to root)")
    args = parser.parse_args(argv)

    repo_root = Path(args.root).resolve()
    if not repo_root.is_dir():
        sys.exit(f"Repo root '{repo_root}' is not a directory")

    index = walk_repo(repo_root)

    out_path = repo_root / args.output
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"Wrote {len(index):,} entries to {out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
