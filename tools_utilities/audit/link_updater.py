#!/usr/bin/env python3
"""link_updater.py

Scan *.md and *.rst files, detect relative links pointing to pre-reorg paths,
and rewrite them to the new locations using the combined mapping CSVs.

Usage:
    python link_updater.py --map audit_outputs/path_mapping_draft.csv --dry-run

With --dry-run the script prints diffs only; without it, files are updated
in-place.  A backup copy (<file>.bak) is written before modification.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MD_LINK_RE = re.compile(r"\]\(([^)]+)\)")


def load_mapping(csv_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["old_path"].rstrip("/")] = row["new_path"].rstrip("/")
    return mapping


def rewrite_links(text: str, mapping: Dict[str, str]) -> str:
    def _replace(match):
        url = match.group(1)
        for old, new in mapping.items():
            if url.startswith(old):
                return f"]({url.replace(old, new, 1)})"
        return match.group(0)

    return MD_LINK_RE.sub(_replace, text)


def process_file(path: Path, mapping: Dict[str, str], dry_run: bool):
    original = path.read_text(encoding="utf-8")
    updated = rewrite_links(original, mapping)
    if original == updated:
        return
    if dry_run:
        print(f"Would update links in {path}")
    else:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(updated, encoding="utf-8")
        print(f"Updated links in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, type=Path, help="CSV with old_path,new_path columns")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mapping = load_mapping(args.map)
    doc_files = list(PROJECT_ROOT.rglob("*.md")) + list(PROJECT_ROOT.rglob("*.rst"))
    for f in doc_files:
        # skip mapping or audit outputs themselves
        if "audit_outputs" in f.parts:
            continue
        process_file(f, mapping, args.dry_run)

    print("Link update pass complete.")


if __name__ == "__main__":
    main()
