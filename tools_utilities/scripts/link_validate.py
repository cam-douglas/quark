#!/usr/bin/env python3
"""Link Validator – scans markdown files for broken internal links.

Usage:
    python tools_utilities/scripts/link_validate.py [path ...]
If no path is supplied, defaults to management/rules/roadmaps/ directory.

Integration: Not simulator-integrated; repository tooling for indexing, validation, or CI.
Rationale: Executed by developers/CI to maintain repo health; not part of runtime simulator loop.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import List, Tuple

link_pat = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def collect_md_files(paths: List[Path]) -> List[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files.extend(list(p.rglob("*.md")))
        elif p.suffix.lower() == ".md":
            files.append(p)
    return files


def validate_md(md_path: Path, repo_root: Path) -> List[Tuple[Path, str]]:
    broken: List[Tuple[Path, str]] = []
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    for match in link_pat.finditer(text):
        target = match.group(2)
        if target.startswith("http"):
            continue
        t_path = (md_path.parent / target).resolve()
        if not t_path.exists():
            broken.append((md_path.relative_to(repo_root), target))
    return broken


def main():
    repo_root = Path(__file__).resolve().parents[2]
    args = [Path(a) for a in sys.argv[1:]] or [repo_root / "management" / "rules" / "roadmaps"]
    md_files = collect_md_files(args)
    broken: List[Tuple[Path, str]] = []
    for md in md_files:
        broken.extend(validate_md(md, repo_root))

    if broken:
        print("Broken links detected:")
        for src, tgt in broken:
            print(f" - {src}: {tgt}")
        sys.exit(1)
    print("✅ All links valid.")


if __name__ == "__main__":
    main()
