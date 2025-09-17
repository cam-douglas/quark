"""Cursor rules loader utility.

Reads `.mdc` rule files from `.quark/rules/` (or `.cursor/rules/` fallback) and
returns their concatenated text.  Each file is sorted by filename so that
metadata `order:` keys (if added in future) or alphabetical order provides a
predictable deterministic ordering.

NOTE: The legacy monolithic `.cursorrules` remains the authoritative source
until migration completes; this loader is provided behind a feature-flag so it
can be integrated once validated.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List


def discover_rule_files(repo_root: str | Path) -> List[Path]:
    """Return all *.mdc files under .quark/rules or .cursor/rules."""
    root = Path(repo_root)
    quark_dir = root / ".quark" / "rules"
    cursor_dir = root / ".cursor" / "rules"

    if quark_dir.is_dir():
        search_root = quark_dir
    elif cursor_dir.is_dir():
        search_root = cursor_dir
    else:
        return []

    return sorted(search_root.glob("*.mdc"))


def load_concatenated_rules(repo_root: str | Path, include_metadata: bool = True) -> str:
    """Concatenate all rule files into a single string.

    Parameters
    ----------
    repo_root
        Project root containing the rules directories.
    include_metadata
        If *False*, YAML front-matter (lines between leading and trailing '---')
        is stripped from each file.
    """
    texts: List[str] = []
    for path in discover_rule_files(repo_root):
        content = path.read_text(encoding="utf-8")
        if not include_metadata:
            if content.startswith("---"):
                # Strip first YAML block
                _, _, remainder = content.partition("---\n")  # skip first '---\n'
                # partition again to remove closing '---'
                _, _, remainder = remainder.partition("---\n")
                content = remainder
        texts.append(content.strip())

    return "\n\n".join(texts)


def main() -> None:  # pragma: no cover – manual utility
    repo_root = Path(os.getcwd())
    print(load_concatenated_rules(repo_root, include_metadata=False))


if __name__ == "__main__":
    main()
