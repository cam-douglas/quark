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
    """Return all *.mdc files under known rules directories.

    Search order (first existing directories are scanned):
      1) $QUARK_RULES_DIR
      2) .quark/rules
      3) .cursor/rules
      4) management/rules
      5) management/rules/archive
      6) management/rules/roadmap

    The result is a deterministic, alphabetically sorted list without duplicates.
    """
    root = Path(repo_root)

    # Environment override to point at an external rules directory if desired
    env_dir = os.environ.get("QUARK_RULES_DIR")

    candidate_dirs: List[Path] = []
    if env_dir:
        candidate_dirs.append(Path(env_dir))

    candidate_dirs.extend(
        [
            root / ".quark" / "rules",
            root / ".cursor" / "rules",
            root / "management" / "rules",
            root / "management" / "rules" / "archive",
            root / "management" / "rules" / "roadmap",
        ]
    )

    seen: set[Path] = set()
    results: List[Path] = []
    for directory in candidate_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.mdc")):
            # De-duplicate while preserving order
            if path not in seen:
                seen.add(path)
                results.append(path)

    return results


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
