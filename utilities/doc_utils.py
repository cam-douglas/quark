# utilities/doc_utils.py
"""Utility helpers for accessing documentation files.

Provides convenience constants and a loader so Quark can easily reference
any file inside the ``docs/`` hierarchy regardless of the caller's CWD.
"""
from pathlib import Path

# Resolve repository root (two levels up from this file: utilities/ -> repo)
_REPO_ROOT = Path(__file__).resolve().parents[1]

DOCS_ROOT: Path = _REPO_ROOT / "docs"
"""Path object pointing to the root ``docs/`` directory."""

INDEX_PATH: Path = DOCS_ROOT / "INDEX.md"
"""Convenience Path for the master documentation index file."""


def open_doc(relative_path: str) -> str:
    """Return the text contents of a documentation file.

    Parameters
    ----------
    relative_path
        Path relative to ``docs/`` (e.g. ``"overview/README.md"``).

    Returns
    -------
    str
        UTF-8 decoded file contents.
    """
    target = DOCS_ROOT / relative_path
    if not target.exists():
        raise FileNotFoundError(f"Documentation file not found: {target}")
    return target.read_text(encoding="utf-8")
