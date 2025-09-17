"""Fail CI if new modules lack a README/DESIGN with required headers."""
from __future__ import annotations

import re
from pathlib import Path

REQUIRED_HEADERS = {
    "Overview",
    "Requirements",
    "Design",
    "Technology Capsules",
    "Phase Plan",
    "Deliverables",
    "Success-Criteria Sprint",
}
HEADER_RE = re.compile(r"^(#+) (.+)$", re.MULTILINE)

# Directories to exclude from this check
EXCLUDE_DIRS = {
    "node_modules",
    "typings",
    ".venv",
    ".git",
    "__pycache__",
    "build",
    "dist",
    "docs/historical",
}


def _iter_design_files(root: Path):
    for md in root.rglob("README.md"):
        if any(d in md.parts for d in EXCLUDE_DIRS):
            continue
        yield md
    for md in root.rglob("DESIGN.md"):
        if any(d in md.parts for d in EXCLUDE_DIRS):
            continue
        yield md


def test_design_template_headers():
    repo_root = Path(__file__).resolve().parents[2]
    missing: dict[str, set[str]] = {}

    for file_path in _iter_design_files(repo_root):
        text = file_path.read_text(encoding="utf-8")
        headers = {m.group(2).strip() for m in HEADER_RE.finditer(text)}
        missing_headers = REQUIRED_HEADERS - headers
        if missing_headers:
            missing[str(file_path)] = missing_headers

    assert not missing, "Missing template headers: " + "; ".join(
        f"{p} → {', '.join(sorted(hdrs))}" for p, hdrs in missing.items()
    )
