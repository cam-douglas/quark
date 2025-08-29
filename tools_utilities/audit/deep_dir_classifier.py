#!/usr/bin/env python3
"""
Deep Directory Classifier

Walks the Quark repository recursively (excluding heavy dirs) and assigns
category + priority to *every* directory, not just first-level.  Uses rules:
    • any path under brain_architecture/ or brain_modules/ → category=brain, priority=2
    • under ml_architecture/, integration/, training/ → ml, priority=3
    • under quark_state_system/, tasks/ → state, priority=2
    • under management/ → governance, priority=2
    • testing/ and children → tests, priority=3
    • tools_utilities/, gym/, optimization/ → utilities, priority=4
    • docs/, documentation/ → documentation, priority=4
    • data_knowledge/ → data, priority=4
    • heavy dirs (datasets, models, external, mlruns, state_snapshots) → heavy, priority=5 (skipped)
    • logs/ → logs, priority=4
Outputs a CSV `audit_outputs/deep_dir_classification.csv`.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "audit_outputs" / "deep_dir_classification.csv"

HEAVY = {"datasets", "external", "models", "mlruns", "state_snapshots", "venv"}

RULES: Tuple[Tuple[str, str, int]] = (
    ("brain_architecture", "brain", 2),
    ("brain_modules", "brain", 2),
    ("ml_architecture", "ml", 3),
    ("integration", "ml", 3),
    ("training", "ml", 3),
    ("quark_state_system", "state", 2),
    ("tasks", "state", 2),
    ("management", "governance", 2),
    ("testing", "tests", 3),
    ("tools_utilities", "utilities", 4),
    ("gym", "utilities", 4),
    ("optimization", "utilities", 4),
    ("docs", "documentation", 4),
    ("documentation", "documentation", 4),
    ("data_knowledge", "data", 4),
    ("logs", "logs", 4),
)


def classify(path: Path) -> Tuple[str, int]:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    parts = rel.split("/")
    if parts[0] in HEAVY:
        return "heavy", 5
    for prefix, cat, pr in RULES:
        if rel.startswith(prefix):
            return cat, pr
    return "unknown", 4


def walk_and_classify():
    rows = []
    for d in PROJECT_ROOT.rglob("*"):
        if not d.is_dir():
            continue
        if d.name.startswith("."):
            continue
        if d.relative_to(PROJECT_ROOT).parts[0] in HEAVY:
            continue  # Skip heavy tree
        cat, pr = classify(d)
        rows.append((d.relative_to(PROJECT_ROOT).as_posix(), cat, pr))
    rows.sort()
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["directory", "category", "priority"])
        writer.writerows(rows)
    print(f"Deep directory classification written to {OUTPUT.relative_to(PROJECT_ROOT)} (rows={len(rows)})")


if __name__ == "__main__":
    walk_and_classify()
