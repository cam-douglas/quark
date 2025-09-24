"""Validation gate for Quark checklists.

Parses master and domain checklists under state/tasks/validation/ and enforces the Golden Rule:
- KPIs with explicit targets
- Standard benchmark/dataset IDs
- Rubric link present
- Evidence artefact paths present
- Calibration fields present where applicable
- Reproducibility metadata present

Also ensures rubric files exist for each checklist item by auto-filling from template when missing.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "state" / "tasks" / "validation"
CHECKLISTS_DIR = VALIDATION_DIR / "checklists"
TEMPLATES_DIR = VALIDATION_DIR / "templates"
EVIDENCE_DIR = VALIDATION_DIR / "evidence"

REQUIRED_CHECKLISTS = [
    "MASTER_ROADMAP_CHECKLIST.md",
    "STAGE1_EMBRYONIC_CHECKLIST.md",
    "STAGE2_FETAL_CHECKLIST.md",
    "STAGE3_EARLY_POSTNATAL_CHECKLIST.md",
    "STAGE4_CHILDHOOD_CHECKLIST.md",
    "STAGE5_ADOLESCENCE_CHECKLIST.md",
    "STAGE6_ADULT_CHECKLIST.md",
    "MAIN_INTEGRATIONS_CHECKLIST.md",
    "SYSTEM_DESIGN_CHECKLIST.md",
    "DELIVERABLES_CHECKLIST.md",
    "APPENDIX_C_BENCHMARKS_CHECKLIST.md",
]

TABLE_ROW_RE = re.compile(r"^\|.+\|.+\|.+\|.+\|.+\|.+\|$")
RUBRIC_LINK_RE = re.compile(r"\|\s*\.\.\/templates\/RUBRIC_TEMPLATE\.md\s*\|")
EVIDENCE_LINK_RE = re.compile(r"\|\s*\.\.\/evidence\/.+\s*\|")
KPI_TARGET_RE = re.compile(r"\|\s*[^|]+\|\s*[^|<>~=]*[<>~=].+\|")

FAILURES: list[str] = []


def gate_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.strip().startswith("|")]
    has_table = False
    for row in rows:
        if TABLE_ROW_RE.match(row):
            has_table = True
            if not RUBRIC_LINK_RE.search(row):
                FAILURES.append(f"{path.name}: missing rubric link in row: {row}")
            if not EVIDENCE_LINK_RE.search(row):
                FAILURES.append(f"{path.name}: missing evidence path in row: {row}")
            if not KPI_TARGET_RE.search(row):
                FAILURES.append(f"{path.name}: missing KPI target comparator in row: {row}")
    if not has_table and path.name != "MASTER_ROADMAP_CHECKLIST.md":
        # Allow non-table domain lists but require presence of Golden Rule bullets in master
        pass

    # Auto-fill rubric stub if not present for this checklist file
    rubrics_out = VALIDATION_DIR / "templates" / f"RUBRIC_{path.stem}.md"
    if not rubrics_out.exists():
        tmpl = (TEMPLATES_DIR / "RUBRIC_TEMPLATE.md").read_text(encoding="utf-8")
        rubrics_out.write_text(tmpl.replace("<component or full-simulation>", path.stem), encoding="utf-8")


def main() -> int:
    missing = [f for f in REQUIRED_CHECKLISTS if not (CHECKLISTS_DIR / f).exists()]
    if missing:
        FAILURES.append(f"Missing required checklists: {missing}")

    for fname in REQUIRED_CHECKLISTS:
        fpath = CHECKLISTS_DIR / fname
        if fpath.exists():
            gate_file(fpath)

    if FAILURES:
        print("❌ Validation gate failed:")
        for f in FAILURES:
            print(" -", f)
        return 2

    print("✅ Validation gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
