#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_JSON = ROOT / "repo_indexes" / "RULES_INDEX.json"
CURSOR_DIR = ROOT / ".cursor" / "rules"
QUARK_DIR = ROOT / ".quark" / "rules"
VALIDATION_ANCHOR = ROOT / "state" / "tasks" / "validation"

FAILURES: list[str] = []

def fail(msg: str) -> None:
    FAILURES.append(msg)


def load_index() -> dict:
    if not INDEX_JSON.exists():
        fail(f"Missing index: {INDEX_JSON}")
        return {}
    try:
        return json.loads(INDEX_JSON.read_text())
    except Exception as e:
        fail(f"Invalid JSON in {INDEX_JSON}: {e}")
        return {}


def check_paths(entries: list[dict]) -> None:
    for r in entries:
        p = Path(r.get("path", ""))
        if not p.exists():
            fail(f"Indexed path missing: {p}")
        if r.get("type") == "cursor" and CURSOR_DIR not in p.parents:
            fail(f"Cursor rule not under .cursor/rules: {p}")
        if r.get("type") == "quark" and QUARK_DIR not in p.parents:
            fail(f"Quark rule not under .quark/rules: {p}")


def check_metadata(entries: list[dict]) -> None:
    for r in entries:
        if r.get("description") is None:
            # allow empty string but not None unless file lacks header
            pass
        if r.get("alwaysApply") is None and r.get("type") == "cursor":
            # Some rules may rely on applyIntelligently; not hard fail
            pass
        # priority can be null; only validate type
        if r.get("type") not in {"cursor", "quark"}:
            fail(f"Unknown rule type: {r}")


def check_validation_anchoring(entries: list[dict]) -> None:
    # Any rule with 'VALIDATION' or 'GOLDEN RULE' in title must reference anchor
    keywords = ("VALIDATION", "GOLDEN RULE", "GLOBAL PRIORITY")
    for r in entries:
        title = (r.get("title") or "").upper()
        if any(k in title for k in keywords):
            p = Path(r["path"])  # already validated exists
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                fail(f"Cannot read rule {p}: {e}")
                continue
            anchor_str = str(VALIDATION_ANCHOR)
            if anchor_str not in text:
                fail(f"Validation rule missing anchor reference: {p} (expected path: {anchor_str})")


def main() -> int:
    data = load_index()
    rules = data.get("rules", []) if isinstance(data, dict) else []
    if not rules:
        fail("No rules found in index")
    else:
        check_paths(rules)
        check_metadata(rules)
        check_validation_anchoring(rules)
    if FAILURES:
        print("❌ RULES VALIDATION FAILED:")
        for f in FAILURES:
            print(" - ", f)
        return 2
    print("✅ Rules index validation passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
