import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_research_plan_rule_present_and_nonempty():
    rule_path = PROJECT_ROOT / ".quark" / "rules" / "research-plan-implement-verify.mdc"
    assert rule_path.exists(), f"Missing rule file: {rule_path}"
    content = rule_path.read_text(encoding="utf-8").strip()
    assert content, "Rule file is empty"
    assert "Research ▸ Plan ▸ Implement ▸ Summarize ▸ Verify" in content


def test_cursor_and_quark_rules_directories_exist():
    quark_rules_dir = PROJECT_ROOT / ".quark" / "rules"
    cursor_rules_dir = PROJECT_ROOT / ".cursor" / "rules"
    assert quark_rules_dir.is_dir()
    assert cursor_rules_dir.is_dir()


def test_overlap_topics_detectable():
    # Detect common topics by keywords; not exhaustive but guards regressions
    cursor_rules = (PROJECT_ROOT / ".cursor" / "rules").glob("*.mdc")
    quark_rules = (PROJECT_ROOT / ".quark" / "rules").glob("*.mdc")

    cursor_text = "\n".join(p.read_text(encoding="utf-8") for p in cursor_rules)
    quark_text = "\n".join(p.read_text(encoding="utf-8") for p in quark_rules)

    # Key duplicate-topic hints
    duplicates = [
        ("Documentation-first", "README"),
        ("Placeholders strictly forbidden", "stubs"),
        ("ALWAYS execute ALL shell commands", "non-interactive flags"),
        ("coverage", "90%"),
    ]

    found = 0
    for a, b in duplicates:
        if a in cursor_text and a in quark_text:
            found += 1
        if b in cursor_text and b in quark_text:
            found += 1

    # At least one overlap should be detectable in this repo setup
    assert found >= 1


