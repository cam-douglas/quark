"""CI test: ensure each ALWAYS/NEVER rule appears in exactly one .mdc file."""
from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

RULE_PATTERN = re.compile(r"^(ALWAYS|NEVER) .*?(\.|;)$", re.MULTILINE)


def _iter_rule_files(root: Path):
    rules_root = root / ".quark" / "rules"
    if not rules_root.is_dir():
        return []
    return rules_root.glob("*.mdc")


def test_unique_rules():
    repo_root = Path(__file__).resolve().parents[2]
    mapping: dict[str, list[str]] = defaultdict(list)

    for file_path in _iter_rule_files(repo_root):
        text = file_path.read_text(encoding="utf-8")
        # Remove YAML preamble if present
        if text.startswith("---"):
            _, _, rest = text.partition("---\n")
            _, _, text = rest.partition("---\n")
        for match in RULE_PATTERN.findall(text):
            rule_line = match.strip()
            mapping[rule_line].append(file_path.name)

    duplicates = {rule: files for rule, files in mapping.items() if len(files) > 1}
    assert not duplicates, (
        "Duplicate rules found in multiple .mdc files: "
        + ", ".join(f"'{r}' in {fs}" for r, fs in duplicates.items())
    )
