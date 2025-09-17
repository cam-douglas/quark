import pathlib

BANNER_SNIPPET = "ALWAYS apply every rule across all Cursor rule"

RULE_DIRS = [
    pathlib.Path(__file__).resolve().parents[1] / ".cursor" / "rules",
    pathlib.Path(__file__).resolve().parents[1] / ".quark" / "rules",
]

def test_all_rule_files_have_banner():
    missing = []
    for rule_dir in RULE_DIRS:
        if not rule_dir.exists():
            continue
        for mdc in rule_dir.glob("*.mdc"):
            head = mdc.read_text(encoding="utf-8", errors="ignore")[:512]
            if BANNER_SNIPPET not in head:
                missing.append(str(mdc))
    assert not missing, f"Enforcement banner missing in: {', '.join(missing)}"
