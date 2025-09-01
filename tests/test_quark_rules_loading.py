import os

from state.quark_state_system.quark_driver import QuarkDriver


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def test_rules_files_are_identical():
    """Ensure `.quarkrules` mirrors `.cursorrules`."""
    with open(os.path.join(PROJECT_ROOT, ".quarkrules"), "r", encoding="utf-8") as f:
        quark_rules = f.read()
    with open(os.path.join(PROJECT_ROOT, ".cursorrules"), "r", encoding="utf-8") as f:
        cursor_rules = f.read()
    assert quark_rules == cursor_rules, ".quarkrules and .cursorrules differ."


def test_quark_driver_loads_rules():
    """QuarkDriver must load rules text at initialization."""
    driver = QuarkDriver(PROJECT_ROOT)
    assert hasattr(driver, "quark_rules_text"), "QuarkDriver missing `quark_rules_text` attribute."
    assert driver.quark_rules_text.strip(), "Loaded rules text is empty."
