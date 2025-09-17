import os

from state.quark_state_system.quark_driver import QuarkDriver


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def test_rules_directories_exist():
    """Ensure both .quark/rules and .cursor/rules directories exist."""
    quark_rules_dir = os.path.join(PROJECT_ROOT, ".quark", "rules")
    cursor_rules_dir = os.path.join(PROJECT_ROOT, ".cursor", "rules")
    assert os.path.isdir(quark_rules_dir), f"Directory not found: {quark_rules_dir}"
    assert os.path.isdir(cursor_rules_dir), f"Directory not found: {cursor_rules_dir}"


def test_quark_driver_loads_rules():
    """QuarkDriver must load rules text at initialization."""
    driver = QuarkDriver(PROJECT_ROOT)
    assert hasattr(driver, "quark_rules_text"), "QuarkDriver missing `quark_rules_text` attribute."
    assert driver.quark_rules_text.strip(), "Loaded rules text is empty."
