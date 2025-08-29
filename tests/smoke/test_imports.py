import importlib
import pytest

@pytest.mark.parametrize(
    "pkg",
    [
        "brain.architecture",
        "ml.architecture",
        "state.quark_state_system",
        "utilities.tools_utilities",
    ],
)
def test_import(pkg):
    importlib.import_module(pkg)
