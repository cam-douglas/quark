import importlib

import pytest


@pytest.mark.parametrize(
    "pkg",
    [
        "brain.architecture",
        "state.quark_state_system",
        "tools_utilities",
    ],
)
def test_import(pkg):
    importlib.import_module(pkg)
