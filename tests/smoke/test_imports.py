import importlib
import pytest

@pytest.mark.parametrize("pkg", [
    "brain",
    "ml",
    "state",
    "utilities",
])
def test_import(pkg):
    importlib.import_module(pkg)
