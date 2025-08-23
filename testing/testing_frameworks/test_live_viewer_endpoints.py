import importlib
import json
from types import ModuleType

import pytest

# Ensure Brian2 runs in no-GUI mode
import os
os.environ.setdefault("BRIAN2_DEVICE", "runtime")

MODULE_PATH = "brain_architecture.neural_core.live_viewer"

@pytest.fixture(scope="module")
def app_module() -> ModuleType:
    mod = importlib.import_module(MODULE_PATH)
    return mod

@pytest.fixture()
def client(app_module):
    app = app_module.app
    return app.test_client()

def test_healthz(client):
    rv = client.get("/healthz")
    assert rv.status_code == 200
    assert rv.data == b"ok"

def test_status_initial(client):
    rv = client.get("/status")
    assert rv.status_code == 200
    data = json.loads(rv.get_data(as_text=True))
    assert data["state"] in {"stopped", "running"}
    assert "mode" in data


def test_start_stop_cycle(client, app_module):
    # Start
    rv = client.get("/start")
    assert rv.status_code == 200
    # Ensure state flips to running
    rv2 = client.get("/status")
    data = json.loads(rv2.get_data(as_text=True))
    assert data["state"] == "running"
    # Stop
    rv3 = client.get("/stop")
    assert rv3.status_code == 200
    rv4 = client.get("/status")
    data2 = json.loads(rv4.get_data(as_text=True))
    assert data2["state"] == "stopped"
