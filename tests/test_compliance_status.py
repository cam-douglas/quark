import json
import os
from pathlib import Path

from tools_utilities.compliance_system.core_system import QuarkComplianceSystem


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_compliance_status_persistence(tmp_path):
    system = QuarkComplianceSystem(str(PROJECT_ROOT))

    # Ensure stop state first
    system.stop_system()
    status = system.get_system_status()
    assert status.get("running") is False

    # Start system and verify persisted running state
    system.start_system(background=True)
    status = system.get_system_status()
    assert status.get("running") is True

    # Status file contains pid and running flag
    status_file = PROJECT_ROOT / "logs" / "compliance_system_status.json"
    assert status_file.exists()
    with open(status_file, "r") as f:
        data = json.load(f)
    assert data.get("running") is True
    # pid may be None in some environments, but key should exist
    assert "pid" in data

    # Clean up
    system.stop_system()
    status = system.get_system_status()
    assert status.get("running") is False


