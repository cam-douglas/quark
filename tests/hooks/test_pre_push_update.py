import importlib
import pathlib
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.heavy

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "tools_utilities" / "scripts" / "pre_push_update.py"


def test_script_runs_without_error(tmp_path):
    # copy repo subset to tmp to avoid mutating real files
    tmp_repo = tmp_path / "repo"
    shutil.copytree(PROJECT_ROOT, tmp_repo, dirs_exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            str(tmp_repo / "tools_utilities" / "scripts" / "pre_push_update.py"),
        ]
    )
    assert result.returncode == 0


def test_link_validator_breaks_on_missing(monkeypatch):
    mod = importlib.import_module("tools_utilities.scripts.pre_push_update")
    # monkeypatch MASTER path to temp file with broken link
    tmp_md = pathlib.Path(mod.ROOT) / "tmp_master.md"
    tmp_md.write_text("[Bad](nonexistent.md)")
    monkeypatch.setattr(mod, "MASTER", tmp_md)
    with pytest.raises(SystemExit):
        mod.validate_links()
