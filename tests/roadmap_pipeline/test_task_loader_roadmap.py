import importlib
import shutil
from pathlib import Path


def test_sync_duplicate_skip(tmp_path):
    # copy tasks yaml dir to temp
    project_root = Path.cwd()
    task_dir = project_root / "state" / "tasks"
    tmp_task_dir = tmp_path / "tasks"
    shutil.copytree(task_dir, tmp_task_dir)

    # monkeypatch loader paths
    import state.quark_state_system.task_loader as tl

    tl._TASK_DIR = tmp_task_dir  # type: ignore
    tl._PRIORITY_FILES = {
        "high": tmp_task_dir / "tasks_high.yaml",
        "medium": tmp_task_dir / "tasks_medium.yaml",
        "low": tmp_task_dir / "tasks_low.yaml",
    }
    importlib.reload(tl)

    snapshot = {"Sample Planned Item": "planned"}
    tl.sync_with_roadmaps(snapshot)
    count_after_first = len(list(tl.get_tasks()))
    tl.sync_with_roadmaps(snapshot)  # run again – should not duplicate
    count_after_second = len(list(tl.get_tasks()))
    assert count_after_first == count_after_second
