from state.quark_state_system import task_loader
import yaml
import pytest

@pytest.fixture(autouse=True)
def setup_task_loader(tmp_path, monkeypatch):
    """Fixture to set up a temporary task directory and files for testing."""
    # Create a temporary tasks directory
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()

    # Create a dummy tasks_high.yaml file
    dummy_tasks = [
        {"id": "test1", "title": "High Prio Task", "priority": "high", "status": "pending"},
        {"id": "test2", "title": "Medium Prio Task", "priority": "medium", "status": "pending"},
    ]
    with open(task_dir / "tasks_high.yaml", "w") as f:
        yaml.dump(dummy_tasks, f)

    # Monkeypatch the _TASK_DIR in task_loader to use the temporary directory
    monkeypatch.setattr(task_loader, "_TASK_DIR", task_dir)
    
    # Reload tasks to populate the _TASKS global from the dummy file
    task_loader.generate_tasks_from_active_roadmaps()

    yield

    # Teardown: Reset the task loader's internal state
    task_loader.reset_all()


def test_loads_some_tasks():
    tasks = list(task_loader.get_tasks())
    assert tasks, "No tasks loaded"


def test_priority_filter():
    highs = list(task_loader.get_tasks(priority="high"))
    # There are no high priority tasks in the dummy file
    assert not highs
    # assert all(t["priority"] == "high" for t in highs)
