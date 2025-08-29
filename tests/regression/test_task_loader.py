from state.quark_state_system import task_loader

def test_loads_some_tasks():
    tasks = list(task_loader.get_tasks())
    assert tasks, "No tasks loaded"

def test_priority_filter():
    highs = list(task_loader.get_tasks(priority="high"))
    assert all(t["priority"] == "high" for t in highs)
