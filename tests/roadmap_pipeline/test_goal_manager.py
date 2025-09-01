from state.quark_state_system import goal_manager, task_loader


def test_next_goal_priority():
    # ensure at least one pending high priority task exists
    task_loader._TASKS.append({"id": "test1", "title": "High Prio", "priority": "high", "status": "pending"})  # type: ignore
    task_loader._TASKS.append({"id": "test2", "title": "Low Prio", "priority": "low", "status": "pending"})  # type: ignore
    nxt = goal_manager.next_goal()
    assert nxt["title"] == "High Prio"
