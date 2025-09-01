from management.rules.roadmaps import roadmap_controller as rc


def test_index_parsing():
    rows = rc.get_index()
    # Ensure index contains at least the master roadmap
    titles = [r["title"] for r in rows]
    assert any("Master Roadmap" in t for t in titles)


def test_status_snapshot_mapping():
    snap = rc.status_snapshot()
    assert isinstance(snap, dict)
    # snapshot keys should be non-empty titles
    assert snap, "Snapshot should not be empty"
    for k, v in snap.items():
        assert k, "Title missing"
        assert v in {"done", "progress", "planned", "—", "unknown"}
