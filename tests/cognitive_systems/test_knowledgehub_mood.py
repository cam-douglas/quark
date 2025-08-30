import types

from brain.architecture.neural_core.cognitive_systems.knowledge_hub import KnowledgeHub


def test_mood_drive_forwarding(monkeypatch):
    calls = {"mood": None, "drives": None}

    class _MockAdapter:
        def set_mood(self, mood):
            calls["mood"] = mood

        def set_drives(self, drives):
            calls["drives"] = drives

    # Patch EpisodicMemory stub with adapter attribute
    mem_stub = types.SimpleNamespace(_e8_adapter=_MockAdapter())

    kh = KnowledgeHub(mem_stub)

    mood = {"entropy": 0.2}
    drives = {"curiosity": 0.8}
    kh.set_mood(mood)
    kh.set_drives(drives)

    assert calls["mood"] == mood
    assert calls["drives"] == drives
