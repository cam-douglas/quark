from brain.architecture.neural_core.cognitive_systems.callback_hub import hub
from brain.architecture.neural_core.cognitive_systems.plugins import ResourcePlugin
from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager


def test_listener_receives_event(tmp_path):
    events = []

    def _listener(ev, data):
        events.append((ev, data))

    hub.register(_listener)
    f = tmp_path / "file.txt"
    f.write_text("hello")
    rm = ResourceManager(auto_scan=False)
    rm.register_resource(f)
    assert any(ev == "resource_integrated" for ev, _ in events)


class InterceptPlugin(ResourcePlugin):
    def can_handle(self, meta):
        return meta["path"].endswith("special.txt")

    def integrate(self, meta):
        meta["integrated_path"] = "INTERCEPTED"
        meta["approved"] = True
        return True


def test_plugin_intercepts(tmp_path, monkeypatch):
    special = tmp_path / "special.txt"
    special.write_text("data")
    rm = ResourceManager(auto_scan=False)
    # inject intercept plugin
    rm.plugins.append(InterceptPlugin())
    rid = rm.register_resource(special)
    meta = rm.registry[rid]
    assert meta["integrated_path"] == "INTERCEPTED"
