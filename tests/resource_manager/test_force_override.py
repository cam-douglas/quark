from pathlib import Path

from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager


def test_force_license_override(tmp_path):
    gpl_file = tmp_path / "gpl_mod.py"
    gpl_file.write_text("""# GNU GENERAL PUBLIC LICENSE\nprint('hi')\n""")

    rm = ResourceManager(auto_scan=False)
    rid = rm.register_resource(gpl_file, metadata={"force": True})
    meta = rm.registry[rid]
    assert meta["approved"] is True
    assert Path(meta["integrated_path"]).exists()
