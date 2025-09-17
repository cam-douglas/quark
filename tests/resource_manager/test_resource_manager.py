from pathlib import Path

from brain.architecture.neural_core.cognitive_systems.resource_management import (
    ResourceManager,
    compute_sha256,
)


def test_register_and_approve(tmp_path):
    # Create a dummy small file
    f = tmp_path / "dummy.txt"
    data = b"hello world"
    f.write_bytes(data)

    rm = ResourceManager(auto_scan=False)
    rid = rm.register_resource(f)

    # Resource is auto-approved and integrated for safe files
    meta = rm.registry[rid]
    assert meta["approved"] is True
    assert Path(meta["integrated_path"]).exists()


def test_hash_util(tmp_path):
    f = tmp_path / "data.bin"
    content = b"abc123"
    f.write_bytes(content)
    assert compute_sha256(f) == compute_sha256(f)  # deterministic


def test_sandbox_compile_failure(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def oops(:\n    pass")  # syntax error
    rm = ResourceManager(auto_scan=False)
    rid = rm.register_resource(bad)
    # Should not be integrated due to sandbox failure
    meta = rm.registry.get(rid, {})
    assert "integrated_path" not in meta


def test_gpl_block(tmp_path):
    gpl = tmp_path / "lib.py"
    gpl.write_text("""# GNU GENERAL PUBLIC LICENSE\nprint('hi')\n""")
    rm = ResourceManager(auto_scan=False)
    rid = rm.register_resource(gpl)
    meta = rm.registry.get(rid, {})
    # GPL should be blocked unless force flag set
    assert meta.get("license") == "GPL-3.0"
    assert "integrated_path" not in meta
