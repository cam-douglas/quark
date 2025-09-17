from unittest import mock
from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager

def test_run_training_job_invokes_cli(monkeypatch):
    rm = ResourceManager()
    called = {}

    def fake_run(cmd, check=False):
        called['cmd'] = cmd
        class R: returncode = 0
        return R()

    with mock.patch('subprocess.run', fake_run):
        rc = rm.run_training_job('train', {'bucket': 'demo', 'train_prefix': 'datasets/'})
        assert rc == 0
        assert 'train quark' in " ".join(called['cmd'])
        assert '--override' in called['cmd']

    # fine-tune path
    with mock.patch('subprocess.run', fake_run):
        rc = rm.run_training_job('fine_tune', {})
        assert rc == 0
        assert 'fine-tune quark' in " ".join(called['cmd'])

def test_cli_alias_resolution(monkeypatch):
    from tools_utilities.scripts.quark_cli import resolve_bucket
    assert resolve_bucket('tokyo bucket') == 'quark-main-tokyo-bucket'
