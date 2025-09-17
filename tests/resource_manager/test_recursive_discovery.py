import json
import subprocess
import types
from pathlib import Path
from unittest import mock

import boto3
import pytest
# moto >=5: use mock_aws with service filter
from moto import mock_aws

# Local imports
from tools_utilities.scripts import dataset_discovery as dd
from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
from brain.architecture.neural_core.cognitive_systems.knowledge_hub import KnowledgeHub


@pytest.fixture()
def s3_bucket(tmp_path):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = "quark-unit-test-bucket"
        s3.create_bucket(Bucket=bucket)
        # create fake shard keys under nested prefix
        for i in range(3):
            key = f"data/datasets/alphagenome/train/shard-{i:02d}.json"
            s3.put_object(Bucket=bucket, Key=key, Body=json.dumps([{"x": i}]).encode())
        yield bucket


def test_discover_root_prefix(s3_bucket):
    stats = dd.discover(s3_bucket, depth=4, root_prefix="data/datasets/alphagenome")
    assert any(k.startswith("data/datasets/alphagenome/") for k in stats)


def test_resource_manager_prefix_conversion(tmp_path, s3_bucket):
    rm = ResourceManager(auto_scan=False)
    local_path = Path("/Users/test/quark/data/datasets/alphagenome")
    with mock.patch.object(subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as m:
        rc = rm.run_training_job("train", overrides={"bucket": s3_bucket}, dataset_local_path=str(local_path))
    assert rc == 0
    # Ensure train_prefix injected in command args
    cmd = " ".join(m.call_args[0][0])
    assert "train_prefix=alphagenome/" in cmd


def test_knowledgehub_command_parsing(s3_bucket, tmp_path):
    hub = KnowledgeHub()
    with mock.patch.object(ResourceManager, "run_training_job", return_value=0) as m:
        msg = hub.handle_command("train quark with /Users/test/quark/data/datasets/alphagenome")
    assert "Training launched" in msg
    assert m.called
