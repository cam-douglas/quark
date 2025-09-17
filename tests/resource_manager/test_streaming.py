"""Tests for StreamingManager + StreamDataset using moto mock S3."""

import json

import boto3
import pytest

import importlib
moto = importlib.import_module("moto")
if not hasattr(moto, "mock_s3"):
    pytest.skip("moto.mock_s3 unavailable in this moto version", allow_module_level=True)
from moto import mock_s3

from tools_utilities.scripts.s3_streaming_manager import StreamingManager
from brain.ml.dataset_shards.stream_dataset import StreamDataset

BUCKET = "test-stream-bucket"
PREFIX = "datasets/test/"

@pytest.fixture(scope="module")
def s3_setup():
    with mock_s3():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        # create 3 tiny shards
        for i in range(3):
            key = f"{PREFIX}shard-{i:02d}.jsonl"
            payload = "\n".join(json.dumps({"x": j}) for j in range(4)).encode()
            s3.put_object(Bucket=BUCKET, Key=key, Body=payload)
        yield s3


def test_streaming_manager_download(tmp_path, s3_setup):
    sm = StreamingManager(bucket=BUCKET, cache_dir=tmp_path, max_cache_gb=1)
    key = f"{PREFIX}shard-00.jsonl"
    with sm.open(key) as fh:
        data = fh.read()
    assert data.startswith(b"{\"x\": 0")
    # cached file exists
    assert any(tmp_path.rglob("*.jsonl"))


def test_stream_dataset_iteration(tmp_path, s3_setup):
    sm = StreamingManager(bucket=BUCKET, cache_dir=tmp_path, max_cache_gb=1)

    def _loads(buf):
        return [json.loads(line) for line in buf.splitlines()]

    ds = StreamDataset(sm, prefix=PREFIX, deserialize=_loads, shuffle=False, prefetch=0)
    samples = list(ds)
    # 3 shards * 4 lines each = 12 samples
    assert len(samples) == 12
    assert samples[0]["x"] == 0
