from unittest import mock
import subprocess
import types

import boto3
import pytest
from moto import mock_aws

from tools_utilities.scripts.dataset_discovery import discover_shard_groups
from tools_utilities.scripts import quark_cli as qc


@pytest.fixture()
def s3_bucket():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = "quark-shard-auto-bucket"
        s3.create_bucket(Bucket=bucket)
        # make single shard group under root
        payload = b"x" * 1024  # 1 KB
        for i in range(10):
            s3.put_object(
                Bucket=bucket,
                Key=f"root/train/shard-{i:02d}.json",
                Body=payload,
            )
        yield bucket


def test_discover_single_group(s3_bucket):
    groups = discover_shard_groups(s3_bucket, "root", min_size_kb=0)
    assert list(groups) == ["root/train/"]


def test_cli_autoselect_single(monkeypatch, s3_bucket):
    """CLI should auto-select shard group when only one exists."""
    monkeypatch.setattr("builtins.input", lambda _: "0")  # default choice
    overrides = ["--override", f"bucket={s3_bucket}"]
    forwarded = []
    with mock.patch.object(subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as m:
        out = qc._ensure_dataset_overrides(forwarded + overrides)
    # Should include auto-selected train_prefix root/train/
    flat = " ".join(out)
    assert "train_prefix=root/train/" in flat or "train_prefix=root/train" in flat


def test_discover_multiple_groups(s3_bucket, monkeypatch):
    # add another group
    s3 = boto3.client("s3")
    payload = b"y" * 1024
    for i in range(10):
        s3.put_object(
            Bucket=s3_bucket,
            Key=f"root/val/shard-{i:02d}.json",
            Body=payload,
        )

    # Simulate user selecting second option (index 1)
    monkeypatch.setattr("builtins.input", lambda _: "1")
    forwarded = []  # let CLI discover shard groups
    overrides = ["--override", f"bucket={s3_bucket}"]
    with mock.patch.object(subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as m:
        out = qc._ensure_dataset_overrides(forwarded + overrides)
    flat = " ".join(out)
    assert "train_prefix=root/" in flat


def test_no_groups(s3_bucket):
    # Non-existent root returns empty dict
    groups = discover_shard_groups(s3_bucket, "does/not/exist", min_size_kb=0)
    assert groups == {}
