"""aws_utils.py – Helper utilities for Quark’s SageMaker automation.

This tiny module aims to keep the training / fine-tuning scripts readable by
abstracting the boilerplate needed to spin up SageMaker jobs on the fly.
The functions here are *best-effort*: they will create missing resources but
will not modify existing ones aside from attaching the required policies.

Dependencies: boto3 and docker CLI (Docker daemon running).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Final

import boto3
from botocore.exceptions import ClientError

_ROLE_NAME: Final = "quark-sagemaker-role"
_REPO_NAME: Final = "quark-brain-train"
_POLICIES: Final = [
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
]


def ensure_iam_role() -> str:  # noqa: D401
    """Return an IAM role ARN suitable for SageMaker; create if absent."""
    iam = boto3.client("iam")
    try:
        role = iam.get_role(RoleName=_ROLE_NAME)["Role"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
        trust = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        role = iam.create_role(
            RoleName=_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Role auto-created by Quark for SageMaker jobs",
        )["Role"]
    # attach missing policies
    attached = {p["PolicyArn"] for p in iam.list_attached_role_policies(RoleName=_ROLE_NAME)["AttachedPolicies"]}
    for pol in _POLICIES:
        if pol not in attached:
            iam.attach_role_policy(RoleName=_ROLE_NAME, PolicyArn=pol)
    return role["Arn"]


def ensure_ecr_image(tag: str = "latest") -> str:  # noqa: D401
    """Build & push the local Dockerfile into ECR and return the image URI."""
    sts = boto3.client("sts")
    region = boto3.session.Session().region_name
    account = sts.get_caller_identity()["Account"]
    repo_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{_REPO_NAME}"
    ecr = boto3.client("ecr")
    try:
        ecr.describe_repositories(repositoryNames=[_REPO_NAME])
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "RepositoryNotFoundException":
            ecr.create_repository(repositoryName=_REPO_NAME)
        else:
            raise
    # Use AWS CLI to obtain password; safer & avoids CLI warnings.
    login_cmd = [
        "aws",
        "ecr",
        "get-login-password",
        "--region",
        region,
    ]
    pwd_proc = subprocess.run(login_cmd, capture_output=True, text=True, check=True)
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", repo_uri.split("/")[0]],
        input=pwd_proc.stdout,
        text=True,
        check=True,
    )
    image_uri = f"{repo_uri}:{tag}"
    # Build image only if tag missing in ECR; lightweight head request
    try:
        ecr.describe_images(repositoryName=_REPO_NAME, imageIds=[{"imageTag": tag}])
    except ClientError:
        dockerfile = Path(__file__).with_name("Dockerfile")
        if not dockerfile.exists():
            print("[aws_utils] No Dockerfile next to script; using SageMaker stock image.")
            return ""
        subprocess.run(["docker", "build", "-t", image_uri, str(dockerfile.parent)], check=True)
        subprocess.run(["docker", "push", image_uri], check=True)
    return image_uri


def default_image_uri() -> str:
    """Return SageMaker’s latest PyTorch GPU image for current region."""
    region = boto3.session.Session().region_name
    return (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04"
    )


# ---------------------------------------------------------------------------
# CLI entry-point (one-time setup)
# ---------------------------------------------------------------------------


def _cli():  # noqa: D401
    """Minimal CLI so users can run `python aws_utils.py --setup` once."""
    import argparse

    p = argparse.ArgumentParser(description="Quark AWS bootstrap helper")
    p.add_argument(
        "--setup",
        action="store_true",
        help="Create IAM role and ECR image if they do not yet exist.",
    )
    args = p.parse_args()

    if args.setup:
        role = ensure_iam_role()
        img = ensure_ecr_image()
        if not img:
            img = default_image_uri()
            print("No Dockerfile found; using stock SageMaker image:", img)
        print("Setup complete:\n IAM role:", role, "\n Image URI:", img)


if __name__ == "__main__":  # pragma: no cover
    _cli()
