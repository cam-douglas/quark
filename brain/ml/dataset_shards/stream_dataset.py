from __future__ import annotations

"""StreamDataset – stream shards from S3 using StreamingManager.
Only lightweight helper; heavyweight objects live elsewhere.
"""

import logging
import random
from typing import Callable, Iterable, List, Optional

import boto3

try:
    import torch
    from torch.utils.data import IterableDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("StreamDataset requires PyTorch. Install with `pip install torch`. ") from exc

from tools_utilities.scripts.s3_streaming_manager import StreamingManager

log = logging.getLogger(__name__)


def _identity(buf: bytes):
    return [buf]


class StreamDataset(IterableDataset):
    """IterableDataset that streams dataset shards directly from S3."""

    def __init__(
        self,
        sm: StreamingManager,
        *,
        prefix: str,
        deserialize: Callable[[bytes], List] = _identity,
        shuffle: bool = True,
        seed: Optional[int] = None,
        prefetch: int = 1,
    ) -> None:
        super().__init__()
        self.sm = sm
        self.prefix = prefix.rstrip("/")
        self.deserialize = deserialize
        self.shuffle = shuffle
        self.seed = seed
        self.prefetch = max(prefetch, 0)
        self._keys: List[str] | None = None

    def __iter__(self) -> Iterable:
        if self._keys is None:
            self._keys = self._list_keys()
            log.info("StreamDataset discovered %d shards under %s", len(self._keys), self.prefix)

        keys = list(self._keys)
        if self.shuffle:
            rng = random.Random(self.seed if self.seed is not None else torch.initial_seed())
            rng.shuffle(keys)

        loop = None
        if self.prefetch > 0:
            import asyncio
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)

        for idx, key in enumerate(keys):
            if self.prefetch > 0 and loop is not None:
                upcoming = keys[idx + 1 : idx + 1 + self.prefetch]
                if upcoming:
                    loop.run_until_complete(self.sm.prefetch(upcoming))
            with self.sm.open(key, binary=True) as fh:
                payload = fh.read()
            shard_bytes = len(payload)
            first = True
            for sample in self.deserialize(payload):
                if first:
                    yield sample, shard_bytes
                    first = False
                else:
                    yield sample, 0
        if loop is not None:
            loop.close()

    # --------------------------------------------------------------
    def _list_keys(self) -> List[str]:
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=self.sm.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        if not keys:
            raise RuntimeError(f"No objects found with prefix '{self.prefix}' in bucket '{self.sm.bucket}'.")
        return keys
