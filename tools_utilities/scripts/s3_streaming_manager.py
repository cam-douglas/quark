

"""
Integration: Support utilities used by brain/state; indirectly integrated where imported.
Rationale: Shared helpers (performance, IO, streaming) used across runtime components.
"""
import boto3
import logging
from pathlib import Path
from botocore.exceptions import ClientError
from tqdm import tqdm
import threading

class S3StreamingManager:
    """Manages streaming access to models and datasets in S3"""

    def __init__(self, bucket_name: str = "quark-large-assets"):
        self.s3_client = boto3.client("s3")
        self.bucket_name = bucket_name
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def upload_directory_to_s3(self, local_directory: str, s3_prefix: str) -> bool:
        """Upload a directory to S3, skipping existing files and showing total progress."""
        local_dir_path = Path(local_directory)
        if not local_dir_path.is_dir():
            self.logger.error(f"❌ Local directory not found: {local_directory}")
            return False

        self.logger.info(f"Preparing to upload directory {local_directory} to s3://{self.bucket_name}/{s3_prefix}")

        files_to_upload = []
        total_upload_size = 0

        self.logger.info("Counting local files to scan...")
        total_files = sum(1 for f in local_dir_path.rglob('*') if f.is_file())
        self.logger.info(f"Found {total_files} total files to check against S3.")

        self.logger.info("Scanning local files and S3 to determine files to upload...")
        with tqdm(total=total_files, unit='file', desc="Scanning files") as pbar:
            for local_file_path in local_dir_path.rglob('*'):
                if local_file_path.is_file():
                    pbar.update(1)
                    s3_key = f"{s3_prefix}/{local_file_path.relative_to(local_dir_path)}"
                    try:
                        self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                    except ClientError as e:
                        if e.response['Error']['Code'] == '404':
                            file_size = local_file_path.stat().st_size
                            files_to_upload.append({'path': local_file_path, 'key': s3_key, 'size': file_size})
                            total_upload_size += file_size
                        else:
                            self.logger.error(f"Error checking file on S3 {s3_key}: {e}")
                            return False

        if not files_to_upload:
            self.logger.info("✅ All files are already synced to S3.")
            return True

        self.logger.info(f"Total upload size: {total_upload_size / (1024*1024):.2f} MB")

        try:
            with tqdm(total=total_upload_size, unit='B', unit_scale=True, desc="Total Upload Progress") as pbar:

                class ProgressCallback:
                    def __init__(self, pbar):
                        self._pbar = pbar
                        self._lock = threading.Lock()

                    def __call__(self, bytes_amount):
                        with self._lock:
                            self._pbar.update(bytes_amount)

                progress_callback = ProgressCallback(pbar)

                for file_info in files_to_upload:
                    pbar.set_description(f"Uploading {file_info['path'].name}")
                    try:
                        self.s3_client.upload_file(
                            str(file_info['path']),
                            self.bucket_name,
                            file_info['key'],
                            Callback=progress_callback
                        )
                    except FileNotFoundError:
                        self.logger.warning(f"⚠️ File not found during upload, skipping: {file_info['path']}")

            self.logger.info(f"🎉 Successfully uploaded directory {local_directory}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to upload directory {local_directory}: {e}")
            return False

# -----------------------------------------------------------------------------
# Lightweight Streaming Utilities (added 2025-09-01)
# -----------------------------------------------------------------------------
from collections import OrderedDict
import asyncio
import time
import botocore

class _LRUDiskCache:
    """Least-recently-used on-disk cache capped by `max_bytes`."""

    def __init__(self, cache_dir: str | Path, max_bytes: int = 20 * 1024**3):
        self.root = Path(cache_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._index: OrderedDict[str, int] = OrderedDict()  # key -> size bytes
        self._refresh_index()

    def _refresh_index(self):
        self._index.clear()
        total = 0
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                size = p.stat().st_size
                self._index[str(p)] = size
                total += size
        self._trim(total)

    def _trim(self, current: int):
        while current > self.max_bytes and self._index:
            path_str, size = self._index.popitem(last=False)
            try:
                Path(path_str).unlink(missing_ok=True)
                current -= size
            except Exception:
                break

    def get_path(self, key: str) -> Path:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # move to end (most recently used)
            size = path.stat().st_size
            self._index.pop(str(path), None)
            self._index[str(path)] = size
        return path


class StreamingManager:
    """On-demand downloader with async prefetch + disk-based LRU cache.

    Example
    -------
    >>> sm = StreamingManager(bucket="quark-main-tokyo-bucket")
    >>> with sm.open("datasets/myset/train-00001.parquet") as fh:
    ...     data = fh.read()
    """

    def __init__(self, bucket: str, cache_dir: str = "~/.cache/quark_stream", max_cache_gb: int = 20):
        self.bucket = bucket
        self.s3 = boto3.client("s3")
        self.cache = _LRUDiskCache(cache_dir, max_cache_gb * 1024**3)
        self.logger = logging.getLogger(__name__ + ".stream")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def open(self, key: str, binary: bool = True):
        """Return a readable file handle. Downloads to cache if missing."""
        path = self.cache.get_path(key)
        if not path.exists():
            self._download(path, key)
        mode = "rb" if binary else "r"
        return path.open(mode)

    async def prefetch(self, keys: list[str]):
        """Asynchronously download a list of keys into the cache."""
        loop = asyncio.get_event_loop()
        await asyncio.gather(*(loop.run_in_executor(None, self._download, self.cache.get_path(k), k) for k in keys))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    _MAX_RETRIES = 5

    def _download(self, path: Path, key: str):
        if path.exists():
            return  # already cached

        tmp_path = path.with_suffix(".part")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                self.s3.download_file(self.bucket, key, str(tmp_path))
                tmp_path.rename(path)
                self.logger.debug("Downloaded %s to %s", key, path)
                return
            except botocore.exceptions.BotoCoreError as e:
                self.logger.warning("Download failed (%d/%d) for %s: %s", attempt, self._MAX_RETRIES, key, e)
                if attempt == self._MAX_RETRIES:
                    self.logger.error("Giving up on %s", key)
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                    raise
                time.sleep(2 ** attempt)  # exponential back-off
