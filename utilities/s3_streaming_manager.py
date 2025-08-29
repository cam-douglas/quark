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
        
        self.logger.info("Scanning local files and S3 to determine files to upload...")
        for local_file_path in local_dir_path.rglob('*'):
            if local_file_path.is_file():
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
