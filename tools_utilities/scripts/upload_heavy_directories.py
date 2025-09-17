

"""
Integration: Support utilities used by brain/state; indirectly integrated where imported.
Rationale: Shared helpers (performance, IO, streaming) used across runtime components.
"""
import argparse
from pathlib import Path

from tools_utilities.scripts.s3_streaming_manager import S3StreamingManager

def main():
    """
    Uploads heavy directories to S3, skipping existing files.
    """
    parser = argparse.ArgumentParser(description="Upload heavy directories to AWS S3.")
    parser.add_argument(
        "--dirs",
        nargs='+',
        required=True,
        help="List of directories to upload."
    )
    args = parser.parse_args()

    streaming_manager = S3StreamingManager()

    # The script is in utilities/, so root is one level up
    quark_root = Path(__file__).parent.parent

    directories_to_upload = args.dirs

    for directory_name in directories_to_upload:
        local_dir = quark_root / directory_name
        if local_dir.is_dir():
            s3_prefix = directory_name
            streaming_manager.upload_directory_to_s3(str(local_dir), s3_prefix)
        else:
            print(f"Directory '{directory_name}' not found in project root. Skipping.")

if __name__ == "__main__":
    main()
