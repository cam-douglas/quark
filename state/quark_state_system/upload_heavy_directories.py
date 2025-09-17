


"""
Integration: Indirect integration via QuarkDriver and AutonomousAgent; orchestrates simulator runs.
Rationale: State system validates, plans, and triggers actions that the simulator executes.
"""
import argparse
from pathlib import Path
import sys

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from s3_streaming_manager import S3StreamingManager

def main():
    """
    Uploads heavy directories ('models' and 'datasets') to S3, skipping existing files.
    """
    parser = argparse.ArgumentParser(description="Upload heavy directories to AWS S3.")
    parser.add_argument(
        "--dirs",
        nargs='+',
        default=['models', 'datasets'],
        help="List of directories to upload. Defaults to 'models' and 'datasets'."
    )
    args = parser.parse_args()

    streaming_manager = S3StreamingManager()

    quark_root = Path(__file__).parent.parent.parent

    directories_to_upload = args.dirs

    for directory_name in directories_to_upload:
        local_dir = quark_root / directory_name
        if local_dir.is_dir():
            s3_prefix = directory_name  # e.g., 'models' or 'datasets'
            streaming_manager.upload_directory_to_s3(str(local_dir), s3_prefix)
        else:
            print(f"Directory '{local_dir}' not found, skipping.")

if __name__ == "__main__":
    main()
