#!/usr/bin/env python3
"""
Quark Project AWS S3 Sync System
Syncs local files to AWS S3 bucket for backup and distribution
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aws_s3_sync.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class QuarkAWSS3Sync:
    def __init__(self, local_path: str = "files_for_aws3_upload", 
                 bucket: str = "quark-offload-us-west-2",
                 prefix: str = "quark-files/files_for_aws3_upload"):
        self.local_path = Path(local_path)
        self.bucket = bucket
        self.prefix = prefix
        self.s3_uri = f"s3://{bucket}/{prefix}"
        
        if not self.local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {self.local_path}")
    
    def run_sync(self, dry_run: bool = False, exclude_patterns: Optional[List[str]] = None) -> bool:
        """Run the S3 sync operation"""
        try:
            # Default exclude patterns
            if exclude_patterns is None:
                exclude_patterns = [
                    "*.DS_Store", "*.tmp", "*.log", "__pycache__/*",
                    "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db"
                ]
            
            # Build AWS CLI command
            cmd = [
                "aws", "s3", "sync",
                str(self.local_path),
                self.s3_uri,
                "--delete",  # Remove files in S3 that don't exist locally
                "--storage-class", "STANDARD",
                "--metadata-directive", "COPY"
            ]
            
            # Add exclude patterns
            for pattern in exclude_patterns:
                cmd.extend(["--exclude", pattern])
            
            # Add dry-run flag
            if dry_run:
                cmd.append("--dryrun")
            
            logger.info(f"Running S3 sync command: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("S3 sync completed successfully!")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"S3 sync failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during sync: {e}")
            return False
    
    def get_sync_status(self) -> tuple[Optional[Dict], Optional[Dict]]:
        """Check sync status by comparing local vs S3"""
        try:
            # Get local file count and size
            local_stats = self._get_local_stats()
            
            # Get S3 file count and size
            s3_stats = self._get_s3_stats()
            
            logger.info(f"Local files: {local_stats['count']} files, {local_stats['size']} bytes")
            logger.info(f"S3 files: {s3_stats['count']} files, {s3_stats['size']} bytes")
            
            return local_stats, s3_stats
            
        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return None, None
    
    def _get_local_stats(self) -> Dict:
        """Get local directory statistics"""
        count = 0
        total_size = 0
        
        for root, dirs, files in os.walk(self.local_path):
            count += len(files)
            for file in files:
                try:
                    file_path = Path(root) / file
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                except (OSError, PermissionError):
                    continue
        
        return {"count": count, "size": total_size}
    
    def _get_s3_stats(self) -> Dict:
        """Get S3 directory statistics"""
        try:
            cmd = [
                "aws", "s3", "ls", self.s3_uri, "--recursive", "--summarize"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output to get file count and size
            lines = result.stdout.strip().split('\n')
            count = 0
            total_size = 0
            
            for line in lines:
                if line.startswith('Total Objects:'):
                    count = int(line.split(':')[1].strip())
                elif line.startswith('Total Size:'):
                    size_str = line.split(':')[1].strip()
                    # Convert size string (e.g., "55.0 GiB") to bytes
                    if 'GiB' in size_str:
                        size_gb = float(size_str.replace(' GiB', ''))
                        total_size = int(size_gb * 1024**3)
                    elif 'MiB' in size_str:
                        size_mb = float(size_str.replace(' MiB', ''))
                        total_size = int(size_mb * 1024**2)
                    elif 'KiB' in size_str:
                        size_kb = float(size_str.replace(' KiB', ''))
                        total_size = int(size_kb * 1024)
                    else:
                        total_size = int(size_str)
            
            return {"count": count, "size": total_size}
            
        except Exception as e:
            logger.error(f"Error getting S3 stats: {e}")
            return {"count": 0, "size": 0}
    
    def list_s3_contents(self) -> List[str]:
        """List contents of the S3 directory"""
        try:
            cmd = ["aws", "s3", "ls", self.s3_uri, "--recursive"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.stdout.strip():
                return [line.strip() for line in result.stdout.strip().split('\n')]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error listing S3 contents: {e}")
            return []
    
    def estimate_upload_time(self) -> str:
        """Estimate upload time based on file size and typical upload speeds"""
        try:
            local_stats = self._get_local_stats()
            total_size_gb = local_stats['size'] / (1024**3)
            
            # Conservative estimates for upload speeds (varies by connection)
            speeds = {
                "Fast (100 Mbps)": 12.5,  # MB/s
                "Medium (50 Mbps)": 6.25,  # MB/s
                "Slow (25 Mbps)": 3.125    # MB/s
            }
            
            estimates = {}
            for speed_name, speed_mbps in speeds.items():
                time_seconds = (total_size_gb * 1024) / speed_mbps
                hours = int(time_seconds // 3600)
                minutes = int((time_seconds % 3600) // 60)
                estimates[speed_name] = f"{hours}h {minutes}m"
            
            return estimates
            
        except Exception as e:
            logger.error(f"Error estimating upload time: {e}")
            return {}

def main():
    parser = argparse.ArgumentParser(description="Quark Project AWS S3 Sync")
    parser.add_argument("--local-path", default="files_for_aws3_upload", 
                       help="Local path to sync (default: files_for_aws3_upload)")
    parser.add_argument("--bucket", default="quark-offload-us-west-2",
                       help="S3 bucket name (default: quark-offload-us-west-2)")
    parser.add_argument("--prefix", default="quark-files/files_for_aws3_upload",
                       help="S3 prefix/path (default: quark-files/files_for_aws3_upload)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be uploaded without actually uploading")
    parser.add_argument("--status", action="store_true",
                       help="Show sync status only")
    parser.add_argument("--list-s3", action="store_true",
                       help="List contents of S3 directory")
    parser.add_argument("--estimate-time", action="store_true",
                       help="Estimate upload time")
    
    args = parser.parse_args()
    
    try:
        # Initialize sync object
        sync = QuarkAWSS3Sync(
            local_path=args.local_path,
            bucket=args.bucket,
            prefix=args.prefix
        )
        
        if args.status:
            # Show status only
            local_stats, s3_stats = sync.get_sync_status()
            if local_stats and s3_stats:
                print(f"\n📊 Sync Status Summary:")
                print(f"Local: {local_stats['count']:,} files, {local_stats['size']:,} bytes")
                print(f"S3:    {s3_stats['count']:,} files, {s3_stats['size']:,} bytes")
                
                if local_stats['count'] > 0 and s3_stats['count'] > 0:
                    sync_percentage = (s3_stats['count'] / local_stats['count']) * 100
                    print(f"Sync:  {sync_percentage:.1f}% complete")
                    
        elif args.list_s3:
            # List S3 contents
            print(f"📁 S3 Contents for {sync.s3_uri}:")
            contents = sync.list_s3_contents()
            if contents:
                for item in contents:
                    print(f"  {item}")
            else:
                print("  No files found in S3 directory")
                
        elif args.estimate_time:
            # Estimate upload time
            print(f"⏱️  Upload Time Estimates for {args.local_path}:")
            estimates = sync.estimate_upload_time()
            for speed, time_estimate in estimates.items():
                print(f"  {speed}: {time_estimate}")
                
        else:
            # Run the sync
            print(f"🚀 Starting AWS S3 sync...")
            print(f"From: {args.local_path}")
            print(f"To:   {sync.s3_uri}")
            
            local_stats = sync._get_local_stats()
            print(f"Size:  {local_stats['size']:,} bytes ({local_stats['size']/(1024**3):.2f} GB)")
            
            if args.dry_run:
                print("🔍 DRY RUN MODE - No files will be uploaded")
            
            success = sync.run_sync(dry_run=args.dry_run)
            
            if success:
                print("✅ Sync completed successfully!")
                if not args.dry_run:
                    # Show final status
                    local_stats, s3_stats = sync.get_sync_status()
            else:
                print("❌ Sync failed!")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
