#!/usr/bin/env python3
"""
GCS Sync to Google Cloud Service - Intelligent bucket organization and synchronization.

ACTIVATION WORDS: sync to gcs, sync to google cloud, sync to google cloud service,
                  download from gcs, download from google cloud, download from google cloud service

This script maintains your local single data/ directory while automatically
organizing uploads into the appropriate GCS buckets based on content type.
Includes automatic cleanup of heavy directories after successful upload.
"""

import os
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuarkDataSync:
    """Intelligent GCS sync with automatic bucket organization."""
    
    def __init__(self, project_root: str = "/Users/camdouglas/quark"):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        
        # Heavy directories to clean up after successful sync (size threshold in MB)
        self.heavy_directories = {
            "external/": 100,  # Clean if > 100MB
            "archive/": 50,    # Clean if > 50MB
            "models/": 200,    # Clean if > 200MB
            "datasets/": 500,  # Clean if > 500MB
        }
        
        # Bucket mapping configuration
        self.bucket_mapping = {
            "raw": {
                "bucket": "gs://quark-data-raw",
                "patterns": [
                    "experimental_papers/",
                    "external/",
                    "web_archives/",
                    "archive/",
                    "*.pdf",
                    "*.xml",
                    "*.html",
                    "*_raw.*",
                    "*_original.*"
                ],
                "description": "Raw, unprocessed data from external sources"
            },
            "processed": {
                "bucket": "gs://quark-data-processed", 
                "patterns": [
                    "datasets/",
                    "knowledge/",
                    "structures/",
                    "tools/",
                    "*.nii.gz",
                    "*.npy",
                    "*_processed.*",
                    "*_clean.*",
                    "*_features.*"
                ],
                "description": "Processed, ML-ready datasets and features"
            },
            "models": {
                "bucket": "gs://quark-models",
                "patterns": [
                    "models/",
                    "*.pth",
                    "*.onnx", 
                    "*.ckpt",
                    "*.pkl",
                    "*_model.*",
                    "*_checkpoint.*",
                    "*.h5"
                ],
                "description": "Trained models, checkpoints, and artifacts"
            },
            "experiments": {
                "bucket": "gs://quark-experiments",
                "patterns": [
                    "reports/",
                    "logs/",
                    "results/",
                    "metrics/",
                    "*_report.*",
                    "*_results.*",
                    "*_metrics.*",
                    "*.log"
                ],
                "description": "Experimental results, reports, and logs"
            }
        }
    
    def classify_file(self, file_path: Path) -> str:
        """Classify a file into the appropriate bucket category."""
        relative_path = file_path.relative_to(self.data_dir)
        path_str = str(relative_path)
        
        # Check each bucket category
        for category, config in self.bucket_mapping.items():
            for pattern in config["patterns"]:
                if pattern.endswith("/"):
                    # Directory pattern
                    if path_str.startswith(pattern):
                        return category
                elif "*" in pattern:
                    # Wildcard pattern
                    import fnmatch
                    if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                        return category
                else:
                    # Exact match
                    if pattern in path_str:
                        return category
        
        # Default to processed if no specific match
        return "processed"
    
    def get_files_to_sync(self) -> Dict[str, List[Path]]:
        """Get all files organized by target bucket."""
        files_by_bucket = {category: [] for category in self.bucket_mapping.keys()}
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return files_by_bucket
        
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                category = self.classify_file(file_path)
                files_by_bucket[category].append(file_path)
        
        return files_by_bucket
    
    def sync_to_bucket(self, files: List[Path], bucket: str, dry_run: bool = False) -> bool:
        """Sync files to a specific GCS bucket with progress monitoring."""
        if not files:
            return True
        
        # Calculate total size for progress tracking
        total_size = sum(f.stat().st_size for f in files if f.exists() and f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        
        logger.info(f"📤 Syncing {len(files)} files ({total_size_mb:.1f} MB) to {bucket}")
        
        if dry_run:
            logger.info(f"DRY RUN: Would sync {len(files)} files ({total_size_mb:.1f} MB) to {bucket}")
            return True
        
        # Sync files with progress updates
        uploaded_count = 0
        uploaded_size = 0
        
        for i, file_path in enumerate(files, 1):
            relative_path = file_path.relative_to(self.data_dir)
            target_path = f"{bucket}/{relative_path}"
            
            # Get file size for progress tracking
            file_size = file_path.stat().st_size if file_path.exists() else 0
            file_size_mb = file_size / (1024 * 1024)
            
            sync_cmd = ["gsutil", "cp", str(file_path), target_path]
            result = subprocess.run(sync_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Failed to sync {file_path}: {result.stderr}")
                return False
            else:
                uploaded_count += 1
                uploaded_size += file_size
                uploaded_size_mb = uploaded_size / (1024 * 1024)
                
                # Show progress every 10 files or for large files
                if i % 10 == 0 or file_size_mb > 10 or i == len(files):
                    progress_pct = (uploaded_size / total_size * 100) if total_size > 0 else 0
                    logger.info(f"📤 Progress: {uploaded_count}/{len(files)} files ({uploaded_size_mb:.1f}/{total_size_mb:.1f} MB, {progress_pct:.1f}%)")
                
                logger.debug(f"✅ Synced: {file_path} → {target_path}")
        
        logger.info(f"✅ Successfully synced {uploaded_count} files ({uploaded_size_mb:.1f} MB) to {bucket}")
        return True
    
    def download_from_buckets(self, target_dir: Path = None, dry_run: bool = False) -> bool:
        """Download and merge all bucket contents back to local data directory."""
        if target_dir is None:
            target_dir = self.data_dir
        
        logger.info(f"Downloading all bucket contents to {target_dir}")
        
        for category, config in self.bucket_mapping.items():
            bucket = config["bucket"]
            logger.info(f"Downloading from {bucket}")
            
            if dry_run:
                logger.info(f"DRY RUN: Would download from {bucket}")
                continue
            
            # Download entire bucket contents
            cmd = ["gsutil", "-m", "cp", "-r", f"{bucket}/*", str(target_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to download from {bucket}: {result.stderr}")
                return False
        
        return True
    
    def get_directory_size(self, directory: Path) -> float:
        """Get directory size in MB."""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (OSError, PermissionError):
            pass
        return total_size / (1024 * 1024)  # Convert to MB
    
    def verify_sync_success(self, local_files: List[Path], bucket: str) -> bool:
        """Verify that all local files were successfully synced to GCS bucket."""
        logger.info(f"Verifying sync success for {bucket}")
        
        # Get list of files in bucket
        cmd = ["gsutil", "ls", "-r", bucket]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Failed to list bucket contents: {result.stderr}")
            return False
        
        bucket_files = set()
        for line in result.stdout.strip().split('\n'):
            if line and not line.endswith('/'):  # Skip directories
                # Extract relative path from bucket URL
                if bucket in line:
                    rel_path = line.replace(f"{bucket}/", "")
                    bucket_files.add(rel_path)
        
        # Check if all local files are in bucket
        missing_files = []
        for local_file in local_files:
            rel_path = str(local_file.relative_to(self.data_dir))
            if rel_path not in bucket_files:
                missing_files.append(rel_path)
        
        if missing_files:
            logger.warning(f"Missing files in {bucket}: {missing_files[:5]}...")
            return False
        
        logger.info(f"✅ All {len(local_files)} files verified in {bucket}")
        return True
    
    def get_sync_summary(self, synced_files: Dict[str, List[Path]]) -> Tuple[float, Dict[str, float]]:
        """Get summary of synced data sizes and heavy directories to clean."""
        total_synced = 0.0
        heavy_dirs_to_clean = {}
        
        # Calculate total synced size
        for category, files in synced_files.items():
            for file_path in files:
                if file_path.exists() and file_path.is_file():
                    total_synced += file_path.stat().st_size / (1024 * 1024)  # MB
        
        # Check heavy directories that would be cleaned
        for dir_pattern, size_threshold in self.heavy_directories.items():
            dir_path = self.data_dir / dir_pattern.rstrip('/')
            
            if not dir_path.exists():
                continue
            
            dir_size = self.get_directory_size(dir_path)
            
            if dir_size > size_threshold:
                # Verify this directory has files that were synced
                has_synced_files = False
                files_in_dir = []
                
                for category, files in synced_files.items():
                    category_files = [f for f in files if dir_pattern.rstrip('/') in str(f.relative_to(self.data_dir))]
                    if category_files:
                        has_synced_files = True
                        files_in_dir.extend(category_files)
                        break
                
                if has_synced_files:
                    heavy_dirs_to_clean[str(dir_path)] = {
                        'size': dir_size,
                        'files': files_in_dir,
                        'category': category
                    }
        
        return total_synced, heavy_dirs_to_clean
    
    def confirm_cleanup(self, total_synced: float, heavy_dirs_to_clean: Dict[str, dict]) -> bool:
        """Ask user for confirmation before cleaning up heavy directories."""
        if not heavy_dirs_to_clean:
            logger.info("ℹ️ No heavy directories to clean up")
            return True
        
        total_to_remove = sum(info['size'] for info in heavy_dirs_to_clean.values())
        
        print(f"\n🎉 Sync completed successfully!")
        print(f"📊 Data synced to GCS: {total_synced:.1f} MB")
        print(f"\n🧹 Heavy directories ready for cleanup:")
        
        for dir_path, info in heavy_dirs_to_clean.items():
            print(f"   • {dir_path}: {info['size']:.1f} MB ({len(info['files'])} files)")
        
        print(f"\n📦 Total data to be removed: {total_to_remove:.1f} MB")
        print(f"💾 This will free up {total_to_remove:.1f} MB of local storage")
        
        while True:
            response = input(f"\n❓ Remove these heavy directories? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no', '']:
                print("ℹ️ Skipping cleanup - heavy directories preserved")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    def cleanup_heavy_directories(self, heavy_dirs_to_clean: Dict[str, dict], dry_run: bool = False) -> bool:
        """Clean up heavy directories after user confirmation and verification."""
        if dry_run:
            logger.info("🧹 [DRY RUN] Would clean up heavy directories")
            return True
        
        if not heavy_dirs_to_clean:
            return True
        
        cleaned_dirs = []
        total_cleaned = 0.0
        
        for dir_path, info in heavy_dirs_to_clean.items():
            # Verify sync success before cleanup
            bucket_for_cleanup = None
            for category, bucket_info in self.bucket_mapping.items():
                if category == info['category']:
                    bucket_for_cleanup = bucket_info["bucket"]
                    break
            
            if bucket_for_cleanup:
                logger.info(f"Verifying {Path(dir_path).name} files in {bucket_for_cleanup}...")
                
                if self.verify_sync_success(info['files'], bucket_for_cleanup):
                    logger.info(f"🗑️ Removing: {dir_path} ({info['size']:.1f}MB)")
                    try:
                        shutil.rmtree(dir_path)
                        cleaned_dirs.append(Path(dir_path).name)
                        total_cleaned += info['size']
                    except Exception as e:
                        logger.error(f"❌ Failed to remove {dir_path}: {e}")
                        return False
                else:
                    logger.warning(f"❌ Skipping cleanup of {dir_path} - sync verification failed")
                    return False
        
        if cleaned_dirs:
            logger.info(f"✅ Successfully removed {len(cleaned_dirs)} directories ({total_cleaned:.1f}MB freed)")
        
        return True
    
    def show_classification_preview(self) -> None:
        """Show how files would be classified without syncing."""
        files_by_bucket = self.get_files_to_sync()
        
        print("\n🗂️  **File Classification Preview**\n")
        
        total_files = 0
        for category, files in files_by_bucket.items():
            if files:
                config = self.bucket_mapping[category]
                print(f"**{config['bucket']}** ({len(files)} files)")
                print(f"   {config['description']}")
                
                # Show first few files as examples
                for i, file_path in enumerate(files[:5]):
                    relative_path = file_path.relative_to(self.data_dir)
                    print(f"   • {relative_path}")
                
                if len(files) > 5:
                    print(f"   ... and {len(files) - 5} more files")
                print()
                
                total_files += len(files)
        
        print(f"**Total**: {total_files} files to sync")
    
    def sync_all(self, dry_run: bool = False, download: bool = False, cleanup: bool = True) -> bool:
        """Sync all local data to appropriate GCS buckets with optional cleanup."""
        if download:
            return self.download_from_buckets(dry_run=dry_run)
        
        files_by_bucket = self.get_files_to_sync()
        synced_files = {}
        
        success = True
        for category, files in files_by_bucket.items():
            if files:
                bucket = self.bucket_mapping[category]["bucket"]
                if self.sync_to_bucket(files, bucket, dry_run=dry_run):
                    synced_files[category] = files
                else:
                    success = False
        
        # Clean up heavy directories after successful sync with user confirmation
        if success and cleanup and synced_files and not dry_run:
            total_synced, heavy_dirs_to_clean = self.get_sync_summary(synced_files)
            
            if heavy_dirs_to_clean:
                if self.confirm_cleanup(total_synced, heavy_dirs_to_clean):
                    success = self.cleanup_heavy_directories(heavy_dirs_to_clean, dry_run=dry_run)
            else:
                logger.info(f"✅ Sync completed! {total_synced:.1f} MB synced to GCS")
        elif success and synced_files and dry_run:
            # Just show summary for dry runs
            total_synced, heavy_dirs_to_clean = self.get_sync_summary(synced_files)
            if heavy_dirs_to_clean:
                total_to_remove = sum(info['size'] for info in heavy_dirs_to_clean.values())
                logger.info(f"[DRY RUN] Would sync {total_synced:.1f} MB and clean up {total_to_remove:.1f} MB")
        
        return success


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Quark GCS Data Sync")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be synced without actually doing it")
    parser.add_argument("--preview", action="store_true",
                       help="Show file classification preview")
    parser.add_argument("--download", action="store_true",
                       help="Download from buckets to local (reverse sync)")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Skip automatic cleanup of heavy directories")
    parser.add_argument("--project-root", default="/Users/camdouglas/quark",
                       help="Path to Quark project root")
    
    args = parser.parse_args()
    
    syncer = QuarkDataSync(args.project_root)
    
    if args.preview:
        syncer.show_classification_preview()
    else:
        cleanup = not args.no_cleanup
        success = syncer.sync_all(dry_run=args.dry_run, download=args.download, cleanup=cleanup)
        if success:
            action = "downloaded" if args.download else "synced"
            logger.info(f"✅ Successfully {action} all data!")
        else:
            logger.error("❌ Some files failed to sync")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
