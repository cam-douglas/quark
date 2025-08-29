#!/usr/bin/env python3
"""
Simple Auto-Sync for Cursor
Automatically syncs files when needed.
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoSync:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.cloud_refs = self.project_root / ".cloud_references"
        self.cache_dir = self.project_root / ".cursor_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load cloud references
        self.cloud_files = self._load_cloud_references()
        
    def _load_cloud_references(self):
        """Load cloud file references."""
        cloud_files = {}
        if not self.cloud_refs.exists():
            return cloud_files
            
        for ref_file in self.cloud_refs.rglob("*.ref"):
            try:
                with open(ref_file, 'r') as f:
                    ref_data = json.load(f)
                cloud_files[ref_data['original_path']] = ref_data
            except Exception as e:
                logger.debug(f"Failed to read {ref_file}: {e}")
        return cloud_files
    
    def auto_sync_file(self, file_path: str) -> bool:
        """Auto-sync a file when Cursor needs it."""
        if file_path not in self.cloud_files:
            return False
        
        local_path = self.project_root / file_path
        if local_path.exists():
            return True
        
        logger.info(f"Auto-syncing: {file_path}")
        
        # Try manual upload directory first
        manual_path = self.project_root / "files_for_manual_upload" / file_path
        if manual_path.exists():
            return self._restore_file(manual_path, local_path, file_path)
        
        # Try cache
        cache_path = self.cache_dir / file_path
        if cache_path.exists():
            return self._restore_file(cache_path, local_path, file_path)
        
        return False
    
    def _restore_file(self, source_path: Path, dest_path: Path, file_path: str) -> bool:
        """Restore file from source to destination."""
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            
            if self._verify_file(dest_path, self.cloud_files[file_path]):
                # Cache for future use
                self._cache_file(file_path)
                logger.info(f"✅ Auto-synced: {file_path}")
                return True
            else:
                logger.error(f"❌ Verification failed: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to restore {file_path}: {e}")
            return False
    
    def _verify_file(self, file_path: Path, cloud_data: dict) -> bool:
        """Verify file matches cloud version."""
        try:
            if not file_path.exists():
                return False
            local_size = file_path.stat().st_size
            return local_size == cloud_data['file_size']
        except Exception:
            return False
    
    def _cache_file(self, file_path: str):
        """Cache file for future access."""
        try:
            local_path = self.project_root / file_path
            cache_path = self.cache_dir / file_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, cache_path)
        except Exception as e:
            logger.debug(f"Cache failed for {file_path}: {e}")
    
    def get_status(self) -> dict:
        """Get sync status."""
        total_files = len(self.cloud_files)
        local_files = sum(1 for f in self.cloud_files.keys() 
                         if (self.project_root / f).exists())
        
        return {
            'total_files': total_files,
            'local_files': local_files,
            'synced_percentage': (local_files / total_files * 100) if total_files > 0 else 0
        }

def main():
    """Demo the auto-sync system."""
    sync = AutoSync()
    
    print("🚀 Cursor Auto-Sync System")
    print("=" * 50)
    
    status = sync.get_status()
    print(f"Total files: {status['total_files']}")
    print(f"Local files: {status['local_files']}")
    print(f"Synced: {status['synced_percentage']:.1f}%")
    
    print(f"\n💡 Usage:")
    print(f"  # Auto-sync a file when Cursor needs it")
    print(f"  sync.auto_sync_file('results/experiments/example.png')")
    print(f"  ")
    print(f"  # Get status")
    print(f"  status = sync.get_status()")

if __name__ == "__main__":
    main()
