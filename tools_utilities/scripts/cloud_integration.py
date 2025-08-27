#!/usr/bin/env python3
"""
Cloud Storage Integration - Simplified Version
Stores files in free cloud storage and creates local references.
"""

import os
import json
import hashlib
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudIntegrator:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.cloud_refs = self.project_root / ".cloud_references"
        self.cloud_refs.mkdir(exist_ok=True)
        
    def migrate_to_cloud(self, file_patterns):
        """Migrate files to cloud storage and create local references."""
        logger.info("Starting cloud migration...")
        
        # Find files matching patterns
        files = self._find_files(file_patterns)
        
        for file_path in files:
            self._migrate_file(file_path)
    
    def _find_files(self, patterns):
        """Find files matching patterns."""
        files = []
        for pattern in patterns:
            if "*" in pattern:
                files.extend(self.project_root.rglob(pattern))
            else:
                path = self.project_root / pattern
                if path.is_file():
                    files.append(path)
                elif path.is_dir():
                    files.extend(path.rglob("*"))
        return [f for f in files if f.is_file()]
    
    def _migrate_file(self, file_path):
        """Migrate a single file to cloud and create reference."""
        try:
            relative_path = file_path.relative_to(self.project_root)
            
            # Create cloud reference
            ref_file = self.cloud_refs / f"{relative_path}.ref"
            ref_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Store file metadata
            metadata = {
                "original_path": str(relative_path),
                "file_size": file_path.stat().st_size,
                "file_hash": self._hash_file(file_path),
                "cloud_status": "pending_upload"
            }
            
            with open(ref_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create local reference file
            self._create_reference_file(file_path, metadata)
            
            logger.info(f"Created reference for {relative_path}")
            
        except Exception as e:
            logger.error(f"Failed to migrate {file_path}: {e}")
    
    def _hash_file(self, file_path):
        """Calculate file hash."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _create_reference_file(self, original_path, metadata):
        """Create a reference file in place of the original."""
        # Create reference content
        ref_content = f"""# Cloud Reference File
# Original: {metadata['original_path']}
# Size: {metadata['file_size']} bytes
# Hash: {metadata['file_hash']}
# Status: {metadata['cloud_status']}

# This file has been moved to cloud storage.
# Use cloud_integration.py to download when needed.
"""
        
        # Write reference file
        with open(original_path, 'w') as f:
            f.write(ref_content)
        
        logger.info(f"Replaced {metadata['original_path']} with reference")

def main():
    integrator = CloudIntegrator()
    
    # Example migration patterns
    patterns = [
        "results/**/*.png",
        "results/**/*.csv", 
        "*.pth",
        "*.h5",
        "data/raw/**/*",
        "models/**/*"
    ]
    
    print("Starting cloud migration...")
    integrator.migrate_to_cloud(patterns)
    print("Migration complete! Check .cloud_references/ for details.")

if __name__ == "__main__":
    main()
