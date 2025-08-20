#!/usr/bin/env python3
"""
Cloud Storage Migration Script for Quark Project
Moves large files to free cloud storage to optimize Cursor indexing performance.

Free Cloud Storage Options:
- Google Drive: 15GB free
- Dropbox: 2GB free  
- OneDrive: 5GB free
- GitHub Releases: Unlimited for public repos
- GitLab: 10GB free
- AWS S3: 5GB free tier
- Google Cloud Storage: 5GB free tier

Usage:
    python cloud_storage_migration.py --service google-drive --dry-run
    python cloud_storage_migration.py --service dropbox --upload
"""

import os, sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudStorageMigrator:
    """Manages migration of large files to free cloud storage services."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.migration_log = self.project_root / "cloud_migration_log.json"
        self.migration_status = self._load_migration_status()
        
    def _load_migration_status(self) -> Dict:
        """Load existing migration status from log file."""
        if self.migration_log.exists():
            try:
                with open(self.migration_log, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted migration log, starting fresh")
        return {"migrations": [], "total_size_moved": 0, "files_moved": 0}
    
    def _save_migration_status(self):
        """Save current migration status to log file."""
        with open(self.migration_log, 'w') as f:
            json.dump(self.migration_status, f, indent=2)
    
    def analyze_storage_usage(self) -> Dict:
        """Analyze current storage usage and identify migration candidates."""
        logger.info("Analyzing storage usage...")
        
        total_size = 0
        migration_candidates = []
        
        # File patterns to consider for migration (from ................................................cursorignore)
        migration_patterns = [
            "results/experiments/*.png",
            "results/experiments/*.csv", 
            "results/experiments/*.html",
            "results/training/*.png",
            "results/training/*.csv",
            "results/training/*.html",
            "result_images/",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.svg",
            "data/raw/",
            "data/processed/",
            "*.csv",
            "*.jsonl",
            "*.pkl",
            "*.h5",
            "*.hdf5",
            "models/",
            "*.pth",
            "*.pt",
            "*.ckpt",
            "*.pb",
            "logs/",
            "*.log",
            "*.txt"
        ]
        
        # Scan for large files
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and common exclusions
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', '__pycache__', '.git']]
            
            for file in files:
                file_path = Path(root) / file
                try:
                    file_size = file_path.stat().st_size
                    
                    # Consider files larger than 100KB for migration
                    if file_size > 100 * 1024:  # 100KB threshold
                        relative_path = file_path.relative_to(self.project_root)
                        
                        # Check if file matches migration patterns
                        should_migrate = any(self._matches_pattern(str(relative_path), pattern) 
                                           for pattern in migration_patterns)
                        
                        if should_migrate:
                            migration_candidates.append({
                                "path": str(relative_path),
                                "size": file_size,
                                "size_mb": round(file_size / (1024 * 1024), 2)
                            })
                            total_size += file_size
                            
                except (OSError, PermissionError):
                    continue
        
        # Sort by size (largest first)
        migration_candidates.sort(key=lambda x: x["size"], reverse=True)
        
        analysis = {
            "total_size_bytes": total_size,
            "total_size_gb": round(total_size / (1024**3), 2),
            "candidates_count": len(migration_candidates),
            "candidates": migration_candidates[:100],  # Top 100 largest files
            "estimated_cursor_indexing_improvement": "Significant - large binary files excluded"
        }
        
        return analysis
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches a glob pattern."""
        from fnmatch import fnmatch
        return fnmatch(file_path, pattern)
    
    def create_migration_plan(self, service: str) -> Dict:
        """Create a migration plan for the specified cloud service."""
        analysis = self.analyze_storage_usage()
        
        # Service-specific limits and recommendations
        service_limits = {
            "google-drive": {"free_gb": 15, "recommended": "Best for large files, good integration"},
            "dropbox": {"free_gb": 2, "recommended": "Good for smaller files, excellent sync"},
            "onedrive": {"free_gb": 5, "recommended": "Good balance, Microsoft ecosystem"},
            "github": {"free_gb": "unlimited", "recommended": "Best for public projects, version control"},
            "gitlab": {"free_gb": 10, "recommended": "Good for private projects, CI/CD integration"},
            "aws-s3": {"free_gb": 5, "recommended": "Professional, good for automation"},
            "gcp-storage": {"free_gb": 5, "recommended": "Professional, good for ML workflows"}
        }
        
        if service not in service_limits:
            raise ValueError(f"Unknown service: {service}")
        
        service_info = service_limits[service]
        total_gb = analysis["total_size_gb"]
        
        plan = {
            "service": service,
            "service_limits": service_info,
            "current_usage": total_gb,
            "migration_strategy": self._generate_migration_strategy(service, total_gb, service_info),
            "estimated_time": self._estimate_migration_time(analysis["candidates_count"]),
            "files_to_migrate": analysis["candidates"][:50],  # Top 50 files
            "priority_order": self._prioritize_files(analysis["candidates"])
        }
        
        return plan
    
    def _generate_migration_strategy(self, service: str, total_gb: float, service_info: Dict) -> str:
        """Generate migration strategy based on service and file sizes."""
        if service_info["free_gb"] == "unlimited":
            return f"Full migration to {service} - unlimited storage available"
        
        free_gb = service_info["free_gb"]
        
        if total_gb <= free_gb:
            return f"Full migration possible - {total_gb}GB fits within {free_gb}GB free tier"
        else:
            return f"Partial migration - prioritize largest files. {free_gb}GB free, {total_gb}GB total"
    
    def _estimate_migration_time(self, file_count: int) -> str:
        """Estimate migration time based on file count."""
        if file_count <= 100:
            return "1-2 hours (manual upload)"
        elif file_count <= 500:
            return "2-4 hours (batch upload)"
        else:
            return "4+ hours (automated upload recommended)"
    
    def _prioritize_files(self, candidates: List[Dict]) -> List[str]:
        """Prioritize files for migration based on size and type."""
        # Priority: largest files first, then by type
        priority_order = []
        
        # Group by file type
        by_type = {}
        for candidate in candidates:
            ext = Path(candidate["path"]).suffix.lower()
            if ext not in by_type:
                by_type[ext] = []
            by_type[ext].append(candidate)
        
        # Priority: images, models, data files, logs
        priority_extensions = ['.png', '.jpg', '.jpeg', '.pth', '.pt', '.h5', '.csv', '.log']
        
        for ext in priority_extensions:
            if ext in by_type:
                # Sort by size within each type
                by_type[ext].sort(key=lambda x: x["size"], reverse=True)
                for candidate in by_type[ext]:
                    priority_order.append(candidate["path"])
        
        return priority_order
    
    def generate_upload_instructions(self, service: str) -> str:
        """Generate step-by-step upload instructions for the specified service."""
        instructions = {
            "google-drive": """
Google Drive Migration Instructions:
1. Go to drive.google.com
2. Create a new folder: "Quark_Project_Backup_[DATE]"
3. Upload files in batches of 100MB for best performance
4. Use Google Drive for Desktop for automatic sync
5. Share folder with your Google account for access
            """,
            
            "dropbox": """
Dropbox Migration Instructions:
1. Go to dropbox.com
2. Create folder: "Quark_Project_Backup"
3. Upload files (2GB free limit)
4. Use Dropbox desktop app for sync
5. Consider upgrading to paid plan for more storage
            """,
            
            "github": """
GitHub Releases Migration Instructions:
1. Create a new repository: "quark-project-assets"
2. Use GitHub CLI: gh release create v1.0.0 --title "Project Assets"
3. Upload large files as release assets
4. Update .cursorignore to reference GitHub URLs
5. Use git-lfs for files >100MB
            """,
            
            "aws-s3": """
AWS S3 Migration Instructions:
1. Create S3 bucket: "quark-project-assets-[YOUR-NAME]"
2. Use AWS CLI: aws s3 sync ./results s3://bucket-name/results
3. Set bucket to public-read for free access
4. Use lifecycle policies to manage costs
5. Consider CloudFront for faster access
            """
        }
        
        return instructions.get(service, f"Manual upload to {service} recommended")
    
    def create_backup_script(self, service: str) -> str:
        """Create a backup script for the specified service."""
        script_content = f"""#!/bin/bash
# Backup script for Quark project - {service}
# Generated by cloud_storage_migration.py

PROJECT_ROOT="{self.project_root}"
BACKUP_DIR="$PROJECT_ROOT/cloud_backup_{service}_$(date +%Y%m%d_%H%M%S)"

echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Copy large files to backup directory
echo "Copying large files for {service} upload..."

# Results and experiments
if [ -d "$PROJECT_ROOT/results" ]; then
    cp -r "$PROJECT_ROOT/results" "$BACKUP_DIR/"
fi

# Images and visualizations
if [ -d "$PROJECT_ROOT/result_images" ]; then
    cp -r "$PROJECT_ROOT/result_images" "$BACKUP_DIR/"
fi

# Data files
if [ -d "$PROJECT_ROOT/data" ]; then
    cp -r "$PROJECT_ROOT/data" "$BACKUP_DIR/"
fi

# Models
if [ -d "$PROJECT_ROOT/models" ]; then
    cp -r "$PROJECT_ROOT/models" "$BACKUP_DIR/"
fi

# Logs
if [ -d "$PROJECT_ROOT/logs" ]; then
    cp -r "$PROJECT_ROOT/logs" "$BACKUP_DIR/"
fi

echo "Backup complete: $BACKUP_DIR"
echo "Upload this directory to {service}"
echo "After successful upload, you can delete the local backup"
"""
        
        script_path = self.project_root / f"backup_for_{service}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return str(script_path)

def main():
    parser = argparse.ArgumentParser(description="Migrate large files to free cloud storage")
    parser.add_argument("--service", choices=["google-drive", "dropbox", "onedrive", "github", "gitlab", "aws-s3", "gcp-storage"],
                       default="google-drive", help="Target cloud storage service")
    parser.add_argument("--analyze", action="store_true", help="Analyze storage usage only")
    parser.add_argument("--plan", action="store_true", help="Create migration plan")
    parser.add_argument("--instructions", action="store_true", help="Show upload instructions")
    parser.add_argument("--backup-script", action="store_true", help="Generate backup script")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without doing it")
    
    args = parser.parse_args()
    
    migrator = CloudStorageMigrator(".")
    
    if args.analyze:
        analysis = migrator.analyze_storage_usage()
        print("\n=== STORAGE ANALYSIS ===")
        print(f"Total size: {analysis['total_size_gb']}GB")
        print(f"Migration candidates: {analysis['candidates_count']} files")
        print(f"Cursor indexing improvement: {analysis['estimated_cursor_indexing_improvement']}")
        
        print("\n=== TOP 10 LARGEST FILES ===")
        for i, candidate in enumerate(analysis['candidates'][:10], 1):
            print(f"{i:2d}. {candidate['path']} ({candidate['size_mb']}MB)")
    
    elif args.plan:
        plan = migrator.create_migration_plan(args.service)
        print(f"\n=== MIGRATION PLAN FOR {args.service.upper()} ===")
        print(f"Strategy: {plan['migration_strategy']}")
        print(f"Estimated time: {plan['estimated_time']}")
        print(f"Service limits: {plan['service_limits']}")
        
        print("\n=== PRIORITY FILES ===")
        for i, file_path in enumerate(plan['priority_order'][:20], 1):
            print(f"{i:2d}. {file_path}")
    
    elif args.instructions:
        instructions = migrator.generate_upload_instructions(args.service)
        print(f"\n=== UPLOAD INSTRUCTIONS FOR {args.service.upper()} ===")
        print(instructions)
    
    elif args.backup_script:
        script_path = migrator.create_backup_script(args.service)
        print(f"\n=== BACKUP SCRIPT CREATED ===")
        print(f"Script: {script_path}")
        print(f"Run: ./{os.path.basename(script_path)}")
    
    else:
        # Default: show analysis and plan
        analysis = migrator.analyze_storage_usage()
        plan = migrator.create_migration_plan(args.service)
        
        print("\n=== QUARK PROJECT CLOUD MIGRATION ===")
        print(f"Current project size: {analysis['total_size_gb']}GB")
        print(f"Files to migrate: {analysis['candidates_count']}")
        print(f"Target service: {args.service}")
        print(f"Migration strategy: {plan['migration_strategy']}")
        
        print("\n=== NEXT STEPS ===")
        print("1. Run: python cloud_storage_migration.py --analyze")
        print("2. Run: python cloud_storage_migration.py --plan --service [SERVICE]")
        print("3. Run: python cloud_storage_migration.py --instructions --service [SERVICE]")
        print("4. Run: python cloud_storage_migration.py --backup-script --service [SERVICE]")
        print("5. Upload files to cloud storage")
        print("6. Update .cursorignore to exclude uploaded files")

if __name__ == "__main__":
    main()
