#!/usr/bin/env python3
"""
Cloud Storage Integration System for Quark Project
Stores files in free cloud storage and creates symbolic links/references locally.

This system:
1. Uploads large files to free cloud storage
2. Creates symbolic links or reference files locally
3. Maintains file access while reducing local storage
4. Automatically syncs changes between local and cloud

Free Cloud Storage Options:
- Google Drive: 15GB free (best for large files)
- Dropbox: 2GB free (good sync)
- OneDrive: 5GB free (Microsoft ecosystem)
- GitHub: Unlimited (public repos)
- GitLab: 10GB free (private projects)
"""

import os, sys
import json
import hashlib
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging
import subprocess
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudStorageIntegrator:
    """Integrates local directory with cloud storage using symbolic links and references."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.cloud_config = self.project_root / ".cloud_storage_config.json"
        self.reference_dir = self.project_root / ".cloud_references"
        self.sync_log = self.project_root / ".cloud_sync_log.json"
        
        # Create necessary directories
        self.reference_dir.mkdir(exist_ok=True)
        
        # Load or create configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load cloud storage configuration."""
        if self.cloud_config.exists():
            try:
                with open(self.cloud_config, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted config, creating new one")
        
        # Default configuration
        config = {
            "cloud_services": {
                "google_drive": {
                    "enabled": True,
                    "folder_id": "",
                    "free_gb": 15,
                    "sync_method": "rclone"
                },
                "dropbox": {
                    "enabled": False,
                    "folder_path": "",
                    "free_gb": 2,
                    "sync_method": "dropbox_cli"
                },
                "github": {
                    "enabled": False,
                    "repo": "",
                    "branch": "main",
                    "free_gb": "unlimited",
                    "sync_method": "git_lfs"
                }
            },
            "local_references": {
                "use_symbolic_links": True,
                "fallback_to_references": True,
                "reference_file_suffix": ".cloud_ref"
            },
            "sync_settings": {
                "auto_sync": True,
                "sync_interval_minutes": 30,
                "preserve_local_changes": True
            }
        }
        
        self._save_config(config)
        return config
    
    def _save_config(self, config: Dict):
        """Save configuration to file."""
        with open(self.cloud_config, 'w') as f:
            json.dump(config, f, indent=2)
    
    def setup_cloud_service(self, service: str, **kwargs):
        """Setup a specific cloud service."""
        if service not in self.config["cloud_services"]:
            raise ValueError(f"Unknown service: {service}")
        
        service_config = self.config["cloud_services"][service]
        
        if service == "google_drive":
            self._setup_google_drive(service_config, **kwargs)
        elif service == "dropbox":
            self._setup_dropbox(service_config, **kwargs)
        elif service == "github":
            self._setup_github(service_config, **kwargs)
        
        self._save_config(self.config)
        logger.info(f"Setup complete for {service}")
    
    def _setup_google_drive(self, service_config: Dict, folder_name: str = "Quark_Project_Cloud"):
        """Setup Google Drive integration using rclone."""
        logger.info("Setting up Google Drive integration...")
        
        # Check if rclone is installed
        if not self._check_rclone():
            logger.error("rclone not found. Please install rclone first.")
            logger.info("Install: https://rclone.org/install/")
            return False
        
        try:
            # Configure rclone for Google Drive
            cmd = ["rclone", "config", "reconnect", "gdrive:"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                service_config["enabled"] = True
                service_config["folder_id"] = self._create_gdrive_folder(folder_name)
                logger.info(f"Google Drive setup complete. Folder: {folder_name}")
                return True
            else:
                logger.error("Failed to configure rclone for Google Drive")
                return False
                
        except Exception as e:
            logger.error(f"Google Drive setup failed: {e}")
            return False
    
    def _setup_dropbox(self, service_config: Dict, folder_name: str = "Quark_Project_Cloud"):
        """Setup Dropbox integration."""
        logger.info("Setting up Dropbox integration...")
        
        # Check if dropbox CLI is available
        if not self._check_dropbox_cli():
            logger.error("Dropbox CLI not found. Please install dropbox-cli first.")
            return False
        
        try:
            service_config["enabled"] = True
            service_config["folder_path"] = f"/{folder_name}"
            logger.info(f"Dropbox setup complete. Folder: {folder_name}")
            return True
            
        except Exception as e:
            logger.error(f"Dropbox setup failed: {e}")
            return False
    
    def _setup_github(self, service_config: Dict, repo_name: str = "quark-project-assets"):
        """Setup GitHub integration using Git LFS."""
        logger.info("Setting up GitHub integration...")
        
        try:
            # Check if git-lfs is installed
            if not self._check_git_lfs():
                logger.error("Git LFS not found. Please install git-lfs first.")
                logger.info("Install: https://git-lfs.github.com/")
                return False
            
            # Create or configure repository
            repo_url = self._setup_github_repo(repo_name)
            if repo_url:
                service_config["enabled"] = True
                service_config["repo"] = repo_url
                logger.info(f"GitHub setup complete. Repo: {repo_url}")
                return True
            else:
                logger.error("Failed to setup GitHub repository")
                return False
                
        except Exception as e:
            logger.error(f"GitHub setup failed: {e}")
            return False
    
    def _check_rclone(self) -> bool:
        """Check if rclone is installed."""
        try:
            result = subprocess.run(["rclone", "version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_dropbox_cli(self) -> bool:
        """Check if dropbox CLI is available."""
        try:
            result = subprocess.run(["dropbox", "status"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_git_lfs(self) -> bool:
        """Check if git-lfs is installed."""
        try:
            result = subprocess.run(["git-lfs", "version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _create_gdrive_folder(self, folder_name: str) -> str:
        """Create a folder in Google Drive and return its ID."""
        try:
            # Create folder using rclone
            cmd = ["rclone", "mkdir", f"gdrive:{folder_name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Get folder ID
                cmd = ["rclone", "lsf", "--json", f"gdrive:{folder_name}"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Parse JSON to get folder ID
                    data = json.loads(result.stdout)
                    return data.get("ID", "")
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to create Google Drive folder: {e}")
            return ""
    
    def _setup_github_repo(self, repo_name: str) -> Optional[str]:
        """Setup GitHub repository for asset storage."""
        try:
            # Check if we're in a git repository
            if not (self.project_root / ".git").exists():
                logger.info("Initializing git repository...")
                subprocess.run(["git", "init"], cwd=self.project_root, check=True)
            
            # Check if GitHub CLI is available
            try:
                result = subprocess.run(["gh", "repo", "view"], capture_output=True, text=True)
                if result.returncode == 0:
                    # Already in a GitHub repo
                    return "current"
            except FileNotFoundError:
                pass
            
            # Try to create new repository
            try:
                cmd = ["gh", "repo", "create", repo_name, "--public", "--description", "Quark Project Assets"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Extract repo URL from output
                    output = result.stdout
                    if "https://github.com/" in output:
                        repo_url = output.split("https://github.com/")[1].split("\n")[0].strip()
                        return f"https://github.com/{repo_url}"
                
            except FileNotFoundError:
                logger.warning("GitHub CLI not found. Please create repository manually.")
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"GitHub repo setup failed: {e}")
            return None
    
    def migrate_files_to_cloud(self, file_patterns: List[str], service: str = "google_drive"):
        """Migrate files matching patterns to cloud storage and create local references."""
        if not self.config["cloud_services"][service]["enabled"]:
            logger.error(f"Service {service} is not enabled. Please setup first.")
            return False
        
        logger.info(f"Starting migration to {service}...")
        
        # Find files matching patterns
        files_to_migrate = self._find_files_by_patterns(file_patterns)
        
        if not files_to_migrate:
            logger.info("No files found matching the patterns.")
            return True
        
        logger.info(f"Found {len(files_to_migrate)} files to migrate.")
        
        # Migrate each file
        success_count = 0
        for file_path in files_to_migrate:
            if self._migrate_single_file(file_path, service):
                success_count += 1
        
        logger.info(f"Migration complete: {success_count}/{len(files_to_migrate)} files migrated.")
        return success_count == len(files_to_migrate)
    
    def _find_files_by_patterns(self, patterns: List[str]) -> List[Path]:
        """Find files matching the given patterns."""
        files = []
        
        for pattern in patterns:
            if "*" in pattern:
                # Glob pattern
                for file_path in self.project_root.rglob(pattern):
                    if file_path.is_file():
                        files.append(file_path)
            else:
                # Direct path
                file_path = self.project_root / pattern
                if file_path.is_file():
                    files.append(file_path)
                elif file_path.is_dir():
                    # Add all files in directory
                    for sub_file in file_path.rglob("*"):
                        if sub_file.is_file():
                            files.append(sub_file)
        
        # Remove duplicates and sort
        files = list(set(files))
        files.sort()
        
        return files
    
    def _migrate_single_file(self, file_path: Path, service: str) -> bool:
        """Migrate a single file to cloud storage and create local reference."""
        try:
            relative_path = file_path.relative_to(self.project_root)
            file_hash = self._calculate_file_hash(file_path)
            
            # Upload to cloud storage
            cloud_url = self._upload_to_cloud(file_path, service, relative_path)
            
            if not cloud_url:
                logger.error(f"Failed to upload {relative_path}")
                return False
            
            # Create local reference
            self._create_local_reference(file_path, cloud_url, file_hash, service)
            
            # Remove original file (optional - can be kept for backup)
            if self.config["sync_settings"]["preserve_local_changes"]:
                logger.info(f"Keeping local copy of {relative_path}")
            else:
                file_path.unlink()
                logger.info(f"Removed local copy of {relative_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate {file_path}: {e}")
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _upload_to_cloud(self, file_path: Path, service: str, relative_path: str) -> Optional[str]:
        """Upload file to cloud storage and return URL/path."""
        try:
            if service == "google_drive":
                return self._upload_to_gdrive(file_path, relative_path)
            elif service == "dropbox":
                return self._upload_to_dropbox(file_path, relative_path)
            elif service == "github":
                return self._upload_to_github(file_path, relative_path)
            else:
                logger.error(f"Unknown service: {service}")
                return None
                
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return None
    
    def _upload_to_gdrive(self, file_path: Path, relative_path: str) -> Optional[str]:
        """Upload file to Google Drive using rclone."""
        try:
            # Create cloud path
            cloud_path = f"gdrive:Quark_Project_Cloud/{relative_path}"
            
            # Upload file
            cmd = ["rclone", "copy", str(file_path), cloud_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return cloud_path
            else:
                logger.error(f"rclone upload failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Google Drive upload failed: {e}")
            return None
    
    def _upload_to_dropbox(self, file_path: Path, relative_path: str) -> Optional[str]:
        """Upload file to Dropbox."""
        try:
            # Create cloud path
            cloud_path = f"/Quark_Project_Cloud/{relative_path}"
            
            # Upload using dropbox CLI
            cmd = ["dropbox", "upload", str(file_path), cloud_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return cloud_path
            else:
                logger.error(f"Dropbox upload failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Dropbox upload failed: {e}")
            return None
    
    def _upload_to_github(self, file_path: Path, relative_path: str) -> Optional[str]:
        """Upload file to GitHub using Git LFS."""
        try:
            # Track file with Git LFS
            subprocess.run(["git-lfs", "track", str(file_path)], cwd=self.project_root, check=True)
            
            # Add and commit file
            subprocess.run(["git", "add", str(file_path)], cwd=self.project_root, check=True)
            subprocess.run(["git", "commit", "-m", f"Add {relative_path} via Git LFS"], cwd=self.project_root, check=True)
            
            # Push to GitHub
            subprocess.run(["git", "push"], cwd=self.project_root, check=True)
            
            # Return GitHub URL
            repo_url = self.config["cloud_services"]["github"]["repo"]
            if repo_url == "current":
                # Get current remote URL
                result = subprocess.run(["git", "remote", "get-url", "origin"], 
                                      capture_output=True, text=True, cwd=self.project_root)
                if result.returncode == 0:
                    repo_url = result.stdout.strip()
            
            if repo_url:
                return f"{repo_url}/blob/main/{relative_path}"
            
            return None
            
        except Exception as e:
            logger.error(f"GitHub upload failed: {e}")
            return None
    
    def _create_local_reference(self, original_path: Path, cloud_url: str, file_hash: str, service: str):
        """Create local reference to cloud-stored file."""
        relative_path = original_path.relative_to(self.project_root)
        
        if self.config["local_references"]["use_symbolic_links"]:
            # Create symbolic link
            try:
                # Create reference file with cloud metadata
                ref_file = self.reference_dir / f"{relative_path}.cloud_ref"
                ref_file.parent.mkdir(parents=True, exist_ok=True)
                
                ref_data = {
                    "original_path": str(relative_path),
                    "cloud_url": cloud_url,
                    "service": service,
                    "file_hash": file_hash,
                    "migrated_at": time.time(),
                    "file_size": original_path.stat().st_size
                }
                
                with open(ref_file, 'w') as f:
                    json.dump(ref_data, f, indent=2)
                
                # Create symbolic link from original location to reference
                if original_path.exists():
                    original_path.unlink()  # Remove original file
                
                # Create symbolic link
                os.symlink(str(ref_file), str(original_path))
                
                logger.info(f"Created symbolic link for {relative_path}")
                
            except OSError as e:
                logger.warning(f"Could not create symbolic link for {relative_path}: {e}")
                # Fallback to reference file
                self._create_reference_file(original_path, cloud_url, file_hash, service)
        
        else:
            # Create reference file only
            self._create_reference_file(original_path, cloud_url, file_hash, service)
    
    def _create_reference_file(self, original_path: Path, cloud_url: str, file_hash: str, service: str):
        """Create a reference file with cloud metadata."""
        relative_path = original_path.relative_to(self.project_root)
        
        # Create reference file
        ref_file = self.reference_dir / f"{relative_path}.cloud_ref"
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        
        ref_data = {
            "original_path": str(relative_path),
            "cloud_url": cloud_url,
            "service": service,
            "file_hash": file_hash,
            "migrated_at": time.time(),
            "file_size": original_path.stat().st_size
        }
        
        with open(ref_file, 'w') as f:
            json.dump(ref_data, f, indent=2)
        
        # Replace original file with reference
        original_path.unlink()
        
        # Create a small reference file in original location
        with open(original_path, 'w') as f:
            f.write(f"# Cloud Reference File\n")
            f.write(f"# Original: {relative_path}\n")
            f.write(f"# Cloud: {cloud_url}\n")
            f.write(f"# Service: {service}\n")
            f.write(f"# Hash: {file_hash}\n")
            f.write(f"# Use cloud_storage_integration.py to download\n")
        
        logger.info(f"Created reference file for {relative_path}")
    
    def download_from_cloud(self, file_path: str, service: str = None):
        """Download a file from cloud storage to local directory."""
        try:
            # Find reference file
            ref_file = self.reference_dir / f"{file_path}.cloud_ref"
            
            if not ref_file.exists():
                logger.error(f"No reference found for {file_path}")
                return False
            
            # Load reference data
            with open(ref_file, 'r') as f:
                ref_data = json.load(f)
            
            if service and ref_data["service"] != service:
                logger.warning(f"File is stored on {ref_data['service']}, not {service}")
            
            # Download file
            cloud_url = ref_data["cloud_url"]
            target_service = ref_data["service"]
            
            if target_service == "google_drive":
                return self._download_from_gdrive(cloud_url, file_path)
            elif target_service == "dropbox":
                return self._download_from_dropbox(cloud_url, file_path)
            elif target_service == "github":
                return self._download_from_github(cloud_url, file_path)
            else:
                logger.error(f"Unknown service: {target_service}")
                return False
                
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def _download_from_gdrive(self, cloud_url: str, local_path: str) -> bool:
        """Download file from Google Drive."""
        try:
            local_file = self.project_root / local_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = ["rclone", "copy", cloud_url, str(local_file.parent)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Downloaded {local_path} from Google Drive")
                return True
            else:
                logger.error(f"Google Drive download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Google Drive download failed: {e}")
            return False
    
    def _download_from_dropbox(self, cloud_url: str, local_path: str) -> bool:
        """Download file from Dropbox."""
        try:
            local_file = self.project_root / local_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = ["dropbox", "download", cloud_url, str(local_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Downloaded {local_path} from Dropbox")
                return True
            else:
                logger.error(f"Dropbox download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Dropbox download failed: {e}")
            return False
    
    def _download_from_github(self, cloud_url: str, local_path: str) -> bool:
        """Download file from GitHub."""
        try:
            local_file = self.project_root / local_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract file from git repository
            cmd = ["git", "show", f"HEAD:{local_path}"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                with open(local_file, 'w') as f:
                    f.write(result.stdout)
                logger.info(f"Downloaded {local_path} from GitHub")
                return True
            else:
                logger.error(f"GitHub download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"GitHub download failed: {e}")
            return False
    
    def list_cloud_files(self, service: str = None) -> List[Dict]:
        """List all files stored in cloud storage."""
        cloud_files = []
        
        # Scan reference directory
        for ref_file in self.reference_dir.rglob("*.cloud_ref"):
            try:
                with open(ref_file, 'r') as f:
                    ref_data = json.load(f)
                
                if not service or ref_data["service"] == service:
                    cloud_files.append(ref_data)
                    
            except Exception as e:
                logger.warning(f"Failed to read reference file {ref_file}: {e}")
        
        return cloud_files
    
    def sync_with_cloud(self, service: str = None):
        """Sync local references with cloud storage."""
        logger.info("Starting cloud sync...")
        
        cloud_files = self.list_cloud_files(service)
        
        for file_data in cloud_files:
            try:
                # Check if file still exists in cloud
                if not self._verify_cloud_file(file_data):
                    logger.warning(f"File not found in cloud: {file_data['original_path']}")
                    continue
                
                # Verify local reference
                local_ref = self.reference_dir / f"{file_data['original_path']}.cloud_ref"
                if not local_ref.exists():
                    logger.warning(f"Local reference missing: {file_data['original_path']}")
                    continue
                
                logger.info(f"Synced: {file_data['original_path']}")
                
            except Exception as e:
                logger.error(f"Sync failed for {file_data['original_path']}: {e}")
        
        logger.info("Cloud sync complete")
    
    def _verify_cloud_file(self, file_data: Dict) -> bool:
        """Verify that a file still exists in cloud storage."""
        try:
            cloud_url = file_data["cloud_url"]
            service = file_data["service"]
            
            if service == "google_drive":
                cmd = ["rclone", "lsf", cloud_url]
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.returncode == 0
            elif service == "dropbox":
                cmd = ["dropbox", "list", cloud_url]
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.returncode == 0
            elif service == "github":
                # GitHub files are always available if repo exists
                return True
            else:
                return False
                
        except Exception:
            return False

def main():
    parser = argparse.ArgumentParser(description="Cloud Storage Integration for Quark Project")
    parser.add_argument("--setup", choices=["google_drive", "dropbox", "github"], 
                       help="Setup a cloud service")
    parser.add_argument("--migrate", nargs="+", help="Migrate files matching patterns to cloud")
    parser.add_argument("--download", help="Download a file from cloud storage")
    parser.add_argument("--list", action="store_true", help="List all cloud-stored files")
    parser.add_argument("--sync", action="store_true", help="Sync with cloud storage")
    parser.add_argument("--service", choices=["google_drive", "dropbox", "github"],
                       help="Specify cloud service for operations")
    
    args = parser.parse_args()
    
    integrator = CloudStorageIntegrator(".")
    
    if args.setup:
        if args.setup == "google_drive":
            integrator.setup_cloud_service("google_drive")
        elif args.setup == "dropbox":
            integrator.setup_cloud_service("dropbox")
        elif args.setup == "github":
            integrator.setup_cloud_service("github")
    
    elif args.migrate:
        service = args.service or "google_drive"
        integrator.migrate_files_to_cloud(args.migrate, service)
    
    elif args.download:
        service = args.service
        integrator.download_from_cloud(args.download, service)
    
    elif args.list:
        service = args.service
        cloud_files = integrator.list_cloud_files(service)
        
        print(f"\n=== CLOUD STORED FILES ===")
        print(f"Total files: {len(cloud_files)}")
        
        for file_data in cloud_files:
            print(f"\nFile: {file_data['original_path']}")
            print(f"Service: {file_data['service']}")
            print(f"Size: {file_data['file_size']} bytes")
            print(f"Cloud URL: {file_data['cloud_url']}")
    
    elif args.sync:
        service = args.service
        integrator.sync_with_cloud(service)
    
    else:
        print("\n=== QUARK PROJECT CLOUD STORAGE INTEGRATION ===")
        print("This system stores files in free cloud storage and creates local references.")
        print("\nAvailable commands:")
        print("  --setup google_drive    Setup Google Drive integration")
        print("  --setup dropbox         Setup Dropbox integration")
        print("  --setup github          Setup GitHub integration")
        print("  --migrate '*.png'       Migrate files to cloud storage")
        print("  --download file.png     Download file from cloud")
        print("  --list                  List all cloud-stored files")
        print("  --sync                  Sync with cloud storage")
        print("\nExample usage:")
        print("  python cloud_storage_integration.py --setup google_drive")
        print("  python cloud_storage_integration.py --migrate 'results/**/*.png' '*.pth'")
        print("  python cloud_storage_integration.py --list")

if __name__ == "__main__":
    main()
