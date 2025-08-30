#!/usr/bin/env python3
"""
Data Migration Manager for Quark
Intelligently migrates large data directory (18GB) to S3 with progress tracking
"""

import os
import boto3
import shutil
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

class DataMigrationManager:
    """Manages migration of Quark data directory to S3"""
    
    def __init__(self, bucket_name: str = "quark-tokyo-bucket", region: str = "ap-northeast-1"):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Source and destination paths
        self.source_data_dir = Path("/Users/camdouglas/quark/data")
        self.new_data_dir = Path("/Users/camdouglas/quark/datasets")  # More appropriate name
        self.s3_data_prefix = "datasets/quark_data"
        
        # Migration tracking
        self.migration_log = []
        self.uploaded_files = {}
        self.failed_files = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load previous migration state if exists
        self._load_migration_state()
    
    def _load_migration_state(self):
        """Load previous migration state to resume if interrupted"""
        state_file = Path.home() / ".quark" / "migration_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.uploaded_files = state.get("uploaded_files", {})
                    self.failed_files = state.get("failed_files", {})
                self.logger.info(f"Loaded migration state: {len(self.uploaded_files)} files already uploaded")
            except Exception as e:
                self.logger.warning(f"Could not load migration state: {e}")
    
    def _save_migration_state(self):
        """Save migration state for resumability"""
        state_file = Path.home() / ".quark" / "migration_state.json"
        state_file.parent.mkdir(exist_ok=True)
        
        state = {
            "timestamp": datetime.now().isoformat(),
            "uploaded_files": self.uploaded_files,
            "failed_files": self.failed_files,
            "migration_log": self.migration_log[-100:]  # Keep last 100 log entries
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save migration state: {e}")
    
    def analyze_data_directory(self) -> Dict[str, Any]:
        """Analyze the data directory structure and content"""
        
        if not self.source_data_dir.exists():
            return {"error": "Data directory does not exist"}
        
        analysis = {
            "total_size_bytes": 0,
            "total_files": 0,
            "file_types": {},
            "directory_structure": {},
            "large_files": [],  # Files > 100MB
            "estimated_upload_time": 0
        }
        
        self.logger.info("Analyzing data directory...")
        
        for root, dirs, files in os.walk(self.source_data_dir):
            root_path = Path(root)
            relative_root = root_path.relative_to(self.source_data_dir)
            
            if str(relative_root) != ".":
                analysis["directory_structure"][str(relative_root)] = len(files)
            
            for file in files:
                file_path = root_path / file
                try:
                    file_size = file_path.stat().st_size
                    analysis["total_size_bytes"] += file_size
                    analysis["total_files"] += 1
                    
                    # Track file types
                    extension = file_path.suffix.lower()
                    analysis["file_types"][extension] = analysis["file_types"].get(extension, 0) + 1
                    
                    # Track large files
                    if file_size > 100 * 1024 * 1024:  # 100MB
                        analysis["large_files"].append({
                            "path": str(file_path.relative_to(self.source_data_dir)),
                            "size_mb": file_size / (1024 * 1024)
                        })
                
                except Exception as e:
                    self.logger.warning(f"Could not analyze file {file_path}: {e}")
        
        # Convert bytes to GB
        analysis["total_size_gb"] = analysis["total_size_bytes"] / (1024**3)
        
        # Estimate upload time (assuming 10MB/s average)
        analysis["estimated_upload_time"] = analysis["total_size_bytes"] / (10 * 1024 * 1024)  # seconds
        
        return analysis
    
    def create_migration_plan(self) -> Dict[str, Any]:
        """Create a smart migration plan based on file analysis"""
        
        analysis = self.analyze_data_directory()
        if "error" in analysis:
            return analysis
        
        plan = {
            "migration_strategy": "progressive",
            "phases": [],
            "total_size_gb": analysis["total_size_gb"],
            "estimated_duration_hours": analysis["estimated_upload_time"] / 3600,
            "cost_estimate": self._estimate_s3_cost(analysis["total_size_bytes"]),
            "recommendations": []
        }
        
        # Phase 1: Essential data first (smaller files)
        plan["phases"].append({
            "phase": 1,
            "name": "Essential Models and Processed Data",
            "description": "Upload processed datasets and essential models first",
            "file_patterns": ["*.npy", "*.pt", "*.pkl", "*.json"],
            "priority": "high",
            "estimated_size_gb": analysis["total_size_gb"] * 0.7  # Most data is processed
        })
        
        # Phase 2: Raw data
        plan["phases"].append({
            "phase": 2,
            "name": "Raw Training Data",
            "description": "Upload raw AMASS and SMPL data",
            "file_patterns": ["*.npz", "*.txt", "*.png"],
            "priority": "medium",
            "estimated_size_gb": analysis["total_size_gb"] * 0.3
        })
        
        # Add recommendations
        if analysis["total_size_gb"] > 15:
            plan["recommendations"].append("⚠️ Large dataset - consider compression before upload")
        
        if len(analysis["large_files"]) > 0:
            plan["recommendations"].append(f"📦 {len(analysis['large_files'])} files > 100MB - will use multipart upload")
        
        plan["recommendations"].append("💾 Data will be moved locally to /Users/camdouglas/quark/datasets")
        plan["recommendations"].append("☁️ S3 backup enables streaming access from any instance")
        
        return plan
    
    def _estimate_s3_cost(self, total_bytes: int) -> Dict[str, float]:
        """Estimate S3 storage and transfer costs"""
        
        # S3 Standard pricing (approximate)
        storage_cost_per_gb_month = 0.023  # USD
        transfer_cost_per_gb = 0.0  # First 100GB free from EC2 to S3
        
        size_gb = total_bytes / (1024**3)
        
        return {
            "storage_monthly_usd": size_gb * storage_cost_per_gb_month,
            "transfer_usd": 0.0,  # Free from EC2
            "total_first_month_usd": size_gb * storage_cost_per_gb_month
        }
    
    def move_data_locally(self) -> Dict[str, Any]:
        """Move data directory to more appropriate location"""
        
        if not self.source_data_dir.exists():
            return {"error": "Source data directory does not exist"}
        
        if self.new_data_dir.exists():
            return {"error": f"Destination directory already exists: {self.new_data_dir}"}
        
        try:
            self.logger.info(f"Moving {self.source_data_dir} to {self.new_data_dir}")
            
            # Create parent directory
            self.new_data_dir.parent.mkdir(exist_ok=True)
            
            # Move the directory
            shutil.move(str(self.source_data_dir), str(self.new_data_dir))
            
            self.migration_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "moved_locally",
                "from": str(self.source_data_dir),
                "to": str(self.new_data_dir),
                "status": "success"
            })
            
            return {
                "status": "success",
                "moved_from": str(self.source_data_dir),
                "moved_to": str(self.new_data_dir),
                "message": "Data directory moved successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to move data directory: {e}")
            return {"error": str(e)}
    
    def upload_file_to_s3(self, local_path: Path, s3_key: str) -> bool:
        """Upload a single file to S3 with error handling"""
        
        # Skip if already uploaded
        if s3_key in self.uploaded_files:
            return True
        
        try:
            file_size = local_path.stat().st_size
            
            # Use multipart upload for large files
            if file_size > 100 * 1024 * 1024:  # 100MB
                self._multipart_upload(local_path, s3_key)
            else:
                self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            
            # Track successful upload
            self.uploaded_files[s3_key] = {
                "local_path": str(local_path),
                "size_bytes": file_size,
                "uploaded_at": datetime.now().isoformat()
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload {local_path}: {e}")
            self.failed_files[s3_key] = {"error": str(e), "local_path": str(local_path)}
            return False
    
    def _multipart_upload(self, local_path: Path, s3_key: str):
        """Upload large file using multipart upload"""
        
        # Initialize multipart upload
        response = self.s3_client.create_multipart_upload(
            Bucket=self.bucket_name,
            Key=s3_key
        )
        upload_id = response['UploadId']
        
        try:
            parts = []
            part_size = 100 * 1024 * 1024  # 100MB parts
            file_size = local_path.stat().st_size
            
            with open(local_path, 'rb') as f:
                part_number = 1
                
                with tqdm(desc=f"Uploading {local_path.name}", total=file_size, unit='B', unit_scale=True) as pbar:
                    while True:
                        data = f.read(part_size)
                        if not data:
                            break
                        
                        # Upload part
                        response = self.s3_client.upload_part(
                            Bucket=self.bucket_name,
                            Key=s3_key,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=data
                        )
                        
                        parts.append({
                            'ETag': response['ETag'],
                            'PartNumber': part_number
                        })
                        
                        part_number += 1
                        pbar.update(len(data))
            
            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
        except Exception as e:
            # Abort multipart upload on error
            self.s3_client.abort_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id
            )
            raise e
    
    def upload_data_to_s3(self, max_workers: int = 4) -> Dict[str, Any]:
        """Upload all data files to S3 with parallel processing"""
        
        if not self.new_data_dir.exists():
            return {"error": "Data directory not found. Run move_data_locally() first."}
        
        # Get all files to upload
        all_files = []
        for root, dirs, files in os.walk(self.new_data_dir):
            for file in files:
                local_path = Path(root) / file
                relative_path = local_path.relative_to(self.new_data_dir)
                s3_key = f"{self.s3_data_prefix}/{relative_path}"
                all_files.append((local_path, s3_key))
        
        self.logger.info(f"Uploading {len(all_files)} files to S3...")
        
        # Upload with progress tracking
        uploaded_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all upload tasks
            future_to_file = {
                executor.submit(self.upload_file_to_s3, local_path, s3_key): (local_path, s3_key)
                for local_path, s3_key in all_files
            }
            
            # Process completed uploads
            with tqdm(desc="Uploading files", total=len(all_files)) as pbar:
                for future in as_completed(future_to_file):
                    local_path, s3_key = future_to_file[future]
                    
                    try:
                        success = future.result()
                        if success:
                            uploaded_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        self.logger.error(f"Upload task failed for {local_path}: {e}")
                        failed_count += 1
                    
                    pbar.update(1)
                    
                    # Save progress periodically
                    if (uploaded_count + failed_count) % 100 == 0:
                        self._save_migration_state()
        
        # Final save
        self._save_migration_state()
        
        return {
            "total_files": len(all_files),
            "uploaded_successfully": uploaded_count,
            "failed_uploads": failed_count,
            "success_rate": (uploaded_count / len(all_files)) * 100 if all_files else 0,
            "s3_prefix": self.s3_data_prefix
        }
    
    def create_data_access_config(self) -> str:
        """Create configuration for accessing the migrated data"""
        
        config = {
            "data_migration": {
                "old_path": str(self.source_data_dir),
                "new_local_path": str(self.new_data_dir),
                "s3_bucket": self.bucket_name,
                "s3_prefix": self.s3_data_prefix,
                "migration_date": datetime.now().isoformat()
            },
            "access_methods": {
                "local_access": f"datasets/",  # Relative to quark root
                "s3_streaming": f"s3://{self.bucket_name}/{self.s3_data_prefix}/",
                "hybrid": "Use local for frequent access, S3 for archival"
            },
            "recommendations": [
                "Use local path for development and testing",
                "Stream from S3 for production and large-scale training",
                "Consider S3 Intelligent Tiering for cost optimization",
                "Monitor access patterns and adjust caching strategy"
            ]
        }
        
        config_file = Path("/Users/camdouglas/quark/quark_state_system/data_access_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(config_file)

def main():
    """Main function for data migration"""
    print("📦 Quark Data Migration Manager")
    print("=" * 35)
    
    manager = DataMigrationManager()
    
    # Analyze current data
    print("🔍 Analyzing data directory...")
    analysis = manager.analyze_data_directory()
    
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    print(f"📊 Data Analysis:")
    print(f"   Total Size: {analysis['total_size_gb']:.1f}GB")
    print(f"   Total Files: {analysis['total_files']:,}")
    print(f"   Estimated Upload Time: {analysis['estimated_upload_time']/3600:.1f} hours")
    
    print(f"\n📁 File Types:")
    for ext, count in sorted(analysis['file_types'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {ext or 'no extension'}: {count:,} files")
    
    # Create migration plan
    print("\n📋 Creating migration plan...")
    plan = manager.create_migration_plan()
    
    print(f"💰 Cost Estimate: ${plan['cost_estimate']['total_first_month_usd']:.2f}/month")
    print(f"⏱️ Estimated Duration: {plan['estimated_duration_hours']:.1f} hours")
    
    print("\n💡 Recommendations:")
    for rec in plan['recommendations']:
        print(f"   {rec}")
    
    # Move data locally first
    print(f"\n📁 Moving data to appropriate location...")
    move_result = manager.move_data_locally()
    
    if "error" in move_result:
        print(f"⚠️ Move result: {move_result['error']}")
        if "already exists" in move_result['error']:
            print("   (Directory already moved - continuing with upload)")
    else:
        print(f"✅ Moved: {move_result['moved_from']} → {move_result['moved_to']}")
    
    # Create access configuration
    config_file = manager.create_data_access_config()
    print(f"⚙️ Created access config: {config_file}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review migration plan")
    print(f"   2. Run upload_data_to_s3() to start S3 upload")
    print(f"   3. Update Quark paths to use: datasets/ (local) or S3 streaming")
    
    return manager

if __name__ == "__main__":
    main()
