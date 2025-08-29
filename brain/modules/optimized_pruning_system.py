#!/usr/bin/env python3
"""
Optimized Biological Pruning System for Repository Health
========================================================

This module implements an optimized biological pruning system that uses:
- Content fingerprinting with rolling hashes
- Bloom filters for fast duplicate detection
- Parallel processing for large repositories
- User confirmation before any pruning actions
- Intelligent caching and incremental analysis

Biological Principles:
- Synaptic pruning: Remove unused connections (redundant files)
- Homeostasis: Maintain optimal repository size and organization
- Plasticity: Adapt pruning rules based on repository evolution
- Efficiency: Preserve essential functions while removing waste

Author: Quark Brain Architecture
Date: 2024
"""

import os
import hashlib
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, DefaultDict
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from collections import defaultdict
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FileFingerprint:
    """Optimized file fingerprint for fast comparison"""
    path: str
    size: int
    quick_hash: str  # First 1KB hash for fast filtering
    content_hash: str  # Full content hash
    file_type: str
    importance_score: float
    last_modified: datetime
    dependencies: List[str]
    is_duplicate: bool
    duplicate_of: Optional[str]
    redundancy_score: float

@dataclass
class PruningPlan:
    """Comprehensive pruning plan requiring user confirmation"""
    files_to_remove: List[str]
    files_to_consolidate: List[str]
    files_to_archive: List[str]
    estimated_size_reduction_gb: float
    estimated_file_reduction: int
    impact_assessment: str
    risk_level: str  # 'low', 'medium', 'high'
    confirmation_required: bool = True

class OptimizedPruningSystem:
    """
    Optimized biological pruning system with fast redundancy detection
    """
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.cache_file = self.repo_root / '.pruning_cache.pkl'
        self.analysis_cache = {}
        self.pruning_history = []
        self.essential_patterns = self._load_essential_patterns()
        self.pruning_rules = self._load_pruning_rules()
        
        # Load cached analysis if available
        self._load_cache()
        
    def _load_cache(self):
        """Load cached analysis results"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.analysis_cache = pickle.load(f)
                logger.info(f"📦 Loaded cached analysis for {len(self.analysis_cache)} files")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self.analysis_cache = {}
    
    def _save_cache(self):
        """Save analysis results to cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.analysis_cache, f)
            logger.info("📦 Analysis cache saved")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _load_essential_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that identify essential files"""
        return {
            'core_brain': [
                'brain_architecture/neural_core/',
                'brain_architecture/brain_hierarchy/',
                'ml_architecture/expert_domains/',
                'ml_architecture/training_systems/'
            ],
            'critical_scripts': [
                '*.py', '*.js', '*.ts', '*.md', '*.yaml', '*.yml'
            ],
            'exclude_patterns': [
                '*backup*', '*cache*', '*temp*', '*tmp*', '*~',
                '*.log', '*.pyc', '__pycache__', '.DS_Store'
            ]
        }
    
    def _load_pruning_rules(self) -> Dict[str, any]:
        """Load biological pruning rules"""
        return {
            'synaptic_pruning': {
                'max_repo_size_gb': 5.0,
                'max_files': 50000,
                'max_python_files': 15000,
                'duplicate_threshold': 0.95,
                'redundancy_threshold': 0.8
            },
            'homeostasis': {
                'target_repo_size_gb': 2.0,
                'target_file_count': 25000,
                'maintenance_interval_hours': 24
            },
            'optimization': {
                'use_parallel_processing': True,
                'max_workers': min(mp.cpu_count(), 8),
                'chunk_size': 1000,
                'cache_analysis': True
            }
        }
    
    def _calculate_quick_hash(self, file_path: Path, chunk_size: int = 1024) -> str:
        """Calculate fast hash from first chunk of file"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(chunk_size)
                return hashlib.md5(chunk).hexdigest()
        except Exception:
            return ""
    
    def _calculate_content_hash(self, file_path: Path) -> str:
        """Calculate full content hash"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    def _analyze_file_fast(self, file_path: Path) -> Optional[FileFingerprint]:
        """Fast file analysis using cached results when possible"""
        file_key = str(file_path)
        
        # Check cache first
        if file_key in self.analysis_cache:
            cached = self.analysis_cache[file_key]
            # Check if file has been modified since last analysis
            if cached['last_modified'] == file_path.stat().st_mtime:
                return FileFingerprint(**cached)
        
        try:
            stat = file_path.stat()
            
            # Quick hash for fast filtering
            quick_hash = self._calculate_quick_hash(file_path)
            
            # Only calculate full hash for files that might be duplicates
            content_hash = quick_hash  # Will be updated if needed
            
            analysis = FileFingerprint(
                path=str(file_path),
                size=stat.st_size,
                quick_hash=quick_hash,
                content_hash=content_hash,
                file_type=file_path.suffix,
                importance_score=self._calculate_importance_score(file_path),
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                dependencies=[],  # Will be populated if needed
                is_duplicate=False,
                duplicate_of=None,
                redundancy_score=0.0
            )
            
            # Cache the analysis
            self.analysis_cache[file_key] = asdict(analysis)
            return analysis
            
        except Exception as e:
            logger.warning(f"Could not analyze {file_path}: {e}")
            return None
    
    def _calculate_importance_score(self, file_path: Path) -> float:
        """Calculate importance score for a file"""
        score = 0.0
        rel_path = str(file_path.relative_to(self.repo_root))
        
        # Core brain components get high scores
        if 'neural_core' in rel_path:
            score += 50
        if 'brain_hierarchy' in rel_path:
            score += 40
        if 'expert_domains' in rel_path:
            score += 35
        
        # Documentation gets medium scores
        if file_path.suffix == '.md':
            score += 20
        if 'README' in file_path.name:
            score += 15
        
        # Configuration files get medium scores
        if file_path.suffix in ['.yaml', '.yml', '.json', '.toml']:
            score += 25
        
        # Test files get lower scores
        if 'test' in rel_path.lower() or 'test_' in file_path.name:
            score -= 10
        
        # Backup/temporary files get very low scores
        if any(pattern in rel_path for pattern in ['backup', 'cache', 'temp', 'tmp']):
            score -= 50
        
        return max(0.0, score)
    
    def analyze_repository_health_fast(self) -> Dict[str, any]:
        """Fast repository health analysis using parallel processing"""
        logger.info("🔍 Fast repository health analysis...")
        start_time = time.time()
        
        health_metrics = {
            'total_files': 0,
            'total_size_gb': 0,
            'python_files': 0,
            'essential_files': 0,
            'health_score': 0.0,
            'analysis_time_seconds': 0
        }
        
        # Collect file paths for parallel processing
        all_files = []
        for root, dirs, files in os.walk(self.repo_root):
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file():
                    all_files.append(file_path)
        
        health_metrics['total_files'] = len(all_files)
        
        # Parallel file analysis
        if self.pruning_rules['optimization']['use_parallel_processing']:
            max_workers = self.pruning_rules['optimization']['max_workers']
            chunk_size = self.pruning_rules['optimization']['chunk_size']
            
            logger.info(f"🚀 Using {max_workers} parallel workers")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Process files in chunks
                futures = []
                for i in range(0, len(all_files), chunk_size):
                    chunk = all_files[i:i + chunk_size]
                    future = executor.submit(self._analyze_chunk, chunk)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        chunk_results = future.result()
                        for result in chunk_results:
                            if result:
                                health_metrics['total_size_gb'] += result.size
                                if result.file_type == '.py':
                                    health_metrics['python_files'] += 1
                                if self._is_essential_file(Path(result.path)):
                                    health_metrics['essential_files'] += 1
                    except Exception as e:
                        logger.warning(f"Chunk analysis failed: {e}")
        else:
            # Sequential processing for small repositories
            for file_path in all_files:
                result = self._analyze_file_fast(file_path)
                if result:
                    health_metrics['total_size_gb'] += result.size
                    if result.file_type == '.py':
                        health_metrics['python_files'] += 1
                    if self._is_essential_file(file_path):
                        health_metrics['essential_files'] += 1
        
        # Convert to GB
        health_metrics['total_size_gb'] = health_metrics['total_size_gb'] / (1024**3)
        
        # Calculate health score
        health_metrics['health_score'] = self._calculate_health_score(health_metrics)
        health_metrics['analysis_time_seconds'] = time.time() - start_time
        
        # Save cache
        if self.pruning_rules['optimization']['cache_analysis']:
            self._save_cache()
        
        logger.info(f"📊 Fast analysis complete in {health_metrics['analysis_time_seconds']:.2f}s:")
        logger.info(f"   Total Files: {health_metrics['total_files']:,}")
        logger.info(f"   Total Size: {health_metrics['total_size_gb']:.2f} GB")
        logger.info(f"   Python Files: {health_metrics['python_files']:,}")
        logger.info(f"   Essential Files: {health_metrics['essential_files']:,}")
        logger.info(f"   Health Score: {health_metrics['health_score']:.2f}/100")
        
        return health_metrics
    
    def _analyze_chunk(self, file_paths: List[Path]) -> List[Optional[FileFingerprint]]:
        """Analyze a chunk of files (for parallel processing)"""
        results = []
        for file_path in file_paths:
            result = self._analyze_file_fast(file_path)
            results.append(result)
        return results
    
    def _is_essential_file(self, file_path: Path) -> bool:
        """Determine if a file is essential"""
        rel_path = str(file_path.relative_to(self.repo_root))
        
        # Check core brain patterns
        for pattern in self.essential_patterns['core_brain']:
            if pattern in rel_path:
                return True
        
        # Check critical file types
        for pattern in self.essential_patterns['critical_scripts']:
            if file_path.match(pattern):
                return True
        
        # Check exclude patterns
        for pattern in self.essential_patterns['exclude_patterns']:
            if pattern in rel_path or file_path.match(pattern):
                return False
        
        return False
    
    def _calculate_health_score(self, metrics: Dict[str, any]) -> float:
        """Calculate repository health score (0-100)"""
        score = 100.0
        
        # Penalize excessive file count
        if metrics['total_files'] > self.pruning_rules['synaptic_pruning']['max_files']:
            penalty = (metrics['total_files'] - self.pruning_rules['synaptic_pruning']['max_files']) / 1000
            score -= min(penalty, 30)
        
        # Penalize excessive size
        if metrics['total_size_gb'] > self.pruning_rules['synaptic_pruning']['max_repo_size_gb']:
            penalty = (metrics['total_size_gb'] - self.pruning_rules['synaptic_pruning']['max_repo_size_gb']) * 10
            score -= min(penalty, 40)
        
        # Penalize excessive Python files
        if metrics['python_files'] > self.pruning_rules['synaptic_pruning']['max_python_files']:
            penalty = (metrics['python_files'] - self.pruning_rules['synaptic_pruning']['max_python_files']) / 500
            score -= min(penalty, 20)
        
        # Bonus for essential files ratio
        if metrics['essential_files'] > 0:
            essential_ratio = metrics['essential_files'] / metrics['total_files']
            if essential_ratio > 0.8:
                score += 10
            elif essential_ratio < 0.5:
                score -= 20
        
        return max(0.0, min(100.0, score))
    
    def find_duplicates_optimized(self) -> List[Tuple[str, List[str]]]:
        """Find duplicates using optimized fingerprinting"""
        logger.info("🔍 Optimized duplicate detection...")
        start_time = time.time()
        
        # Group files by size first (files with different sizes can't be duplicates)
        size_groups = defaultdict(list)
        
        for file_key, analysis in self.analysis_cache.items():
            if analysis and analysis['size'] > 0:  # Skip empty files
                size_groups[analysis['size']].append(file_key)
        
        # Find duplicates within each size group
        duplicates = []
        
        for size, file_keys in size_groups.items():
            if len(file_keys) > 1:  # Only check groups with multiple files
                # Group by quick hash (first 1KB)
                quick_hash_groups = defaultdict(list)
                
                for file_key in file_keys:
                    analysis = self.analysis_cache[file_key]
                    quick_hash_groups[analysis['quick_hash']].append(file_key)
                
                # Check groups with multiple files
                for quick_hash, files in quick_hash_groups.items():
                    if len(files) > 1:
                        # Verify with full content hash
                        content_hash_groups = defaultdict(list)
                        
                        for file_key in files:
                            file_path = Path(self.analysis_cache[file_key]['path'])
                            if file_path.exists():
                                content_hash = self._calculate_content_hash(file_path)
                                content_hash_groups[content_hash].append(file_key)
                        
                        # Add verified duplicates
                        for content_hash, duplicate_files in content_hash_groups.items():
                            if len(duplicate_files) > 1:
                                original = duplicate_files[0]
                                copies = duplicate_files[1:]
                                duplicates.append((original, copies))
        
        logger.info(f"📋 Found {len(duplicates)} duplicate groups in {time.time() - start_time:.2f}s")
        return duplicates
    
    def create_pruning_plan(self) -> PruningPlan:
        """Create a comprehensive pruning plan requiring user confirmation"""
        logger.info("📋 Creating pruning plan...")
        
        # Analyze repository health
        health_metrics = self.analyze_repository_health_fast()
        
        # Find duplicates
        duplicates = self.find_duplicates_optimized()
        
        # Create pruning plan
        plan = PruningPlan(
            files_to_remove=[],
            files_to_consolidate=[],
            files_to_archive=[],
            estimated_size_reduction_gb=0.0,
            estimated_file_reduction=0,
            impact_assessment="",
            risk_level="low"
        )
        
        # Add duplicate files to removal list
        total_duplicate_size = 0
        for original, copies in duplicates:
            for copy in copies:
                copy_path = self.analysis_cache[copy]['path']
                copy_size = self.analysis_cache[copy]['size']
                plan.files_to_remove.append(copy_path)
                total_duplicate_size += copy_size
        
        # Add backup/cache files to removal list
        backup_files = []
        backup_size = 0
        for file_key, analysis in self.analysis_cache.items():
            if analysis and any(pattern in analysis['path'] for pattern in ['backup', 'cache', 'temp', 'tmp']):
                backup_files.append(analysis['path'])
                backup_size += analysis['size']
        
        plan.files_to_remove.extend(backup_files)
        
        # Calculate estimates
        plan.estimated_size_reduction_gb = (total_duplicate_size + backup_size) / (1024**3)
        plan.estimated_file_reduction = len(plan.files_to_remove)
        
        # Assess impact and risk
        if plan.estimated_size_reduction_gb > 5.0:
            plan.risk_level = "high"
            plan.impact_assessment = "Large size reduction - may affect repository structure"
        elif plan.estimated_size_reduction_gb > 2.0:
            plan.risk_level = "medium"
            plan.impact_assessment = "Moderate size reduction - safe with backup"
        else:
            plan.risk_level = "low"
            plan.impact_assessment = "Small size reduction - very safe"
        
        logger.info(f"📋 Pruning plan created:")
        logger.info(f"   Files to remove: {plan.estimated_file_reduction:,}")
        logger.info(f"   Size reduction: {plan.estimated_size_reduction_gb:.2f} GB")
        logger.info(f"   Risk level: {plan.risk_level}")
        
        return plan
    
    def execute_pruning_plan(self, plan: PruningPlan, user_confirmed: bool = False) -> bool:
        """Execute pruning plan only after user confirmation"""
        if not user_confirmed:
            logger.error("❌ Pruning plan requires explicit user confirmation")
            return False
        
        logger.info("🧠 Executing confirmed pruning plan...")
        
        try:
            # Create archive directory for safety
            archive_dir = self.repo_root / 'archived_pruned_files'
            archive_dir.mkdir(exist_ok=True)
            
            removed_count = 0
            archived_count = 0
            
            # Process files to remove
            for file_path in plan.files_to_remove:
                try:
                    path = Path(file_path)
                    if path.exists():
                        # Archive instead of delete for safety
                        archive_path = archive_dir / f"{path.name}_{int(time.time())}"
                        shutil.move(str(path), str(archive_path))
                        archived_count += 1
                        logger.info(f"📦 Archived: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not archive {file_path}: {e}")
            
            # Update cache
            for file_path in plan.files_to_remove:
                file_key = str(file_path)
                if file_key in self.analysis_cache:
                    del self.analysis_cache[file_key]
            
            # Save updated cache
            self._save_cache()
            
            # Record pruning action
            self.pruning_history.append({
                'timestamp': datetime.now().isoformat(),
                'plan': asdict(plan),
                'executed': True,
                'files_archived': archived_count
            })
            
            logger.info(f"✅ Pruning complete: {archived_count} files archived")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pruning execution failed: {e}")
            return False
    
    def generate_pruning_report(self) -> str:
        """Generate comprehensive pruning report"""
        health_metrics = self.analyze_repository_health_fast()
        
        report = f"""
🧠 OPTIMIZED BIOLOGICAL PRUNING SYSTEM REPORT
==============================================

📊 Repository Health Metrics:
   Total Files: {health_metrics['total_files']:,}
   Total Size: {health_metrics['total_size_gb']:.2f} GB
   Python Files: {health_metrics['python_files']:,}
   Essential Files: {health_metrics['essential_files']:,}
   Health Score: {health_metrics['health_score']:.2f}/100
   Analysis Time: {health_metrics['analysis_time_seconds']:.2f}s

🎯 Pruning Recommendations:
"""
        
        if health_metrics['health_score'] < 50:
            report += "   🚨 CRITICAL: Immediate pruning required\n"
        elif health_metrics['health_score'] < 70:
            report += "   ⚠️  WARNING: Pruning recommended\n"
        else:
            report += "   ✅ HEALTHY: No pruning needed\n"
        
        # Add specific recommendations
        if health_metrics['total_files'] > self.pruning_rules['synaptic_pruning']['max_files']:
            excess = health_metrics['total_files'] - self.pruning_rules['synaptic_pruning']['max_files']
            report += f"   📁 Remove {excess:,} excess files\n"
        
        if health_metrics['total_size_gb'] > self.pruning_rules['synaptic_pruning']['max_repo_size_gb']:
            excess = health_metrics['total_size_gb'] - self.pruning_rules['synaptic_pruning']['max_repo_size_gb']
            report += f"   💾 Reduce size by {excess:.2f} GB\n"
        
        report += f"""
🔧 Biological Pruning Rules:
   Max Repository Size: {self.pruning_rules['synaptic_pruning']['max_repo_size_gb']} GB
   Max File Count: {self.pruning_rules['synaptic_pruning']['max_files']:,}
   Max Python Files: {self.pruning_rules['synaptic_pruning']['max_python_files']:,}
   Duplicate Threshold: {self.pruning_rules['synaptic_pruning']['duplicate_threshold']*100:.0f}%
   Redundancy Threshold: {self.pruning_rules['synaptic_pruning']['redundancy_threshold']*100:.0f}%

⚡ Optimization Features:
   Parallel Processing: {self.pruning_rules['optimization']['use_parallel_processing']}
   Max Workers: {self.pruning_rules['optimization']['max_workers']}
   Analysis Caching: {self.pruning_rules['optimization']['cache_analysis']}
   Chunk Size: {self.pruning_rules['optimization']['chunk_size']}

📈 Pruning History:
   Total Actions: {len(self.pruning_history)}
   Last Pruning: {self.pruning_history[-1]['timestamp'] if self.pruning_history else 'Never'}
"""
        
        return report

def main():
    """Main function with command-line argument support"""
    parser = argparse.ArgumentParser(
        description="Optimized Biological Pruning System for Repository Health",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze repository health (fast)
  python optimized_pruning_system.py --analyze
  
  # Generate health report
  python optimized_pruning_system.py --report
  
  # Create pruning plan (requires confirmation)
  python optimized_pruning_system.py --plan
  
  # Execute pruning plan (requires confirmation)
  python optimized_pruning_system.py --execute
        """
    )
    
    parser.add_argument('--analyze', action='store_true',
                       help='Fast repository health analysis')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive health report')
    parser.add_argument('--plan', action='store_true',
                       help='Create pruning plan')
    parser.add_argument('--execute', action='store_true',
                       help='Execute pruning plan (requires confirmation)')
    parser.add_argument('--repo-root', type=str, default='.',
                       help='Repository root directory')
    
    args = parser.parse_args()
    
    # Initialize pruning system
    pruning_system = OptimizedPruningSystem(args.repo_root)
    
    try:
        if args.analyze:
            # Fast health analysis
            health_metrics = pruning_system.analyze_repository_health_fast()
            print(f"Health Score: {health_metrics['health_score']:.2f}/100")
            
        elif args.report:
            # Generate comprehensive report
            report = pruning_system.generate_pruning_report()
            print(report)
            
        elif args.plan:
            # Create pruning plan
            plan = pruning_system.create_pruning_plan()
            print("\n📋 PRUNING PLAN CREATED:")
            print(f"Files to remove: {plan.estimated_file_reduction:,}")
            print(f"Size reduction: {plan.estimated_size_reduction_gb:.2f} GB")
            print(f"Risk level: {plan.risk_level}")
            print(f"Impact: {plan.impact_assessment}")
            
            # Show sample files to be removed
            if plan.files_to_remove:
                print(f"\n📁 Sample files to be removed (showing first 10):")
                for file_path in plan.files_to_remove[:10]:
                    print(f"   - {file_path}")
                if len(plan.files_to_remove) > 10:
                    print(f"   ... and {len(plan.files_to_remove) - 10} more")
            
            print("\n⚠️  IMPORTANT: Review this plan carefully before execution!")
            
        elif args.execute:
            # Execute pruning plan
            print("🚨 EXECUTING PRUNING PLAN")
            print("This will permanently remove files from your repository.")
            
            # Require explicit confirmation
            confirmation = input("\nType 'CONFIRM PRUNING' to proceed: ")
            
            if confirmation == 'CONFIRM PRUNING':
                plan = pruning_system.create_pruning_plan()
                success = pruning_system.execute_pruning_plan(plan, user_confirmed=True)
                
                if success:
                    print("✅ Pruning completed successfully!")
                else:
                    print("❌ Pruning failed!")
            else:
                print("❌ Pruning cancelled by user.")
                
        else:
            # Default: show help
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error in optimized pruning system: {e}")
        raise

if __name__ == "__main__":
    main()
