#!/usr/bin/env python3
"""
Biological Pruning System for Repository Health
==============================================

This module implements a biological pruning system that mimics natural neural pruning
processes to automatically maintain repository health. It removes redundant files,
duplicates, and non-essential components while preserving core functionality.

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
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import shutil
import difflib
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    path: str
    size: int
    hash: str
    file_type: str
    importance_score: float
    last_modified: datetime
    dependencies: List[str]
    is_duplicate: bool
    duplicate_of: Optional[str]
    redundancy_score: float

@dataclass
class PruningDecision:
    """Decision made by the pruning system"""
    file_path: str
    action: str  # 'keep', 'remove', 'consolidate', 'archive'
    reason: str
    confidence: float
    impact_assessment: str

class BiologicalPruningSystem:
    """
    Biological pruning system that mimics neural pruning processes
    """
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.essential_patterns = self._load_essential_patterns()
        self.pruning_rules = self._load_pruning_rules()
        self.analysis_cache = {}
        self.pruning_history = []
        
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
                'duplicate_threshold': 0.95,  # 95% similarity = duplicate
                'redundancy_threshold': 0.8
            },
            'homeostasis': {
                'target_repo_size_gb': 2.0,
                'target_file_count': 25000,
                'maintenance_interval_hours': 24
            },
            'plasticity': {
                'adapt_to_usage_patterns': True,
                'learn_from_pruning_history': True,
                'evolve_rules_based_on_impact': True
            }
        }
    
    def analyze_repository_health(self) -> Dict[str, any]:
        """Analyze overall repository health"""
        logger.info("🔍 Analyzing repository health...")
        
        health_metrics = {
            'total_files': 0,
            'total_size_gb': 0,
            'python_files': 0,
            'duplicate_files': 0,
            'redundant_files': 0,
            'essential_files': 0,
            'health_score': 0.0
        }
        
        # Count files and calculate sizes
        for root, dirs, files in os.walk(self.repo_root):
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file():
                    health_metrics['total_files'] += 1
                    
                    try:
                        size = file_path.stat().st_size
                        health_metrics['total_size_gb'] += size
                        
                        if file.endswith('.py'):
                            health_metrics['python_files'] += 1
                            
                        if self._is_essential_file(file_path):
                            health_metrics['essential_files'] += 1
                            
                    except (OSError, PermissionError):
                        continue
        
        # Convert to GB
        health_metrics['total_size_gb'] = health_metrics['total_size_gb'] / (1024**3)
        
        # Calculate health score
        health_metrics['health_score'] = self._calculate_health_score(health_metrics)
        
        logger.info(f"📊 Repository Health Analysis Complete:")
        logger.info(f"   Total Files: {health_metrics['total_files']:,}")
        logger.info(f"   Total Size: {health_metrics['total_size_gb']:.2f} GB")
        logger.info(f"   Python Files: {health_metrics['python_files']:,}")
        logger.info(f"   Essential Files: {health_metrics['essential_files']:,}")
        logger.info(f"   Health Score: {health_metrics['health_score']:.2f}/100")
        
        return health_metrics
    
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
    
    def find_duplicate_files(self) -> List[Tuple[str, List[str]]]:
        """Find duplicate files using content hashing"""
        logger.info("🔍 Searching for duplicate files...")
        
        hash_groups = defaultdict(list)
        duplicates = []
        
        for root, dirs, files in os.walk(self.repo_root):
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file() and file_path.stat().st_size < 1024*1024:  # Skip files > 1MB
                    try:
                        file_hash = self._calculate_file_hash(file_path)
                        hash_groups[file_hash].append(str(file_path))
                    except (OSError, PermissionError):
                        continue
        
        # Find groups with multiple files
        for file_hash, file_list in hash_groups.items():
            if len(file_list) > 1:
                duplicates.append((file_list[0], file_list[1:]))
        
        logger.info(f"📋 Found {len(duplicates)} duplicate file groups")
        return duplicates
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        except (OSError, PermissionError):
            return ""
        return hash_sha256.hexdigest()
    
    def find_redundant_files(self) -> List[FileAnalysis]:
        """Find redundant files based on content similarity"""
        logger.info("🔍 Analyzing file redundancy...")
        
        redundant_files = []
        python_files = []
        
        # Collect Python files for analysis
        for root, dirs, files in os.walk(self.repo_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if file_path.is_file():
                        try:
                            analysis = self._analyze_file(file_path)
                            if analysis:
                                python_files.append(analysis)
                        except (OSError, PermissionError):
                            continue
        
        # Find redundant files based on content similarity
        for i, file1 in enumerate(python_files):
            for file2 in python_files[i+1:]:
                similarity = self._calculate_file_similarity(file1, file2)
                if similarity > self.pruning_rules['synaptic_pruning']['redundancy_threshold']:
                    file1.redundancy_score = similarity
                    file1.is_duplicate = True
                    file1.duplicate_of = file2.path
                    redundant_files.append(file1)
        
        logger.info(f"📋 Found {len(redundant_files)} redundant files")
        return redundant_files
    
    def _analyze_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single file"""
        try:
            stat = file_path.stat()
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            return FileAnalysis(
                path=str(file_path),
                size=stat.st_size,
                hash=self._calculate_file_hash(file_path),
                file_type=file_path.suffix,
                importance_score=self._calculate_importance_score(file_path, content),
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                dependencies=self._extract_dependencies(content),
                is_duplicate=False,
                duplicate_of=None,
                redundancy_score=0.0
            )
        except Exception as e:
            logger.warning(f"Could not analyze {file_path}: {e}")
            return None
    
    def _calculate_importance_score(self, file_path: Path, content: str) -> float:
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
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract import statements and dependencies"""
        dependencies = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith(('import ', 'from ')):
                # Extract module name
                if line.startswith('import '):
                    module = line[7:].split(' as ')[0].split('.')[0]
                else:  # from ... import
                    module = line[5:].split(' import ')[0].split('.')[0]
                dependencies.append(module)
        
        return dependencies
    
    def _calculate_file_similarity(self, file1: FileAnalysis, file2: FileAnalysis) -> float:
        """Calculate similarity between two files"""
        try:
            with open(file1.path, 'r', encoding='utf-8', errors='ignore') as f1:
                content1 = f1.read()
            with open(file2.path, 'r', encoding='utf-8', errors='ignore') as f2:
                content2 = f2.read()
            
            # Use difflib to calculate similarity
            similarity = difflib.SequenceMatcher(None, content1, content2).ratio()
            return similarity
        except Exception:
            return 0.0
    
    def execute_pruning(self, dry_run: bool = True) -> List[PruningDecision]:
        """Execute the biological pruning process"""
        logger.info("🧠 Executing biological pruning process...")
        
        # Analyze repository health
        health_metrics = self.analyze_repository_health()
        
        # If repository is healthy, no pruning needed
        if health_metrics['health_score'] > 80:
            logger.info("✅ Repository is healthy. No pruning needed.")
            return []
        
        pruning_decisions = []
        
        # Find duplicates
        duplicates = self.find_duplicate_files()
        for original, copies in duplicates:
            for copy in copies:
                decision = PruningDecision(
                    file_path=copy,
                    action='remove',
                    reason=f'Duplicate of {original}',
                    confidence=0.95,
                    impact_assessment='Low - removing redundant copy'
                )
                pruning_decisions.append(decision)
        
        # Find redundant files
        redundant_files = self.find_redundant_files()
        for file_analysis in redundant_files:
            if file_analysis.redundancy_score > self.pruning_rules['synaptic_pruning']['redundancy_threshold']:
                decision = PruningDecision(
                    file_path=file_analysis.path,
                    action='consolidate',
                    reason=f'Redundant with {file_analysis.duplicate_of} (similarity: {file_analysis.redundancy_score:.2f})',
                    confidence=file_analysis.redundancy_score,
                    impact_assessment='Medium - consolidating similar functionality'
                )
                pruning_decisions.append(decision)
        
        # Execute pruning decisions
        if not dry_run:
            self._execute_pruning_decisions(pruning_decisions)
        
        # Log pruning results
        logger.info(f"🧠 Pruning complete. {len(pruning_decisions)} actions taken.")
        
        return pruning_decisions
    
    def _execute_pruning_decisions(self, decisions: List[PruningDecision]):
        """Execute the pruning decisions"""
        for decision in decisions:
            try:
                if decision.action == 'remove':
                    os.remove(decision.file_path)
                    logger.info(f"🗑️  Removed: {decision.file_path}")
                elif decision.action == 'consolidate':
                    # Move to archive instead of deleting
                    archive_path = Path(self.repo_root) / 'archived_redundant_files' / Path(decision.file_path).name
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(decision.file_path, archive_path)
                    logger.info(f"📦 Archived: {decision.file_path} -> {archive_path}")
                
                # Record in pruning history
                self.pruning_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'decision': decision.__dict__,
                    'executed': True
                })
                
            except Exception as e:
                logger.error(f"Failed to execute pruning decision for {decision.file_path}: {e}")
    
    def generate_pruning_report(self) -> str:
        """Generate a comprehensive pruning report"""
        health_metrics = self.analyze_repository_health()
        
        report = f"""
🧠 BIOLOGICAL PRUNING SYSTEM REPORT
====================================

📊 Repository Health Metrics:
   Total Files: {health_metrics['total_files']:,}
   Total Size: {health_metrics['total_size_gb']:.2f} GB
   Python Files: {health_metrics['python_files']:,}
   Essential Files: {health_metrics['essential_files']:,}
   Health Score: {health_metrics['health_score']:.2f}/100

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

📈 Pruning History:
   Total Actions: {len(self.pruning_history)}
   Last Pruning: {self.pruning_history[-1]['timestamp'] if self.pruning_history else 'Never'}
"""
        
        return report

def main():
    """Main function with command-line argument support"""
    parser = argparse.ArgumentParser(
        description="Biological Pruning System for Repository Health",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze repository health
  python biological_pruning_system.py --analyze
  
  # Generate health report
  python biological_pruning_system.py --report
  
  # Execute pruning (dry run)
  python biological_pruning_system.py --prune
  
  # Execute pruning (live)
  python biological_pruning_system.py --prune --live
  
  # Auto-prune when called from Git hooks
  python biological_pruning_system.py --auto-prune
        """
    )
    
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze repository health')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive health report')
    parser.add_argument('--prune', action='store_true',
                       help='Execute pruning process')
    parser.add_argument('--live', action='store_true',
                       help='Execute pruning live (not dry run)')
    parser.add_argument('--auto-prune', action='store_true',
                       help='Automatic pruning mode for Git hooks')
    parser.add_argument('--log-file', type=str,
                       help='Log file path for output')
    parser.add_argument('--repo-root', type=str, default='.',
                       help='Repository root directory')
    
    args = parser.parse_args()
    
    # Set up logging if log file specified
    if args.log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(args.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    # Initialize pruning system
    pruning_system = BiologicalPruningSystem(args.repo_root)
    
    try:
        if args.analyze:
            # Analyze repository health
            health_metrics = pruning_system.analyze_repository_health()
            print(f"Health Score: {health_metrics['health_score']:.2f}/100")
            
        elif args.report:
            # Generate comprehensive report
            report = pruning_system.generate_pruning_report()
            print(report)
            
        elif args.prune:
            # Execute pruning
            dry_run = not args.live
            decisions = pruning_system.execute_pruning(dry_run=dry_run)
            print(f"Pruning complete. {len(decisions)} actions taken.")
            
        elif args.auto_prune:
            # Automatic pruning mode for Git hooks
            logger.info("🧠 Auto-pruning mode activated")
            
            # Check if pruning is needed
            health_metrics = pruning_system.analyze_repository_health()
            
            if health_metrics['health_score'] < 70:
                logger.info("🚨 Pruning required. Executing...")
                decisions = pruning_system.execute_pruning(dry_run=False)
                logger.info(f"✅ Auto-pruning complete. {len(decisions)} actions taken.")
                
                # Re-check health
                new_health = pruning_system.analyze_repository_health()
                logger.info(f"📊 New health score: {new_health['health_score']:.2f}/100")
                
                if new_health['health_score'] < 50:
                    logger.error("❌ Repository still unhealthy after pruning")
                    sys.exit(1)
                else:
                    logger.info("✅ Repository health restored")
            else:
                logger.info("✅ Repository is healthy. No pruning needed.")
                
        else:
            # Default: show help
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error in biological pruning system: {e}")
        if args.auto_prune:
            sys.exit(1)
        else:
            raise

if __name__ == "__main__":
    main()
