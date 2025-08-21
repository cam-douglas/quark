#!/usr/bin/env python3
"""
Integrated Biological Pruning System
===================================

Combines fast duplicate detection, comprehensive redundancy analysis,
and user validation before pruning. Always requires confirmation
before removing any files. Now enhanced with Wolfram Alpha mathematical optimization
and real-time progress rate logging.

Author: Quark Brain Architecture
Date: 2024
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import shutil

# Import our specialized modules
from fast_duplicate_detector import find_duplicates_optimized, find_backup_cache_files
from redundancy_detector import RedundancyDetector

# Import Wolfram mathematical optimization
from mathematical_brain_core import MathematicalBrainCore

@dataclass
class PruningCandidate:
    """A file candidate for pruning"""
    file_path: str
    file_size: int
    reason: str
    confidence: float
    risk_level: str  # 'low', 'medium', 'high'
    category: str  # 'duplicate', 'redundant', 'backup', 'cache'
    alternative_action: str  # 'remove', 'consolidate', 'archive'

@dataclass
class PruningPlan:
    """Complete pruning plan requiring user validation"""
    candidates: List[PruningCandidate]
    total_files: int
    total_size_gb: float
    estimated_reduction_gb: float
    risk_assessment: str
    created_at: datetime
    requires_confirmation: bool = True

@dataclass
class ProgressMetrics:
    """Real-time progress tracking metrics"""
    files_processed: int
    files_removed: int
    files_failed: int
    bytes_processed: int
    bytes_freed: int
    start_time: datetime
    current_time: datetime
    processing_rate: float  # files per second
    speed_mb_per_sec: float  # MB per second
    completion_percentage: float
    estimated_time_remaining: timedelta
    current_file: str
    errors: List[str]

class IntegratedPruningSystem:
    """Integrated pruning system with user validation and Wolfram optimization"""

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.log_file = self.repo_root / 'logs' / 'pruning_operations.log'
        self.progress_log = self.repo_root / 'logs' / 'pruning_progress.log'
        self.log_file.parent.mkdir(exist_ok=True)
        # Initialize mathematical brain core for Wolfram optimization
        self.math_core = MathematicalBrainCore()
        self.progress_metrics = None

    def log_operation(self, message: str):
        """Log pruning operations"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}\n"

        with open(self.log_file, 'a') as f:
            f.write(log_entry)

        print(message)

    def log_progress(self, metrics: ProgressMetrics):
        """Log detailed progress metrics"""
        timestamp = datetime.now().isoformat()
        
        # Calculate additional metrics
        elapsed_time = metrics.current_time - metrics.start_time
        success_rate = (metrics.files_removed / max(metrics.files_processed, 1)) * 100
        
        progress_entry = f"""[{timestamp}] PROGRESS UPDATE:
  Files: {metrics.files_processed:,}/{metrics.total_files:,} ({metrics.completion_percentage:.1f}%)
  Removed: {metrics.files_removed:,} | Failed: {metrics.files_failed:,}
  Success Rate: {success_rate:.1f}%
  Processing Speed: {metrics.processing_rate:.1f} files/sec
  Data Speed: {metrics.speed_mb_per_sec:.1f} MB/sec
  Bytes Freed: {metrics.bytes_freed / (1024**2):.1f} MB
  Elapsed Time: {str(elapsed_time).split('.')[0]}
  ETA: {str(metrics.estimated_time_remaining).split('.')[0]}
  Current File: {metrics.current_file}
  Errors: {len(metrics.errors)}
"""

        # Write to progress log
        with open(self.progress_log, 'a') as f:
            f.write(progress_entry)

        # Display progress bar
        self._display_progress_bar(metrics)

    def _display_progress_bar(self, metrics: ProgressMetrics):
        """Display a visual progress bar with real-time metrics"""
        bar_length = 50
        filled_length = int(bar_length * metrics.completion_percentage / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r🔄 Progress: [{bar}] {metrics.completion_percentage:.1f}% | "
              f"Speed: {metrics.processing_rate:.1f} files/sec | "
              f"ETA: {str(metrics.estimated_time_remaining).split('.')[0]} | "
              f"Freed: {metrics.bytes_freed / (1024**2):.1f} MB", end='', flush=True)

    def _calculate_progress_metrics(self, current_processed: int, current_removed: int, 
                                  current_failed: int, current_bytes_freed: int,
                                  total_files: int, start_time: datetime,
                                  current_file: str, errors: List[str]) -> ProgressMetrics:
        """Calculate real-time progress metrics"""
        current_time = datetime.now()
        elapsed_time = current_time - start_time
        
        # Calculate rates
        if elapsed_time.total_seconds() > 0:
            processing_rate = current_processed / elapsed_time.total_seconds()
            speed_mb_per_sec = (current_bytes_freed / (1024**2)) / elapsed_time.total_seconds()
        else:
            processing_rate = 0
            speed_mb_per_sec = 0
        
        # Calculate completion percentage
        completion_percentage = (current_processed / total_files) * 100 if total_files > 0 else 0
        
        # Estimate time remaining
        if processing_rate > 0:
            remaining_files = total_files - current_processed
            estimated_seconds = remaining_files / processing_rate
            estimated_time_remaining = timedelta(seconds=int(estimated_seconds))
        else:
            estimated_time_remaining = timedelta(seconds=0)
        
        return ProgressMetrics(
            files_processed=current_processed,
            files_removed=current_removed,
            files_failed=current_failed,
            bytes_processed=current_processed,  # Simplified for now
            bytes_freed=current_bytes_freed,
            start_time=start_time,
            current_time=current_time,
            processing_rate=processing_rate,
            speed_mb_per_sec=speed_mb_per_sec,
            completion_percentage=completion_percentage,
            estimated_time_remaining=estimated_time_remaining,
            current_file=current_file,
            errors=errors
        )

    def analyze_repository(self) -> Dict[str, any]:
        """Comprehensive repository analysis with Wolfram optimization"""
        self.log_operation("🔍 Starting comprehensive repository analysis with Wolfram optimization...")
        start_time = time.time()

        # Get basic metrics
        total_files = 0
        total_size = 0
        python_files = 0

        for root, dirs, files in os.walk(self.repo_root):
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file():
                    total_files += 1
                    try:
                        size = file_path.stat().st_size
                        total_size += size
                        if file.endswith('.py'):
                            python_files += 1
                    except Exception:
                        continue

        total_size_gb = total_size / (1024**3)

        # Find duplicates
        self.log_operation("🔍 Detecting duplicate files...")
        duplicates = find_duplicates_optimized(str(self.repo_root))

        # Find backup/cache files
        self.log_operation("🔍 Finding backup and cache files...")
        backup_files = find_backup_cache_files(str(self.repo_root))

        # Use Wolfram Alpha to optimize redundancy detection parameters
        self.log_operation("🧮 Consulting Wolfram Alpha for mathematical optimization...")
        codebase_metrics = {
            'total_files': total_files,
            'python_files': python_files,
            'total_size_gb': total_size_gb
        }
        
        # Get mathematically optimized parameters
        optimization_params = self.math_core.optimize_redundancy_detection(codebase_metrics)
        self.log_operation(f"📊 Wolfram optimization results: {optimization_params}")

        # Find redundancies with optimized parameters
        self.log_operation("🔍 Analyzing functional redundancies with Wolfram-optimized parameters...")
        detector = RedundancyDetector(str(self.repo_root))
        # Use Wolfram-optimized similarity threshold
        detector.similarity_threshold = optimization_params.get('optimal_similarity_threshold', 0.75)
        redundancies = detector.find_all_redundancies()

        elapsed_time = time.time() - start_time

        analysis = {
            'total_files': total_files,
            'total_size_gb': total_size_gb,
            'python_files': python_files,
            'duplicates': duplicates,
            'backup_files': backup_files,
            'redundancies': redundancies,
            'wolfram_optimization': optimization_params,
            'analysis_time_seconds': elapsed_time
        }

        self.log_operation(f"✅ Analysis complete in {elapsed_time:.2f}s")
        self.log_operation(f"📊 Found {len(duplicates)} duplicate groups, {len(backup_files)} backup files")
        self.log_operation(f"🧠 Wolfram optimization applied with {optimization_params.get('confidence_level', 0.7)*100:.1f}% confidence")

        return analysis

    def create_pruning_plan(self, analysis: Dict[str, any]) -> PruningPlan:
        """Create comprehensive pruning plan with Wolfram optimization"""
        self.log_operation("📋 Creating pruning plan with Wolfram-optimized parameters...")

        candidates = []
        total_size_reduction = 0

        # Process duplicates
        for original, copies in analysis['duplicates']:
            for copy in copies:
                try:
                    size = Path(copy).stat().st_size
                    candidates.append(PruningCandidate(
                        file_path=copy,
                        file_size=size,
                        reason=f"Duplicate of {original}",
                        confidence=0.95,
                        risk_level="low",
                        category="duplicate",
                        alternative_action="remove"
                    ))
                    total_size_reduction += size
                except Exception:
                    continue

        # Process backup/cache files
        for backup_file in analysis['backup_files']:
            try:
                size = Path(backup_file).stat().st_size
                candidates.append(PruningCandidate(
                    file_path=backup_file,
                    file_size=size,
                    reason="Backup/cache file",
                    confidence=0.90,
                    risk_level="low",
                    category="backup",
                    alternative_action="remove"
                ))
                total_size_reduction += size
            except Exception:
                continue

        # Process redundancies with Wolfram-optimized confidence
        wolfram_threshold = analysis['wolfram_optimization'].get('optimal_similarity_threshold', 0.75)
        for redundancy_type, type_redundancies in analysis['redundancies'].items():
            for file1, file2, similarity, reason in type_redundancies:
                # Add the second file as candidate for consolidation
                try:
                    size = Path(file2).stat().st_size
                    # Use Wolfram-optimized risk assessment
                    risk_level = "low" if similarity > wolfram_threshold else "medium"
                    candidates.append(PruningCandidate(
                        file_path=file2,
                        file_size=size,
                        reason=f"Wolfram-optimized redundant: {reason}",
                        confidence=similarity,
                        risk_level=risk_level,
                        category="redundant",
                        alternative_action="consolidate"
                    ))
                    total_size_reduction += size
                except Exception:
                    continue

        # Calculate total reduction
        total_reduction_gb = total_size_reduction / (1024**3)

        # Assess overall risk using Wolfram insights
        high_risk_count = len([c for c in candidates if c.risk_level == "high"])
        wolfram_confidence = analysis['wolfram_optimization'].get('confidence_level', 0.7)
        
        if high_risk_count > len(candidates) * 0.3:
            risk_assessment = f"HIGH - Many high-risk candidates, review carefully (Wolfram confidence: {wolfram_confidence*100:.1f}%)"
        elif high_risk_count > len(candidates) * 0.1:
            risk_assessment = f"MEDIUM - Some high-risk candidates, review recommended (Wolfram confidence: {wolfram_confidence*100:.1f}%)"
        else:
            risk_assessment = f"LOW - Mostly safe candidates (Wolfram confidence: {wolfram_confidence*100:.1f}%)"

        plan = PruningPlan(
            candidates=candidates,
            total_files=len(candidates),
            total_size_gb=total_reduction_gb,
            estimated_reduction_gb=total_reduction_gb,
            risk_assessment=risk_assessment,
            created_at=datetime.now()
        )

        self.log_operation(f"📋 Pruning plan created with Wolfram optimization:")
        self.log_operation(f"   Candidates: {len(candidates)}")
        self.log_operation(f"   Size reduction: {total_reduction_gb:.2f} GB")
        self.log_operation(f"   Risk level: {risk_assessment}")
        self.log_operation(f"   Wolfram confidence: {wolfram_confidence*100:.1f}%")

        return plan

    def display_pruning_summary(self, plan: PruningPlan):
        """Display concise pruning summary for user decision"""
        print("\n" + "="*80)
        print("🧠 WOLFRAM-OPTIMIZED PRUNING SUMMARY")
        print("="*80)

        print(f"📊 OVERVIEW:")
        print(f"   Total files to prune: {plan.total_files:,}")
        print(f"   Estimated space savings: {plan.estimated_reduction_gb:.2f} GB")
        print(f"   Risk assessment: {plan.risk_assessment}")

        # Category breakdown
        categories = {}
        for candidate in plan.candidates:
            categories[candidate.category] = categories.get(candidate.category, 0) + 1

        print(f"\n🎯 BREAKDOWN BY CATEGORY:")
        for category, count in categories.items():
            category_size = sum(c.file_size for c in plan.candidates if c.category == category) / (1024**3)
            print(f"   {category.capitalize()}: {count:,} files ({category_size:.2f} GB)")

        # Sample files from each category
        print(f"\n📁 SAMPLE FILES TO BE PRUNED:")
        for category in categories.keys():
            category_files = [c for c in plan.candidates if c.category == category][:3]
            print(f"\n   {category.upper()}:")
            for candidate in category_files:
                print(f"     • {candidate.file_path}")
                print(f"       Size: {candidate.file_size / 1024:.1f} KB | Risk: {candidate.risk_level}")

        print("\n" + "="*80)

    def get_single_confirmation(self, plan: PruningPlan) -> bool:
        """Get single confirmation for entire pruning plan"""
        print("\n🔐 SINGLE CONFIRMATION REQUIRED")
        print("="*50)

        print(f"💾 This will free up {plan.estimated_reduction_gb:.2f} GB of space")
        print(f"📁 Affecting {plan.total_files:,} files across all categories")
        print(f"🧠 Using Wolfram Alpha mathematical optimization")
        print(f"⚠️  Risk level: {plan.risk_assessment}")

        response = input("\nType 'CONFIRM PRUNING' to proceed with all categories: ").strip()
        return response == 'CONFIRM PRUNING'

    def execute_pruning(self, plan: PruningPlan) -> bool:
        """Execute pruning by removing files (not archiving) with progress tracking"""
        self.log_operation("🧠 Executing pruning plan with real-time progress tracking...")
        
        # Initialize progress tracking
        start_time = time.time()
        processed_count = 0
        removed_count = 0
        failed_count = 0
        bytes_freed = 0
        errors = []
        
        # Create progress log
        progress_log = self.repo_root / 'logs' / 'pruning_progress.log'
        with open(progress_log, 'w') as f:
            f.write(f"🧠 QUARK PRUNING PROGRESS LOG\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Total files: {plan.total_files:,}\n")
            f.write(f"Estimated reduction: {plan.estimated_reduction_gb:.2f} GB\n")
            f.write("="*80 + "\n\n")

        print(f"\n📊 Starting pruning of {plan.total_files:,} files...")
        print("🔄 Progress tracking enabled - check logs/pruning_progress.log for detailed metrics\n")

        for candidate in plan.candidates:
            try:
                file_path = Path(candidate.file_path)
                if file_path.exists():
                    # Remove the file directly
                    file_path.unlink()
                    removed_count += 1
                    bytes_freed += candidate.file_size

                    self.log_operation(f"🗑️  Removed: {candidate.file_path}")

            except Exception as e:
                error_msg = f"Could not remove {candidate.file_path}: {e}"
                errors.append(error_msg)
                failed_count += 1
                self.log_operation(f"❌ Error: {error_msg}")

            processed_count += 1

            # Calculate and log progress metrics every 10 files
            if processed_count % 10 == 0:
                elapsed_time = time.time() - start_time
                processing_rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                completion_percentage = (processed_count / plan.total_files) * 100
                speed_mb_per_sec = (bytes_freed / (1024**2)) / elapsed_time if elapsed_time > 0 else 0
                
                # Estimate time remaining
                if processing_rate > 0:
                    remaining_files = plan.total_files - processed_count
                    eta_seconds = remaining_files / processing_rate
                    eta_str = f"{eta_seconds:.0f}s"
                else:
                    eta_str = "calculating..."
                
                # Display progress bar
                bar_length = 40
                filled_length = int(bar_length * completion_percentage / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"\r🔄 [{bar}] {completion_percentage:.1f}% | "
                      f"Speed: {processing_rate:.1f} files/sec | "
                      f"ETA: {eta_str} | "
                      f"Freed: {bytes_freed / (1024**2):.1f} MB", end='', flush=True)
                
                # Log detailed progress
                progress_entry = f"[{datetime.now().isoformat()}] Progress: {processed_count:,}/{plan.total_files:,} ({completion_percentage:.1f}%) | "
                progress_entry += f"Rate: {processing_rate:.1f} files/sec | Speed: {speed_mb_per_sec:.1f} MB/sec | "
                progress_entry += f"Freed: {bytes_freed / (1024**2):.1f} MB | ETA: {eta_str}\n"
                
                with open(progress_log, 'a') as f:
                    f.write(progress_entry)

        # Final progress update
        print()  # New line after progress bar
        
        # Final summary
        total_time = time.time() - start_time
        final_speed = processed_count / total_time if total_time > 0 else 0
        success_rate = (removed_count / max(processed_count, 1)) * 100
        
        self.log_operation(f"✅ Pruning execution complete:")
        self.log_operation(f"   Files processed: {processed_count:,}")
        self.log_operation(f"   Files removed: {removed_count:,}")
        self.log_operation(f"   Files failed: {failed_count:,}")
        self.log_operation(f"   Errors: {len(errors)}")
        self.log_operation(f"   Space freed: {bytes_freed / (1024**3):.2f} GB")
        self.log_operation(f"   Success rate: {success_rate:.1f}%")
        self.log_operation(f"   Average speed: {final_speed:.1f} files/sec")
        self.log_operation(f"   Total time: {total_time:.1f} seconds")
        
        # Log final summary
        with open(progress_log, 'a') as f:
            f.write(f"\n🎉 PRUNING COMPLETED\n")
            f.write(f"Completion time: {datetime.now().isoformat()}\n")
            f.write(f"Total duration: {total_time:.1f} seconds\n")
            f.write(f"Final speed: {final_speed:.1f} files/sec\n")
            f.write(f"Success rate: {success_rate:.1f}%\n")

        if errors:
            self.log_operation("⚠️  Some files could not be removed. Check logs for details.")

        return True

    def run_complete_pruning_workflow(self):
        """Run the complete pruning workflow with single confirmation"""
        self.log_operation("🧠 Starting Wolfram-optimized pruning workflow...")

        # Step 1: Analyze repository with Wolfram optimization
        analysis = self.analyze_repository()

        # Step 2: Create pruning plan with Wolfram parameters
        plan = self.create_pruning_plan(analysis)

        if not plan.candidates:
            self.log_operation("✅ No pruning candidates found. Repository is healthy!")
            return

        # Step 3: Display single summary for user review
        self.display_pruning_summary(plan)

        # Step 4: Get single confirmation
        if self.get_single_confirmation(plan):
            success = self.execute_pruning(plan)
            if success:
                self.log_operation("🎉 Pruning workflow completed successfully!")
                self.log_operation(f"📊 Check logs/pruning_progress.log for detailed progress metrics")
            else:
                self.log_operation("❌ Pruning workflow failed!")
        else:
            self.log_operation("❌ Pruning cancelled by user")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Wolfram-Optimized Integrated Biological Pruning System with Progress Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow with Wolfram optimization and progress tracking
  python integrated_pruning_system.py --workflow

  # Analyze repository only
  python integrated_pruning_system.py --analyze

  # Create pruning plan only
  python integrated_pruning_system.py --plan
        """
    )

    parser.add_argument('--workflow', action='store_true',
                       help='Run complete pruning workflow with Wolfram optimization and progress tracking')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze repository only')
    parser.add_argument('--plan', action='store_true',
                       help='Create pruning plan only')
    parser.add_argument('--repo-root', type=str, default='.',
                       help='Repository root directory')

    args = parser.parse_args()

    # Initialize system
    pruning_system = IntegratedPruningSystem(args.repo_root)

    try:
        if args.workflow:
            # Run complete workflow
            pruning_system.run_complete_pruning_workflow()

        elif args.analyze:
            # Analyze only
            analysis = pruning_system.analyze_repository()
            print(f"\n📊 Analysis Results:")
            print(f"   Total files: {analysis['total_files']:,}")
            print(f"   Total size: {analysis['total_size_gb']:.2f} GB")
            print(f"   Duplicates: {len(analysis['duplicates'])} groups")
            print(f"   Backup files: {len(analysis['backup_files'])}")
            print(f"   Wolfram optimization: {analysis['wolfram_optimization']}")

        elif args.plan:
            # Create plan only
            analysis = pruning_system.analyze_repository()
            plan = pruning_system.create_pruning_plan(analysis)
            pruning_system.display_pruning_summary(plan)

        else:
            # Default: show help
            parser.print_help()

    except Exception as e:
        print(f"❌ Error in integrated pruning system: {e}")
        raise

if __name__ == "__main__":
    main()
