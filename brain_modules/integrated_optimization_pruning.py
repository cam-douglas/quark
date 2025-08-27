#!/usr/bin/env python3
"""
Integrated Optimization and Pruning System
==========================================

Combines the biological pruning system with Cursor/system optimization.
Automatically runs optimization after every pruning operation for peak performance.

Author: Quark Brain Architecture
Date: 2024
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import existing pruning system
from integrated_pruning_system import IntegratedPruningSystem, PruningPlan
from optimized_pruning_system import OptimizedPruningSystem

# Import system optimizer
sys.path.append(str(Path(__file__).parent.parent / "optimization"))
from system_optimizer import SystemOptimizer

class IntegratedOptimizationPruning:
    """Integrated system that combines pruning with automatic optimization"""
    
    def __init__(self, repo_root: str, 
                 fast_mode: bool = False,
                 include_dirs: List[str] = None,
                 exclude_dirs: List[str] = None,
                 auto_optimize: bool = True):
        
        self.repo_root = Path(repo_root)
        self.auto_optimize = auto_optimize
        
        # Initialize pruning systems
        self.integrated_pruner = IntegratedPruningSystem(
            repo_root=repo_root,
            fast_mode=fast_mode,
            include_dirs=include_dirs,
            exclude_dirs=exclude_dirs
        )
        
        self.optimized_pruner = OptimizedPruningSystem(repo_root)
        
        # Initialize system optimizer
        self.system_optimizer = SystemOptimizer()
        
        # Setup logs
        self.log_file = self.repo_root / 'logs' / 'integrated_optimization_pruning.log'
        self.log_file.parent.mkdir(exist_ok=True)
        
        self.optimization_history = []
    
    def log_operation(self, message: str):
        """Log operations with timestamp"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        print(message)
    
    def run_pre_pruning_optimization(self) -> Dict[str, Any]:
        """Run optimization before pruning for better performance"""
        self.log_operation("🚀 Running pre-pruning optimization...")
        
        pre_optimization = {
            "timestamp": datetime.now().isoformat(),
            "type": "pre_pruning",
            "optimizations_applied": [],
            "performance_score": 0,
            "duration_seconds": 0
        }
        
        start_time = time.time()
        
        try:
            # Run system optimization
            report = self.system_optimizer.run_full_optimization()
            
            pre_optimization["optimizations_applied"] = self.system_optimizer.optimizations_applied
            pre_optimization["performance_score"] = report["performance_score"]
            pre_optimization["system_resources"] = report["system_resources"]
            
            self.log_operation(f"✅ Pre-pruning optimization complete:")
            self.log_operation(f"   Performance Score: {report['performance_score']}/100")
            self.log_operation(f"   Optimizations: {len(self.system_optimizer.optimizations_applied)}")
            
        except Exception as e:
            self.log_operation(f"⚠️ Pre-pruning optimization error: {e}")
            pre_optimization["error"] = str(e)
        
        pre_optimization["duration_seconds"] = time.time() - start_time
        self.optimization_history.append(pre_optimization)
        
        return pre_optimization
    
    def run_post_pruning_optimization(self) -> Dict[str, Any]:
        """Run optimization after pruning to clean up and optimize"""
        self.log_operation("🧹 Running post-pruning optimization...")
        
        post_optimization = {
            "timestamp": datetime.now().isoformat(),
            "type": "post_pruning",
            "optimizations_applied": [],
            "performance_score": 0,
            "duration_seconds": 0
        }
        
        start_time = time.time()
        
        try:
            # Clean up Cursor cache after pruning
            self.log_operation("🧹 Cleaning Cursor cache...")
            cache_cleaned = self.system_optimizer.optimize_cursor_cache()
            
            if cache_cleaned:
                post_optimization["optimizations_applied"].append("Cursor cache cleaned")
            
            # Update file descriptor limits after pruning
            limits_updated = self.system_optimizer.optimize_file_limits()
            if limits_updated:
                post_optimization["optimizations_applied"].append("File descriptor limits updated")
            
            # Get updated system metrics
            report = self.system_optimizer.generate_optimization_report()
            post_optimization["performance_score"] = report["performance_score"]
            post_optimization["system_resources"] = report["system_resources"]
            
            # Update .cursorignore if new directories were created
            self._update_cursorignore()
            post_optimization["optimizations_applied"].append("Updated .cursorignore")
            
            self.log_operation(f"✅ Post-pruning optimization complete:")
            self.log_operation(f"   Performance Score: {report['performance_score']}/100")
            self.log_operation(f"   Optimizations: {len(post_optimization['optimizations_applied'])}")
            
        except Exception as e:
            self.log_operation(f"⚠️ Post-pruning optimization error: {e}")
            post_optimization["error"] = str(e)
        
        post_optimization["duration_seconds"] = time.time() - start_time
        self.optimization_history.append(post_optimization)
        
        return post_optimization
    
    def _update_cursorignore(self):
        """Update .cursorignore with new patterns after pruning"""
        cursorignore_path = self.repo_root / ".cursorignore"
        
        # Additional patterns to ignore after pruning
        new_patterns = [
            "# Post-pruning optimization patterns",
            "archived_pruned_files/",
            ".pruning_cache.pkl",
            "logs/pruning_*.log",
            "logs/integrated_optimization_pruning.log"
        ]
        
        try:
            if cursorignore_path.exists():
                with open(cursorignore_path, 'r') as f:
                    content = f.read()
                
                # Only add patterns that don't already exist
                for pattern in new_patterns:
                    if pattern not in content and not pattern.startswith("#"):
                        content += f"\n{pattern}"
                
                with open(cursorignore_path, 'w') as f:
                    f.write(content)
                    
                self.log_operation("📝 Updated .cursorignore with post-pruning patterns")
            
        except Exception as e:
            self.log_operation(f"⚠️ Could not update .cursorignore: {e}")
    
    def run_integrated_pruning_workflow_with_optimization(self) -> Dict[str, Any]:
        """Run complete integrated workflow: optimize → prune → optimize"""
        
        workflow_results = {
            "start_time": datetime.now().isoformat(),
            "pre_optimization": {},
            "pruning_results": {},
            "post_optimization": {},
            "total_duration_seconds": 0,
            "success": False
        }
        
        start_time = time.time()
        
        self.log_operation("🧠⚡ Starting Integrated Optimization + Pruning Workflow")
        self.log_operation("=" * 65)
        
        try:
            # Step 1: Pre-pruning optimization
            if self.auto_optimize:
                workflow_results["pre_optimization"] = self.run_pre_pruning_optimization()
            
            # Step 2: Run pruning workflow
            self.log_operation("🧠 Running biological pruning workflow...")
            
            # Use the integrated pruning system for comprehensive analysis
            analysis = self.integrated_pruner.analyze_repository()
            plan = self.integrated_pruner.create_pruning_plan(analysis)
            
            if not plan.candidates:
                self.log_operation("✅ No pruning candidates found. Repository is healthy!")
                workflow_results["pruning_results"] = {"status": "no_pruning_needed"}
            else:
                # Display pruning summary
                self.integrated_pruner.display_pruning_summary(plan)
                
                # Get user confirmation
                if self.integrated_pruner.get_single_confirmation(plan):
                    pruning_success = self.integrated_pruner.execute_pruning(plan)
                    workflow_results["pruning_results"] = {
                        "status": "completed" if pruning_success else "failed",
                        "files_processed": plan.total_files,
                        "size_reduction_gb": plan.estimated_reduction_gb
                    }
                else:
                    workflow_results["pruning_results"] = {"status": "cancelled_by_user"}
            
            # Step 3: Post-pruning optimization
            if self.auto_optimize and workflow_results["pruning_results"].get("status") == "completed":
                workflow_results["post_optimization"] = self.run_post_pruning_optimization()
            
            workflow_results["success"] = True
            
        except Exception as e:
            self.log_operation(f"❌ Integrated workflow error: {e}")
            workflow_results["error"] = str(e)
            workflow_results["success"] = False
        
        workflow_results["total_duration_seconds"] = time.time() - start_time
        
        # Generate final summary
        self._generate_workflow_summary(workflow_results)
        
        return workflow_results
    
    def _generate_workflow_summary(self, results: Dict[str, Any]):
        """Generate comprehensive workflow summary"""
        
        self.log_operation("\n" + "=" * 65)
        self.log_operation("🎉 INTEGRATED OPTIMIZATION + PRUNING SUMMARY")
        self.log_operation("=" * 65)
        
        # Pre-optimization summary
        if "pre_optimization" in results and results["pre_optimization"]:
            pre = results["pre_optimization"]
            self.log_operation(f"⚡ Pre-Optimization:")
            self.log_operation(f"   Performance Score: {pre.get('performance_score', 'N/A')}/100")
            self.log_operation(f"   Optimizations Applied: {len(pre.get('optimizations_applied', []))}")
            self.log_operation(f"   Duration: {pre.get('duration_seconds', 0):.1f}s")
        
        # Pruning summary
        if "pruning_results" in results:
            pruning = results["pruning_results"]
            self.log_operation(f"🧠 Biological Pruning:")
            self.log_operation(f"   Status: {pruning.get('status', 'unknown')}")
            if pruning.get("files_processed"):
                self.log_operation(f"   Files Processed: {pruning['files_processed']:,}")
                self.log_operation(f"   Size Reduction: {pruning.get('size_reduction_gb', 0):.2f} GB")
        
        # Post-optimization summary
        if "post_optimization" in results and results["post_optimization"]:
            post = results["post_optimization"]
            self.log_operation(f"🧹 Post-Optimization:")
            self.log_operation(f"   Performance Score: {post.get('performance_score', 'N/A')}/100")
            self.log_operation(f"   Optimizations Applied: {len(post.get('optimizations_applied', []))}")
            self.log_operation(f"   Duration: {post.get('duration_seconds', 0):.1f}s")
        
        # Overall summary
        self.log_operation(f"\n📊 Overall Results:")
        self.log_operation(f"   Total Duration: {results.get('total_duration_seconds', 0):.1f}s")
        self.log_operation(f"   Success: {'✅ Yes' if results.get('success') else '❌ No'}")
        self.log_operation(f"   Optimization History: {len(self.optimization_history)} entries")
        
        # Performance recommendations
        self.log_operation(f"\n💡 Performance Recommendations:")
        
        if results.get("success"):
            self.log_operation(f"   ✅ Restart Cursor for optimal performance")
            self.log_operation(f"   ✅ Repository is now optimized and pruned")
            self.log_operation(f"   📊 Check logs/integrated_optimization_pruning.log for details")
        else:
            self.log_operation(f"   ⚠️ Some operations failed - check logs for details")
            self.log_operation(f"   💡 Consider running individual components separately")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        
        # Get current system metrics
        system_report = self.system_optimizer.generate_optimization_report()
        
        # Get repository health
        repo_health = self.optimized_pruner.analyze_repository_health_fast()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_performance": system_report,
            "repository_health": repo_health,
            "optimization_history_count": len(self.optimization_history),
            "last_optimization": self.optimization_history[-1] if self.optimization_history else None,
            "recommendations": self._get_current_recommendations(system_report, repo_health)
        }
    
    def _get_current_recommendations(self, system_report: Dict, repo_health: Dict) -> List[str]:
        """Generate current recommendations based on status"""
        recommendations = []
        
        # System performance recommendations
        if system_report["performance_score"] < 70:
            recommendations.append("🚀 Run system optimization for better performance")
        
        # Repository health recommendations
        if repo_health["health_score"] < 70:
            recommendations.append("🧠 Run biological pruning to improve repository health")
        
        # Memory recommendations
        memory_usage = system_report["system_resources"]["memory"]["usage_percent"]
        if memory_usage > 80:
            recommendations.append("💾 High memory usage - consider restarting applications")
        
        # Disk space recommendations
        disk_usage = system_report["system_resources"]["disk"]["usage_percent"]
        if disk_usage > 85:
            recommendations.append("💽 Low disk space - consider cleaning up large files")
        
        if not recommendations:
            recommendations.append("✅ System is optimally configured!")
        
        return recommendations

def main():
    """Main function for integrated optimization and pruning"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integrated Optimization and Pruning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete integrated workflow (optimize → prune → optimize)
  python integrated_optimization_pruning.py --workflow
  
  # Run optimization only
  python integrated_optimization_pruning.py --optimize-only
  
  # Check current status
  python integrated_optimization_pruning.py --status
  
  # Run pruning with automatic optimization
  python integrated_optimization_pruning.py --prune --auto-optimize
        """
    )
    
    parser.add_argument('--workflow', action='store_true',
                       help='Run complete integrated workflow')
    parser.add_argument('--optimize-only', action='store_true',
                       help='Run optimization only (no pruning)')
    parser.add_argument('--prune', action='store_true',
                       help='Run pruning workflow')
    parser.add_argument('--status', action='store_true',
                       help='Check current optimization status')
    parser.add_argument('--auto-optimize', action='store_true', default=True,
                       help='Automatically run optimization with pruning')
    parser.add_argument('--fast', action='store_true',
                       help='Use fast pruning mode')
    parser.add_argument('--repo-root', type=str, default='.',
                       help='Repository root directory')
    
    args = parser.parse_args()
    
    # Initialize integrated system
    integrated_system = IntegratedOptimizationPruning(
        repo_root=args.repo_root,
        fast_mode=args.fast,
        auto_optimize=args.auto_optimize
    )
    
    try:
        if args.workflow:
            # Run complete integrated workflow
            results = integrated_system.run_integrated_pruning_workflow_with_optimization()
            
        elif args.optimize_only:
            # Run optimization only
            print("🚀 Running system optimization...")
            pre_results = integrated_system.run_pre_pruning_optimization()
            post_results = integrated_system.run_post_pruning_optimization()
            
        elif args.prune:
            # Run pruning with optional optimization
            if args.auto_optimize:
                results = integrated_system.run_integrated_pruning_workflow_with_optimization()
            else:
                integrated_system.integrated_pruner.run_complete_pruning_workflow()
                
        elif args.status:
            # Show current status
            status = integrated_system.get_optimization_status()
            
            print("📊 INTEGRATED SYSTEM STATUS")
            print("=" * 35)
            print(f"System Performance Score: {status['system_performance']['performance_score']}/100")
            print(f"Repository Health Score: {status['repository_health']['health_score']:.1f}/100")
            print(f"Memory Usage: {status['system_performance']['system_resources']['memory']['usage_percent']:.1f}%")
            print(f"Optimization History: {status['optimization_history_count']} entries")
            
            print(f"\n💡 Current Recommendations:")
            for rec in status['recommendations']:
                print(f"   {rec}")
                
        else:
            # Default: show help
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error in integrated optimization and pruning system: {e}")
        raise

if __name__ == "__main__":
    main()
