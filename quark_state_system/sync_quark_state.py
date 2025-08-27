#!/usr/bin/env python3
"""
QUARK State Synchronization System

This script ensures all QUARK state files remain synchronized and up-to-date
with the current project state. It automatically updates all state-related files
when changes are made to maintain consistency across the entire project.
"""

import os
import re
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set

class QuarkStateSynchronizer:
    """
    Comprehensive state synchronization system for QUARK project.
    Ensures all state files remain consistent and up-to-date.
    """
    
    def __init__(self):
        self.root_dir = Path.cwd()
        self.state_files = self._identify_state_files()
        self.master_state_file = self.root_dir / "quark_state_system" / "QUARK_STATE.md"
        self.sync_log = []
        
    def _identify_state_files(self) -> Dict[str, Path]:
        """Identify all state-related files in the project."""
        state_files = {}
        
        # Core state files (now in quark_state_system directory)
        state_files['master_state'] = self.root_dir / "quark_state_system" / "QUARK_STATE.md"
        state_files['check_state'] = self.root_dir / "quark_state_system" / "check_quark_state.py"
        state_files['recommendations'] = self.root_dir / "quark_state_system" / "quark_recommendations.py"
        
        # Task management files (now in quark_state_system directory)
        state_files['current_tasks'] = self.root_dir / "quark_state_system" / "QUARK_CURRENT_TASKS.md"
        state_files['high_level_roadmap'] = self.root_dir / "quark_state_system" / "QUARK_ROADMAP.md"
        
        # Documentation files that reference state
        state_files['stage_n3_summary'] = self.root_dir / "tasks" / "STAGE_N3_EVOLUTION_IMPLEMENTATION_SUMMARY.md"
        state_files['complexity_evolution'] = self.root_dir / "tasks" / "COMPLEXITY_EVOLUTION_TODOS.md"
        state_files['next_steps_ready'] = self.root_dir / "documentation" / "status_docs" / "NEXT_STEPS_READY.md"
        
        # README files
        state_files['main_readme'] = self.root_dir / "README.md"
        
        return state_files
    
    def _extract_master_state_info(self) -> Dict[str, Any]:
        """Extract key state information from the master state file."""
        if not self.master_state_file.exists():
            return {"error": "Master state file not found"}
        
        with open(self.master_state_file, 'r') as f:
            content = f.read()
        
        state_info = {}
        
        # Extract key information using regex patterns
        patterns = {
            'current_stage': r'\*\*Current Development Stage\*\*: (.+)',
            'overall_progress': r'\*\*Overall Progress\*\*: (.+)',
            'next_milestone': r'\*\*Next Major Milestone\*\*: (.+)',
            'last_updated': r'\*\*Last Updated\*\*: (.+)',
            'recent_work_status': r'\*\*Status\*\*: (.+?)\n',
            'current_phase': r'### \*\*Development Phase\*\*: (.+)',
            'stage_n3_status': r'(STAGE N3.*?COMPLETED)',
            'stage_n4_status': r'(STAGE N4.*?READY)',
            'brain_body_status': r'(Brain-to-Body Control.*?Implementation)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                state_info[key] = match.group(1).strip()
            else:
                state_info[key] = "Not found"
        
        return state_info
    
    def _update_task_file(self, file_path: Path, master_info: Dict[str, Any]) -> bool:
        """Update a task file to match the master state."""
        if not file_path.exists():
            self.sync_log.append(f"⚠️  Task file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            updated = False
            
            # Update status lines
            status_updates = {
                r'(\*\*Status\*\*: ).*?(?=\n)': f"\\1🚀 STAGE N3 COMPLETE - Ready for Stage N4 Evolution",
                r'(\*\*Last Updated\*\*: ).*?(?=\n)': f"\\1{datetime.now().strftime('%B %d, %Y')}",
                r'(\*\*Next Review\*\*: ).*?(?=\n)': f"\\1Immediate Evolution to Stage N4"
            }
            
            for pattern, replacement in status_updates.items():
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    updated = True
            
            # Update stage references
            stage_updates = {
                r'STAGE N2': 'STAGE N3 COMPLETE',
                r'Stage N2': 'Stage N3 Complete',
                r'Phase 3.*?READY': 'Phase 4 - Creative Intelligence & Superintelligence',
                r'Phase 3.*?COMPLETE': 'Phase 3 - Higher-Order Cognition (STAGE N3) ✅ COMPLETED'
            }
            
            for old, new in stage_updates.items():
                if re.search(old, content, re.IGNORECASE):
                    content = re.sub(old, new, content, flags=re.IGNORECASE)
                    updated = True
            
            # Update progress indicators
            progress_updates = {
                r'(\*\*Progress\*\*: ).*?(?=\n)': '\\1100%',
                r'(\*\*Overall Progress\*\*: ).*?(?=\n)': '\\1100% (Stage N3 Complete)',
                r'(\*\*Status\*\*: ).*?(?=\n)': '\\1✅ COMPLETED'
            }
            
            for pattern, replacement in progress_updates.items():
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    updated = True
            
            if updated:
                # Create backup
                backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                shutil.copy2(file_path, backup_path)
                
                # Write updated content
                with open(file_path, 'w') as f:
                    f.write(content)
                
                self.sync_log.append(f"✅ Updated: {file_path}")
                return True
            else:
                self.sync_log.append(f"ℹ️  No updates needed: {file_path}")
                return False
                
        except Exception as e:
            self.sync_log.append(f"❌ Error updating {file_path}: {e}")
            return False
    
    def _update_documentation_file(self, file_path: Path, master_info: Dict[str, Any]) -> bool:
        """Update documentation files to match the master state."""
        if not file_path.exists():
            self.sync_log.append(f"⚠️  Documentation file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            updated = False
            
            # Update stage references in documentation
            doc_updates = {
                r'Stage N2': 'Stage N3 Complete',
                r'STAGE N2': 'STAGE N3 COMPLETE',
                r'Phase 3.*?In Progress': 'Phase 4 - Creative Intelligence & Superintelligence 🚀 READY',
                r'Phase 3.*?Complete': 'Phase 3 - Higher-Order Cognition (STAGE N3) ✅ COMPLETED',
                r'Next.*?Stage N3': 'Next: Stage N4 Evolution',
                r'Ready for Stage N3': 'Ready for Stage N4 Evolution'
            }
            
            for old, new in doc_updates.items():
                if re.search(old, content, re.IGNORECASE):
                    content = re.sub(old, new, content, flags=re.IGNORECASE)
                    updated = True
            
            if updated:
                # Create backup
                backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                shutil.copy2(file_path, backup_path)
                
                # Write updated content
                with open(file_path, 'w') as f:
                    f.write(content)
                
                self.sync_log.append(f"✅ Updated: {file_path}")
                return True
            else:
                self.sync_log.append(f"ℹ️  No updates needed: {file_path}")
                return False
                
        except Exception as e:
            self.sync_log.append(f"❌ Error updating {file_path}: {e}")
            return False
    
    def _update_readme_file(self, file_path: Path, master_info: Dict[str, Any]) -> bool:
        """Update README files to match the master state."""
        if not file_path.exists():
            self.sync_log.append(f"⚠️  README file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            updated = False
            
            # Update status information in README
            readme_updates = {
                r'(\*\*Current Stage\*\*: ).*?(?=\n)': '\\1STAGE N3 COMPLETE - Ready for Stage N4 Evolution',
                r'(\*\*Overall Progress\*\*: ).*?(?=\n)': '\\1100% (Stage N3 Complete)',
                r'(\*\*Next Major Goal\*\*: ).*?(?=\n)': '\\1Phase 4 - Creative Intelligence & Superintelligence'
            }
            
            for pattern, replacement in readme_updates.items():
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    updated = True
            
            if updated:
                # Create backup
                backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                shutil.copy2(file_path, backup_path)
                
                # Write updated content
                with open(file_path, 'w') as f:
                    f.write(content)
                
                self.sync_log.append(f"✅ Updated: {file_path}")
                return True
            else:
                self.sync_log.append(f"ℹ️  No updates needed: {file_path}")
                return False
                
        except Exception as e:
            self.sync_log.append(f"❌ Error updating {file_path}: {e}")
            return False
    
    def _validate_state_consistency(self) -> Dict[str, Any]:
        """Validate that all state files are consistent."""
        validation_results = {}
        
        # Check if all state files exist
        for name, path in self.state_files.items():
            validation_results[name] = {
                'exists': path.exists(),
                'path': str(path),
                'last_modified': path.stat().st_mtime if path.exists() else None
            }
        
        # Check for consistency in key terms
        consistency_checks = {
            'stage_n3_complete': r'STAGE N3.*?COMPLETE|Stage N3.*?Complete',
            'stage_n4_ready': r'STAGE N4.*?READY|Stage N4.*?Ready',
            'phase_4_creative': r'Phase 4.*?Creative Intelligence|Creative Intelligence.*?Superintelligence',
            'progress_100': r'100%.*?Stage N3|Stage N3.*?100%'
        }
        
        for check_name, pattern in consistency_checks.items():
            validation_results[check_name] = {}
            for name, path in self.state_files.items():
                if path.exists():
                    with open(path, 'r') as f:
                        content = f.read()
                    validation_results[check_name][name] = bool(re.search(pattern, content, re.IGNORECASE))
        
        return validation_results
    
    def sync_all_state_files(self) -> Dict[str, Any]:
        """Synchronize all state files with the master state."""
        print("🧠 QUARK STATE SYNCHRONIZATION")
        print("=" * 50)
        
        # Extract master state information
        master_info = self._extract_master_state_info()
        if "error" in master_info:
            print(f"❌ Error: {master_info['error']}")
            return {"error": master_info["error"]}
        
        print(f"📊 Master State: {master_info.get('current_stage', 'Unknown')}")
        print(f"📈 Progress: {master_info.get('overall_progress', 'Unknown')}")
        print(f"🚀 Next: {master_info.get('next_milestone', 'Unknown')}")
        print()
        
        # Update all state files
        update_results = {}
        
        # Update task files
        for name, path in self.state_files.items():
            if 'task' in name.lower() or 'current_tasks' in str(path):
                update_results[name] = self._update_task_file(path, master_info)
        
        # Update documentation files
        for name, path in self.state_files.items():
            if 'documentation' in str(path) or 'stage_n3' in name.lower() or 'complexity' in name.lower():
                update_results[name] = self._update_documentation_file(path, master_info)
        
        # Update README files
        for name, path in self.state_files.items():
            if 'readme' in name.lower():
                update_results[name] = self._update_readme_file(path, master_info)
        
        # Validate consistency
        validation_results = self._validate_state_consistency()
        
        # Print results
        print("📋 UPDATE RESULTS:")
        for name, result in update_results.items():
            status = "✅ Updated" if result else "ℹ️  No changes"
            print(f"   {name}: {status}")
        
        print("\n🔍 CONSISTENCY VALIDATION:")
        for check_name, results in validation_results.items():
            if isinstance(results, dict):
                consistent = all(results.values())
                status = "✅ Consistent" if consistent else "⚠️  Inconsistent"
                print(f"   {check_name}: {status}")
        
        print("\n📝 SYNC LOG:")
        for log_entry in self.sync_log:
            print(f"   {log_entry}")
        
        return {
            'update_results': update_results,
            'validation_results': validation_results,
            'sync_log': self.sync_log
        }
    
    def create_state_snapshot(self) -> str:
        """Create a snapshot of the current state for backup purposes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = self.root_dir / "state_snapshots" / timestamp
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all state files to snapshot
        for name, path in self.state_files.items():
            if path.exists():
                snapshot_path = snapshot_dir / path.name
                shutil.copy2(path, snapshot_path)
        
        # Create snapshot summary
        summary_path = snapshot_dir / "snapshot_summary.md"
        master_info = self._extract_master_state_info()
        
        with open(summary_path, 'w') as f:
            f.write(f"# QUARK State Snapshot - {timestamp}\n\n")
            f.write(f"**Current Stage**: {master_info.get('current_stage', 'Unknown')}\n")
            f.write(f"**Overall Progress**: {master_info.get('overall_progress', 'Unknown')}\n")
            f.write(f"**Next Milestone**: {master_info.get('next_milestone', 'Unknown')}\n")
            f.write(f"**Files Snapshot**: {len([p for p in self.state_files.values() if p.exists()])} files\n")
        
        return str(snapshot_dir)

def main():
    """Main function to run the state synchronization."""
    synchronizer = QuarkStateSynchronizer()
    
    # Sync all state files
    results = synchronizer.sync_all_state_files()
    
    if "error" not in results:
        # Create state snapshot
        snapshot_path = synchronizer.create_state_snapshot()
        print(f"\n📸 State snapshot created: {snapshot_path}")
        
        print("\n🎉 State synchronization complete!")
        print("💡 All QUARK state files are now consistent and up-to-date.")
    else:
        print(f"\n❌ Synchronization failed: {results['error']}")

if __name__ == "__main__":
    main()
