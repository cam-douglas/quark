#!/usr/bin/env python3
"""
Scope Selector Module
====================
Helps users identify the appropriate validation scope for their changes.
"""

import subprocess
from pathlib import Path
from typing import List, Set, Dict, Optional


class ScopeSelector:
    """Interactive scope selection for validation."""
    
    def __init__(self, validation_root: Path):
        self.validation_root = validation_root
        self.checklists_dir = validation_root / "checklists"
        self.master_index = validation_root / "VALIDATION_MASTER_INDEX.md"
        
        # Map file patterns to validation domains
        self.domain_mappings = {
            "morphogen": ["STAGE1_EMBRYONIC"],
            "neural": ["STAGE2_FETAL", "MAIN_INTEGRATIONS"],
            "synapse": ["STAGE3_EARLY_POSTNATAL"],
            "myelin": ["STAGE4_CHILDHOOD"],
            "pruning": ["STAGE5_ADOLESCENCE"],
            "adult": ["STAGE6_ADULT"],
            "perception": ["MAIN_INTEGRATIONS", "APPENDIX_C_BENCHMARKS"],
            "cognitive": ["MAIN_INTEGRATIONS", "APPENDIX_C_BENCHMARKS"],
            "language": ["MAIN_INTEGRATIONS", "APPENDIX_C_BENCHMARKS"],
            "action": ["MAIN_INTEGRATIONS", "APPENDIX_C_BENCHMARKS"],
            "social": ["MAIN_INTEGRATIONS", "APPENDIX_C_BENCHMARKS"],
            "vllm": ["SYSTEM_DESIGN", "MAIN_INTEGRATIONS"],
            "deployment": ["DELIVERABLES", "SYSTEM_DESIGN"],
        }
    
    def suggest_from_git_diff(self) -> Set[str]:
        """Suggest validation scope based on git diff."""
        suggestions = set()
        
        try:
            # Get changed files
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True, text=True, check=True
            )
            
            changed_files = result.stdout.strip().split('\n')
            
            # Map files to domains
            for file_path in changed_files:
                file_lower = file_path.lower()
                for pattern, domains in self.domain_mappings.items():
                    if pattern in file_lower:
                        suggestions.update(domains)
        
        except subprocess.CalledProcessError:
            print("⚠️ Could not read git diff")
        
        return suggestions
    
    def interactive_select(self) -> Optional[str]:
        """Interactive menu for scope selection."""
        print("\n📋 Select Validation Scope:")
        print("=" * 40)
        
        # List available checklists
        checklists = sorted(self.checklists_dir.glob("*.md"))
        
        for i, checklist in enumerate(checklists, 1):
            name = checklist.stem.replace("_CHECKLIST", "")
            print(f"{i:2}. {name}")
        
        print("\n 0. Exit")
        
        try:
            choice = int(input("\nSelect scope (number): "))
            if 0 < choice <= len(checklists):
                return checklists[choice - 1].stem
        except (ValueError, IndexError):
            print("Invalid selection")
        
        return None
    
    def show_prerequisites(self, scope: str) -> None:
        """Display prerequisites for selected scope."""
        checklist_path = self.checklists_dir / f"{scope}.md"
        
        if not checklist_path.exists():
            print(f"⚠️ Checklist not found: {scope}")
            return
        
        print(f"\n📌 Prerequisites for {scope}:")
        print("=" * 40)
        
        with open(checklist_path) as f:
            lines = f.readlines()
            in_deps = False
            
            for line in lines:
                if "## Dependencies" in line:
                    in_deps = True
                elif in_deps and line.startswith("##"):
                    break
                elif in_deps and line.strip():
                    print(line.rstrip())
    
    def get_checklist_path(self, scope: str) -> Path:
        """Get full path to checklist file."""
        return self.checklists_dir / f"{scope}.md"
