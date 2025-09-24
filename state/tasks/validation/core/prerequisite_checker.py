#!/usr/bin/env python3
"""
Prerequisite Checker Module
===========================
Validates that all dependencies are met before proceeding with validation.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class PrerequisiteChecker:
    """Check and enforce validation prerequisites."""
    
    def __init__(self, validation_root: Path):
        self.validation_root = validation_root
        self.evidence_dir = validation_root / "evidence"
        self.checklists_dir = validation_root / "checklists"
        
        # Stage ordering
        self.stage_order = [
            "STAGE1_EMBRYONIC",
            "STAGE2_FETAL", 
            "STAGE3_EARLY_POSTNATAL",
            "STAGE4_CHILDHOOD",
            "STAGE5_ADOLESCENCE",
            "STAGE6_ADULT"
        ]
    
    def check_stage_prerequisites(self, target_stage: str) -> Tuple[bool, List[str]]:
        """Check if prerequisite stages are complete."""
        missing = []
        
        if target_stage not in self.stage_order:
            return True, []  # Not a stage checklist
        
        target_idx = self.stage_order.index(target_stage)
        
        # Check all prior stages
        for i in range(target_idx):
            stage = self.stage_order[i]
            if not self._is_stage_complete(stage):
                missing.append(stage)
        
        return len(missing) == 0, missing
    
    def _is_stage_complete(self, stage: str) -> bool:
        """Check if a stage has completed validation."""
        # Look for evidence of stage completion
        stage_evidence = list(self.evidence_dir.glob(f"*/{stage}_complete.json"))
        
        if not stage_evidence:
            return False
        
        # Check the most recent evidence
        latest = sorted(stage_evidence)[-1]
        
        try:
            with open(latest) as f:
                data = json.load(f)
                return data.get("complete", False)
        except (json.JSONDecodeError, KeyError):
            return False
    
    def extract_dependencies(self, checklist_path: Path) -> List[str]:
        """Extract dependencies from a checklist file."""
        deps = []
        
        if not checklist_path.exists():
            return deps
        
        with open(checklist_path) as f:
            lines = f.readlines()
            in_deps = False
            
            for line in lines:
                if "## Dependencies" in line:
                    in_deps = True
                elif in_deps and line.startswith("##"):
                    break
                elif in_deps and "- [" in line:
                    # Extract linked dependency
                    start = line.find("[")
                    end = line.find("]")
                    if start >= 0 and end > start:
                        deps.append(line[start+1:end])
        
        return deps
    
    def validate_evidence_structure(self, run_id: str) -> Dict[str, bool]:
        """Validate that evidence structure is complete."""
        run_dir = self.evidence_dir / run_id
        
        required_files = {
            "metrics.json": False,
            "config.yaml": False,
            "logs.txt": False,
            "seeds.txt": False,
            "environment.txt": False,
            "dataset_hashes.txt": False
        }
        
        if not run_dir.exists():
            return required_files
        
        for file_name in required_files:
            if (run_dir / file_name).exists():
                required_files[file_name] = True
        
        return required_files
    
    def generate_prerequisite_report(self, target_scope: str) -> str:
        """Generate a prerequisite validation report."""
        report = []
        report.append(f"\n📋 Prerequisite Report for {target_scope}")
        report.append("=" * 50)
        
        # Check stage prerequisites
        if "STAGE" in target_scope:
            complete, missing = self.check_stage_prerequisites(target_scope)
            
            if complete:
                report.append("✅ All stage prerequisites met")
            else:
                report.append("⚠️ Missing stage prerequisites:")
                for stage in missing:
                    report.append(f"   ❌ {stage}")
        
        # Check explicit dependencies
        checklist_path = self.checklists_dir / f"{target_scope}.md"
        deps = self.extract_dependencies(checklist_path)
        
        if deps:
            report.append("\n🔗 Explicit Dependencies:")
            for dep in deps:
                report.append(f"   → {dep}")
        
        return "\n".join(report)
