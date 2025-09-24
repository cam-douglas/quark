#!/usr/bin/env python3
"""
Sprint Guide Module
===================
Interactive sprint validation workflow guide - shows what needs validation, never auto-validates.
Modularized to stay under 300 LOC.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Import phase and helper modules
try:
    from .sprint_phases import SprintPhases
    from .validation_helpers import ValidationHelpers
except ImportError:
    # Fallback for direct execution
    from sprint_phases import SprintPhases
    from validation_helpers import ValidationHelpers


class SprintGuide:
    """Interactive guide for sprint validation workflow - manual validation only."""
    
    def __init__(self, validation_root: Path):
        self.validation_root = validation_root
        self.checklists_dir = validation_root / "checklists"
        self.evidence_dir = validation_root / "evidence"
        self.master_checklist = validation_root / "MASTER_VALIDATION_CHECKLIST.md"
        self.master_index = validation_root / "VALIDATION_MASTER_INDEX.md"
        
        self.current_sprint = {
            "phase": "planning",
            "scope": None,
            "run_id": None,
            "validation_requirements": [],
            "evidence_collected": False
        }
        
        # Initialize phase handler
        self.phases = SprintPhases(validation_root, self.current_sprint)
        self.helpers = ValidationHelpers()
    
    def run_interactive_sprint(self) -> None:
        """Run the complete interactive sprint validation workflow."""
        print("\n" + "=" * 60)
        print("🚀 QUARK VALIDATION SPRINT GUIDE")
        print("=" * 60)
        print("\n⚠️  IMPORTANT: This guide shows what needs validation.")
        print("📝 You must perform all validation manually.")
        print("❌ NO automatic validation will be performed.\n")
        
        input("Press Enter to begin the validation guide...")
        
        # Phase 1: Planning - Show what needs validation
        self.phases.phase_planning()
        
        # Phase 2: Scope Selection - Identify validation requirements
        self.phases.phase_scope_selection()
        
        # Phase 3: Prerequisites - Check dependencies
        self.phases.phase_prerequisites()
        
        # Phase 4: Show validation requirements
        self.phases.phase_show_requirements()
        
        # Phase 5: Guide manual validation
        self._phase_manual_validation()
        
        # Phase 6: Evidence collection guidance
        self._phase_evidence_guidance()
        
        # Phase 7: Verification checklist
        self.phases.phase_verification_checklist()
        
        # Phase 8: Finalize
        self.phases.phase_finalize()
        
        print("\n" + "=" * 60)
        print("✅ VALIDATION GUIDE COMPLETE")
        print("=" * 60)
        print("\n📝 Remember: All validation must be done manually!")
    
    def _phase_manual_validation(self) -> None:
        """Phase 5: Guide through manual validation process."""
        print("\n🔧 PHASE 5: MANUAL VALIDATION")
        print("-" * 40)
        
        # Show manual validation steps
        self.helpers.show_manual_validation_steps()
        
        input("\nPress Enter when you have completed manual validation...")
        
        # Ask if they want to record measurements
        record = input("\nWould you like guidance on recording your measurements? (y/n): ").lower()
        if record == 'y':
            self.helpers.guide_measurement_recording(self.current_sprint)
        
        self.current_sprint["phase"] = "evidence"
    
    def _phase_evidence_guidance(self) -> None:
        """Phase 6: Guide evidence collection."""
        print("\n📁 PHASE 6: EVIDENCE COLLECTION - Documenting Your Validation")
        print("-" * 40)
        
        # Create run directory
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_sprint["run_id"] = run_id
        
        print(f"\n📂 Create evidence directory: evidence/{run_id}/")
        print("\nRequired evidence files:")
        
        evidence_files = {
            "metrics.json": "Your KPI measurements",
            "config.yaml": "Configuration used for validation",
            "seeds.txt": "Random seeds for reproducibility",
            "environment.txt": "System and software versions",
            "dataset_hashes.txt": "Data integrity verification",
            "logs.txt": "Execution logs from your validation"
        }
        
        for file_name, description in evidence_files.items():
            print(f"\n📄 {file_name}")
            print(f"   Purpose: {description}")
            print(f"   Status: □ Create manually")
        
        # Offer to create directory structure
        create = input("\nCreate evidence directory structure? (y/n): ").lower()
        if create == 'y':
            run_dir = self.evidence_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n✅ Created: {run_dir}")
            
            # Create template files with guidance
            self.helpers.create_evidence_templates(run_dir, self.current_sprint)
            
            print("\n📝 Template files created with instructions")
            print(f"   Edit files in: {run_dir}")
        
        self.current_sprint["phase"] = "verification"
    
    def quick_validate(self, scope: str = None) -> Dict[str, Any]:
        """Quick validation mode - shows requirements without full sprint flow."""
        print("\n📋 QUICK VALIDATION MODE")
        print("=" * 50)
        print("⚠️  This shows what needs validation - no automatic checks\n")
        
        if not scope:
            # Auto-detect from git changes
            import subprocess
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                changed_files = result.stdout.strip().split('\n')
                scope = self._suggest_scope_from_files(changed_files)
        
        if not scope:
            scope = "MAIN_INTEGRATIONS_CHECKLIST"
        
        print(f"📌 Validation scope: {scope}")
        
        # Show requirements
        checklist_path = self.checklists_dir / f"{scope}.md"
        if checklist_path.exists():
            self._show_checklist_requirements(checklist_path)
        else:
            print(f"⚠️ Checklist not found: {scope}")
            print("\nShowing general requirements:")
            self._show_general_requirements()
        
        # Create evidence directory
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.evidence_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create templates
        sprint_data = {"scope": scope, "measurements": {}}
        self.helpers.create_evidence_templates(run_dir, sprint_data)
        
        print(f"\n📁 Evidence templates created in: {run_dir}")
        print("📝 Fill these out with your manual validation results")
        
        return {
            "scope": scope,
            "run_id": run_id,
            "gate_passed": False,  # Always false - manual validation required
            "message": "Manual validation required - check requirements above"
        }
    
    def _suggest_scope_from_files(self, changed_files: list) -> str:
        """Suggest validation scope based on changed files."""
        for file in changed_files:
            if "morphogen" in file.lower():
                return "STAGE1_EMBRYONIC_CHECKLIST"
            elif "neural" in file.lower():
                return "STAGE2_FETAL_CHECKLIST"
            elif "synapse" in file.lower():
                return "STAGE3_EARLY_POSTNATAL_CHECKLIST"
            elif "integration" in file.lower():
                return "MAIN_INTEGRATIONS_CHECKLIST"
        return None
    
    def _show_checklist_requirements(self, checklist_path: Path) -> None:
        """Show requirements from a checklist."""
        print("\n📋 Validation Requirements:")
        print("-" * 40)
        
        with open(checklist_path) as f:
            lines = f.readlines()
        
        print("\n🎯 Milestone Gates:")
        for line in lines:
            if "- [" in line and "]" in line:
                checked = "[x]" in line.lower()
                status = "✅" if checked else "□"
                gate = line.split("]", 1)[1].strip() if "]" in line else line.strip()
                print(f"   {status} {gate}")
        
        print("\n📊 KPIs to Measure:")
        for i, line in enumerate(lines):
            if "**KPI:**" in line:
                kpi = line.split("**KPI:**")[1].strip()
                print(f"   • {kpi}")
                if i + 1 < len(lines) and "**Target:**" in lines[i + 1]:
                    target = lines[i + 1].split("**Target:**")[1].strip()
                    print(f"     Target: {target}")
        
        print("\n⚠️ All items must be validated manually!")
    
    def _show_general_requirements(self) -> None:
        """Show general validation requirements."""
        print("\n✓ Review MASTER_VALIDATION_CHECKLIST.md")
        print("✓ Identify your validation domain")
        print("✓ Measure all KPIs manually")
        print("✓ Compare against targets")
        print("✓ Document evidence")
        print("\n⚠️ No automatic validation performed!")