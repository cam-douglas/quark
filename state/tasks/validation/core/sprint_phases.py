#!/usr/bin/env python3
"""
Sprint Phases Module
====================
Individual phase implementations for sprint validation.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class SprintPhases:
    """Handles individual phases of the sprint validation process."""
    
    def __init__(self, validation_root: Path, current_sprint: Dict):
        self.validation_root = validation_root
        self.checklists_dir = validation_root / "checklists"
        self.evidence_dir = validation_root / "evidence"
        self.current_sprint = current_sprint
    
    def phase_planning(self) -> None:
        """Phase 1: Planning - Show validation overview."""
        print("\n📋 PHASE 1: PLANNING - Understanding Validation Requirements")
        print("-" * 40)
        
        print("\nValidation Overview:")
        print("• Review MASTER_VALIDATION_CHECKLIST.md for requirements")
        print("• Identify which domain/stage you're working on")
        print("• Understand KPI targets and thresholds")
        print("• Prepare to measure and document manually")
        
        print("\nSelect your validation focus:")
        print("1. Biological Stage Validation (Stages 1-6)")
        print("2. Integration Domain Validation")
        print("3. System Design Validation")
        print("4. Full Brain Simulation")
        print("5. Show all requirements")
        
        while True:
            try:
                choice = input("\nEnter choice (1-5): ").strip()
                if choice in ["1", "2", "3", "4", "5"]:
                    break
                print("Please enter a number between 1 and 5")
            except KeyboardInterrupt:
                print("\n\n⚠️ Validation guide cancelled")
                sys.exit(0)
        
        category_map = {
            "1": "biological",
            "2": "integration",
            "3": "system",
            "4": "full_brain",
            "5": "all"
        }
        self.current_sprint["category"] = category_map.get(choice, "all")
        self.current_sprint["phase"] = "scope_selection"
        print("\n✅ Planning phase complete - requirements identified")
    
    def phase_scope_selection(self) -> None:
        """Phase 2: Scope Selection - Identify specific validation requirements."""
        print("\n🎯 PHASE 2: SCOPE SELECTION - What Needs Validation")
        print("-" * 40)
        
        category = self.current_sprint.get("category", "all")
        available_checklists = self._get_available_checklists(category)
        
        print("\nAvailable validation checklists:")
        for i, checklist in enumerate(available_checklists, 1):
            print(f"{i:2}. {checklist}")
        
        print("\n 0. Show requirements for all")
        
        while True:
            try:
                choice = input("\nSelect checklist to view requirements (0-{}): ".format(len(available_checklists))).strip()
                choice_int = int(choice)
                if 0 <= choice_int <= len(available_checklists):
                    break
                print(f"Please enter a number between 0 and {len(available_checklists)}")
            except (ValueError, KeyboardInterrupt):
                print("\nUsing default scope")
                choice_int = 0
                break
        
        if choice_int == 0:
            self.current_sprint["scope"] = "ALL"
            print("\n📋 Will show requirements for all domains")
        else:
            self.current_sprint["scope"] = available_checklists[choice_int - 1]
            print(f"\n✅ Selected scope: {self.current_sprint['scope']}")
        
        self.current_sprint["phase"] = "prerequisites"
    
    def phase_prerequisites(self) -> None:
        """Phase 3: Prerequisites - Check dependencies."""
        print("\n🔗 PHASE 3: PREREQUISITES - Checking Dependencies")
        print("-" * 40)
        
        scope = self.current_sprint["scope"]
        
        if scope == "ALL":
            print("\n📋 General prerequisites:")
            print("• Ensure previous stages are validated (if applicable)")
            print("• Have measurement tools ready")
            print("• Prepare evidence directory")
        else:
            self._show_specific_prerequisites(scope)
        
        print("\n⚠️ You must verify all prerequisites manually before proceeding")
        input("\nPress Enter when prerequisites are verified...")
        
        self.current_sprint["phase"] = "requirements"
    
    def phase_show_requirements(self) -> None:
        """Phase 4: Show detailed validation requirements."""
        print("\n📊 PHASE 4: VALIDATION REQUIREMENTS - What You Must Validate")
        print("-" * 40)
        
        scope = self.current_sprint["scope"]
        
        if scope == "ALL":
            print("\n📋 Review all checklists in state/tasks/validation/checklists/")
            print("Each checklist contains:")
            print("• Milestone gates to verify")
            print("• KPIs to measure with targets")
            print("• Evidence to collect")
        else:
            checklist_path = self.checklists_dir / f"{scope}.md"
            if checklist_path.exists():
                self._display_detailed_requirements(checklist_path)
            else:
                print(f"⚠️ Checklist not found: {scope}")
        
        self.current_sprint["phase"] = "manual_validation"
    
    def phase_verification_checklist(self) -> None:
        """Phase 7: Verification checklist."""
        print("\n✅ PHASE 7: VERIFICATION CHECKLIST")
        print("-" * 40)
        
        print("\nManually verify each item:")
        
        checklist = [
            "All milestone gates checked",
            "All KPIs measured",
            "Measurements compared against targets",
            "Evidence files created",
            "Configuration documented",
            "Seeds recorded",
            "Environment captured",
            "Logs saved"
        ]
        
        verified_items = []
        for item in checklist:
            status = input(f"\n□ {item} - Verified? (y/n): ").lower()
            verified = status == 'y'
            verified_items.append((item, verified))
            if verified:
                print("  ✅ Verified")
            else:
                print("  ⚠️ Not verified - please complete manually")
        
        # Summary
        print("\n" + "=" * 50)
        print("VERIFICATION SUMMARY")
        all_verified = True
        for item, verified in verified_items:
            status = "✅" if verified else "⚠️"
            print(f"{status} {item}")
            if not verified:
                all_verified = False
        
        if all_verified:
            print("\n✅ All items verified!")
        else:
            print("\n⚠️ Some items need completion")
            print("Please complete all verification manually")
        
        self.current_sprint["phase"] = "finalize"
    
    def phase_finalize(self) -> None:
        """Phase 8: Finalize validation."""
        print("\n🎉 PHASE 8: FINALIZE")
        print("-" * 40)
        
        print("\n📊 Validation Summary:")
        print(f"• Scope: {self.current_sprint.get('scope', 'Not selected')}")
        print(f"• Run ID: {self.current_sprint.get('run_id', 'Not created')}")
        print(f"• Evidence: evidence/{self.current_sprint.get('run_id', 'N/A')}/")
        
        if self.current_sprint.get("measurements"):
            print("\n📈 Recorded Measurements:")
            for kpi, value in self.current_sprint["measurements"].items():
                print(f"   • {kpi}: {value}")
        
        print("\n📝 Next Steps:")
        print("1. Complete any unfinished verification items")
        print("2. Update checklist markdown files manually")
        print("3. Commit evidence to repository")
        print("4. Create PR with validation results")
        
        print("\n⚠️ Remember:")
        print("• All validation is manual - no automatic checks")
        print("• You are responsible for verifying all KPIs meet targets")
        print("• Document everything for reproducibility")
    
    # Helper methods
    def _get_available_checklists(self, category: str) -> List[str]:
        """Get available checklists based on category."""
        available_checklists = []
        for checklist_file in sorted(self.checklists_dir.glob("*.md")):
            name = checklist_file.stem
            
            if category == "biological" and "STAGE" in name:
                available_checklists.append(name)
            elif category == "integration" and "INTEGRATIONS" in name:
                available_checklists.append(name)
            elif category == "system" and "SYSTEM" in name:
                available_checklists.append(name)
            elif category == "full_brain" and "APPENDIX" in name:
                available_checklists.append(name)
            elif category == "all":
                available_checklists.append(name)
        
        return available_checklists
    
    def _show_specific_prerequisites(self, scope: str) -> None:
        """Show specific prerequisites for a scope."""
        # Check stage prerequisites if applicable
        if "STAGE" in scope:
            import re
            match = re.search(r'STAGE(\d)', scope)
            if match:
                stage_num = int(match.group(1))
                if stage_num > 1:
                    print(f"\n⚠️ Stage {stage_num} requires previous stages to be validated:")
                    for i in range(1, stage_num):
                        print(f"   □ Stage {i} validation complete")
        
        # Show explicit dependencies from checklist
        checklist_path = self.checklists_dir / f"{scope}.md"
        if checklist_path.exists():
            deps = self._extract_dependencies(checklist_path)
            if deps:
                print("\n📌 Explicit dependencies:")
                for dep in deps:
                    print(f"   • {dep}")
    
    def _extract_dependencies(self, checklist_path: Path) -> List[str]:
        """Extract dependencies from checklist."""
        deps = []
        with open(checklist_path) as f:
            lines = f.readlines()
            in_deps = False
            for line in lines:
                if "## Dependencies" in line:
                    in_deps = True
                elif in_deps and line.startswith("##"):
                    break
                elif in_deps and line.strip().startswith("-"):
                    deps.append(line.strip()[1:].strip())
        return deps
    
    def _display_detailed_requirements(self, checklist_path: Path) -> None:
        """Display detailed requirements from checklist."""
        print(f"\n📋 Requirements from: {checklist_path.name}")
        print("=" * 50)
        
        with open(checklist_path) as f:
            content = f.read()
        
        lines = content.split('\n')
        
        print("\n🎯 Milestone Gates (Verify Manually):")
        for line in lines:
            if "- [" in line and "]" in line:
                print(f"   {line.strip()}")
        
        print("\n📊 KPIs to Measure:")
        for i, line in enumerate(lines):
            if "**KPI:**" in line:
                print(f"   {line.strip()}")
                # Show target if on next line
                if i + 1 < len(lines) and "**Target:**" in lines[i + 1]:
                    print(f"   {lines[i + 1].strip()}")
        
        print("\n⚠️ You must validate each item manually!")
