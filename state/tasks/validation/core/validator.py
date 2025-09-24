#!/usr/bin/env python3
"""
Core Validator Module
=====================
Main validation orchestrator - Interactive guide only, no automatic validation.
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class QuarkValidator:
    """Master validation orchestrator for Quark project - Interactive only."""
    
    def __init__(self):
        self.validation_root = Path(__file__).parent.parent
        self.workspace_root = self.validation_root.parent.parent.parent
        self.checklists_dir = self.validation_root / "checklists"
        self.evidence_dir = self.validation_root / "evidence"
        self.templates_dir = self.validation_root / "templates"
        self.dashboards_dir = self.validation_root / "dashboards"
        
        # Ensure directories exist
        for dir_path in [self.validation_root, self.checklists_dir, 
                         self.evidence_dir, self.templates_dir, self.dashboards_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def sprint_mode(self) -> None:
        """Interactive sprint validation guide - shows what needs validation."""
        print("\n🚀 QUARK VALIDATION GUIDE - SPRINT MODE")
        print("=" * 50)
        print("ℹ️  This guide will show you what needs validation.")
        print("⚠️  NO automatic validation - you must validate manually.\n")
        
        try:
            # Try relative import first
            try:
                from .sprint_guide import SprintGuide
            except ImportError:
                # Fallback to absolute import
                import sys
                sys.path.insert(0, str(self.validation_root / "core"))
                from sprint_guide import SprintGuide
            
            guide = SprintGuide(self.validation_root)
            guide.run_interactive_sprint()
        except (ImportError, NameError) as e:
            print(f"Sprint guide module initializing... {e}")
            self._run_basic_sprint()
    
    def _run_basic_sprint(self) -> None:
        """Fallback sprint mode - shows validation requirements."""
        print("\n📋 Sprint Validation Requirements:")
        print("\n1️⃣  PLANNING")
        print("   ✓ Review MASTER_VALIDATION_CHECKLIST.md")
        print("   ✓ Identify target KPIs for this sprint")
        print("   ✓ Review acceptance thresholds")
        
        print("\n2️⃣  WHAT NEEDS VALIDATION")
        print("   ✓ Check stage prerequisites")
        print("   ✓ List required KPIs and targets")
        print("   ✓ Identify missing evidence")
        
        print("\n3️⃣  MANUAL VALIDATION STEPS")
        print("   ✓ Run your implementation")
        print("   ✓ Measure each KPI manually")
        print("   ✓ Compare against targets")
        print("   ✓ Document results")
        
        print("\n4️⃣  EVIDENCE COLLECTION")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"   ✓ Create evidence/{run_id}/")
        print("   ✓ Record your measurements")
        print("   ✓ Save configs and seeds")
        
        print("\n5️⃣  VERIFICATION")
        print("   ✓ Check all KPIs meet targets")
        print("   ✓ Verify evidence completeness")
        print("   ✓ Update checklists manually")
        
        print("\n⚠️  Remember: You must perform all validation manually!")
    
    def show_validation_requirements(self, scope: Optional[str] = None, stage: Optional[int] = None) -> None:
        """Show what needs validation for current changes or specified scope."""
        
        # Handle explicit scope specification
        if scope:
            print(f"\n📋 Validation Requirements for: {scope}")
            self._show_scope_requirements(scope)
            return
        
        if stage:
            # Map stage numbers to full names
            stage_names = {
                1: "EMBRYONIC", 2: "FETAL", 3: "EARLY_POSTNATAL",
                4: "CHILDHOOD", 5: "ADOLESCENCE", 6: "ADULT"
            }
            if stage in stage_names:
                full_scope = f"STAGE{stage}_{stage_names[stage]}_CHECKLIST"
                print(f"\n📋 Validation Requirements for Stage {stage}: {full_scope}")
                self._show_scope_requirements(full_scope)
                return
        
        print("\n🔍 Analyzing current changes to identify validation needs...")
        
        # Get git diff to identify changed files
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            changed_files = result.stdout.strip().split('\n')
            print(f"Found {len(changed_files)} changed files")
            
            # Suggest what needs validation
            self._suggest_validation_requirements(changed_files)
        else:
            print("No changes detected. Showing general validation checklist...")
            self._show_general_requirements()
    
    def _suggest_validation_requirements(self, changed_files: List[str]) -> None:
        """Suggest what needs validation based on changed files."""
        requirements = {}
        
        for file in changed_files:
            if "brain/modules/morphogen" in file:
                requirements["STAGE1_EMBRYONIC"] = [
                    "segmentation_dice ≥ 0.80",
                    "neuron_count_error_pct ≤ 10%",
                    "gradient_smoothness ≥ 0.85"
                ]
            elif "brain/modules/neural" in file:
                requirements["STAGE2_FETAL"] = [
                    "neuron_count_error_pct ≤ 10%",
                    "laminar_accuracy ≥ 0.85",
                    "migration_completion ≥ 95%"
                ]
            elif "brain/modules/synapse" in file:
                requirements["STAGE3_EARLY_POSTNATAL"] = [
                    "synapse_density_ratio 0.8-1.2",
                    "ocular_dominance_dprime ≥ 1.5",
                    "critical_period_onset correct"
                ]
            elif "brain/ml" in file:
                requirements["MAIN_INTEGRATIONS"] = [
                    "reasoning_accuracy ≥ 0.75",
                    "calibration ECE ≤ 0.02",
                    "integration tests passing"
                ]
        
        if requirements:
            print("\n⚠️  VALIDATION REQUIRED for these domains:")
            print("=" * 50)
            for domain, kpis in requirements.items():
                print(f"\n📌 {domain}")
                print("   Required validations:")
                for kpi in kpis:
                    print(f"   □ {kpi}")
                print(f"   📄 Checklist: {domain}_CHECKLIST.md")
        else:
            print("\n✅ No specific validation requirements detected for these changes.")
            print("   Consider running general validation checklist.")
    
    def _show_scope_requirements(self, scope: str) -> None:
        """Show detailed requirements for a specific scope."""
        # Normalize and find matching checklist
        scope_normalized = scope.upper().replace("-", "_").replace(" ", "_")
        
        # Smart matching to find the right checklist
        available_checklists = [f.stem for f in sorted(self.checklists_dir.glob("*.md"))]
        
        best_match = self._find_best_checklist_match(scope_normalized, available_checklists)
        
        if not best_match:
            print(f"⚠️ No checklist found for: {scope}")
            print("\nAvailable checklists:")
            for checklist in available_checklists:
                print(f"   → {checklist}")
            return
        
        checklist_path = self.checklists_dir / f"{best_match}.md"
        
        print(f"\n📋 Validation Requirements: {best_match}")
        print("=" * 50)
        
        # Parse and display requirements
        self._display_checklist_requirements(checklist_path)
        
        print("\n⚠️  IMPORTANT: You must validate each item manually!")
        print("📝 Record your results in evidence/<run_id>/")
    
    def _find_best_checklist_match(self, scope_normalized: str, available_checklists: List[str]) -> Optional[str]:
        """Find best matching checklist using smart matching."""
        # First try exact match
        if f"{scope_normalized}_CHECKLIST" in available_checklists:
            return f"{scope_normalized}_CHECKLIST"
        elif scope_normalized in available_checklists:
            return scope_normalized
        
        # Smart matching
        scope_words = set(scope_normalized.lower().split("_"))
        filler_words = {"tasks", "layer", "the", "and", "or", "of", "for"}
        scope_words = scope_words - filler_words
        
        best_match = None
        best_score = 0
        
        for checklist in available_checklists:
            checklist_words = set(checklist.lower().split("_"))
            matching_words = scope_words & checklist_words
            score = len(matching_words)
            
            # Bonus for key terms
            if "foundation" in scope_words and "embryonic" in checklist_words:
                score += 2
            if "integration" in scope_words and "main" in checklist_words:
                score += 1
                
            if score > best_score:
                best_score = score
                best_match = checklist
        
        if best_match and best_score > 0:
            print(f"📝 Best match: {best_match} (confidence: {best_score})")
            return best_match
        
        return None
    
    def _display_checklist_requirements(self, checklist_path: Path) -> None:
        """Display requirements from a checklist file."""
        if not checklist_path.exists():
            print(f"⚠️ Checklist file not found: {checklist_path}")
            return
        
        with open(checklist_path) as f:
            lines = f.readlines()
        
        # Extract and display milestone gates
        print("\n🎯 Milestone Gates (Manual Verification Required):")
        in_gates = False
        for line in lines:
            if "### Milestone Gates" in line:
                in_gates = True
            elif in_gates and line.startswith("##"):
                break
            elif in_gates and "- [" in line:
                checked = "[x]" in line.lower()
                status = "✅" if checked else "□"
                gate = line.split("]", 1)[1].strip() if "]" in line else line.strip()
                print(f"   {status} {gate}")
        
        # Extract and display KPI requirements
        print("\n📊 KPI Requirements (You Must Measure):")
        in_kpis = False
        for line in lines:
            if "### KPI Specifications" in line or "**KPI:**" in line:
                in_kpis = True
            elif in_kpis:
                if "**KPI:**" in line:
                    kpi = line.split("**KPI:**")[1].strip()
                    print(f"\n   📌 {kpi}")
                elif "**Target:**" in line:
                    target = line.split("**Target:**")[1].strip()
                    print(f"      Target: {target}")
                elif "**Benchmark:**" in line:
                    benchmark = line.split("**Benchmark:**")[1].strip()
                    print(f"      Benchmark: {benchmark}")
                elif "**Measurement:**" in line:
                    measurement = line.split("**Measurement:**")[1].strip()
                    print(f"      How to measure: {measurement}")
                elif line.startswith("##") and "KPI" not in line:
                    break
        
        # Show evidence requirements
        print("\n📁 Evidence You Must Collect:")
        print("   □ metrics.json - Your KPI measurements")
        print("   □ config.yaml - Configuration used")
        print("   □ seeds.txt - Random seeds for reproducibility")
        print("   □ environment.txt - System information")
        print("   □ dataset_hashes.txt - Data integrity")
        print("   □ logs.txt - Execution logs")
    
    def _show_general_requirements(self) -> None:
        """Show general validation requirements."""
        print("\n📋 General Validation Checklist")
        print("=" * 50)
        print("\n✓ Review MASTER_VALIDATION_CHECKLIST.md")
        print("✓ Identify which stage/domain you're working on")
        print("✓ Check the relevant checklist in state/tasks/validation/checklists/")
        print("✓ Measure all required KPIs manually")
        print("✓ Compare against targets")
        print("✓ Document evidence")
        print("\n⚠️ No automatic validation - all checks must be done manually!")
    
    def show_metrics(self) -> None:
        """Display validation metrics from recent runs."""
        print("\n📊 Recent Validation Evidence")
        print("=" * 50)
        
        # Find recent evidence runs
        evidence_runs = sorted(self.evidence_dir.glob("*/metrics.json"))
        
        if not evidence_runs:
            print("No validation evidence found.")
            print("\n💡 To create evidence:")
            print("   1. Run your validation manually")
            print("   2. Record results in evidence/<run_id>/metrics.json")
            return
        
        print(f"Found {len(evidence_runs)} validation runs\n")
        
        # Show last 5 runs
        for metrics_file in evidence_runs[-5:]:
            run_id = metrics_file.parent.name
            print(f"📁 Run: {run_id}")
            
            try:
                import json
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    
                # Display recorded metrics
                if "kpis" in metrics and metrics["kpis"]:
                    print("   Recorded KPIs:")
                    for kpi, value in metrics["kpis"].items():
                        print(f"   • {kpi}: {value}")
                else:
                    print("   ⚠️ No KPIs recorded (manual validation needed)")
                print()
            except Exception as e:
                print(f"   Error reading metrics: {e}\n")
    
    def manage_rubrics(self, action: str = "list") -> None:
        """Manage validation rubrics - templates for manual validation."""
        print("\n📝 Rubric Management")
        print("=" * 50)
        
        if action == "list":
            rubric_files = list(self.templates_dir.glob("RUBRIC_*.md"))
            print(f"Found {len(rubric_files)} rubric templates:\n")
            for rubric in rubric_files:
                print(f"   → {rubric.name}")
            print("\n💡 Use these as guides for manual validation")
        elif action == "generate":
            print("Generating rubric templates...")
        try:
            from .rubric_manager import RubricManager
            manager = RubricManager(self.validation_root)
            manager.generate_all_rubrics()
            print("\n✅ Rubric templates generated")
            print("📝 Fill these out during manual validation")
        except ImportError:
            print("⚠️ Rubric manager not available")
    
    def generate_dashboard(self) -> None:
        """Generate validation dashboard from manually collected evidence."""
        print("\n📈 Generating Validation Dashboard")
        print("=" * 50)
        
        try:
            from .dashboard_generator import DashboardGenerator
            generator = DashboardGenerator(self.validation_root)
            dashboard_path = generator.generate_html_dashboard()
            print(f"✅ Dashboard generated: {dashboard_path}")
            print("\n📊 This shows your manually recorded validation results")
        except ImportError:
            print("⚠️ Dashboard generator not available")
            print("💡 Record validation results in evidence/<run_id>/metrics.json")
    
    def validate_rules(self) -> None:
        """Check rules configuration - does not validate code."""
        print("\n📜 Checking Rules Configuration")
        print("=" * 50)
        print("ℹ️  This checks rule files only, not code validation\n")
        
        try:
            from .rules_validator import RulesValidator
            validator = RulesValidator(self.workspace_root)
            result = validator.validate_rules()
            
            if result["success"]:
                print(f"✅ {result['message']}")
                print(f"   Rules configured: {result['rules_count']}")
            else:
                print(f"⚠️ {result['message']}")
                for failure in result["failures"]:
                    print(f"   - {failure}")
            
            # Show summary
            summary = validator.get_rules_summary()
            print(f"\n📊 Rules Summary:")
            print(f"   Total rules: {summary['total_rules']}")
            print(f"   Cursor rules: {summary['cursor_rules']}")
            print(f"   Quark rules: {summary['quark_rules']}")
            
        except ImportError as e:
            print(f"⚠️ Rules validator not available: {e}")
    
    def run_validation_checklist(self) -> int:
        """Show validation checklist for CI - does not auto-validate."""
        print("\n📋 CI VALIDATION CHECKLIST")
        print("=" * 50)
        print("⚠️  Manual validation required for all items!\n")
        
        checklist_items = []
        
        # 1. Rules configuration
        print("1️⃣ Rules Configuration")
        rules_ok = self._check_rules_config()
        checklist_items.append(("Rules configured correctly", rules_ok))
        
        # 2. Evidence structure
        print("\n2️⃣ Evidence Requirements")
        evidence_ok = self._check_evidence_structure()
        checklist_items.append(("Evidence structure exists", evidence_ok))
        
        # 3. Show what needs validation
        print("\n3️⃣ Validation Requirements")
        self._show_validation_needs()
        checklist_items.append(("Validation requirements identified", True))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 CHECKLIST SUMMARY")
        all_ok = True
        for item, status in checklist_items:
            status_icon = "✅" if status else "⚠️"
            print(f"{status_icon} {item}")
            if not status:
                all_ok = False
        
        if all_ok:
            print("\n✅ Ready for manual validation")
            print("📝 Now manually validate each KPI against targets")
            return 0
        else:
            print("\n⚠️ Fix configuration issues before validation")
            return 1
    
    def _check_rules_config(self) -> bool:
        """Check if rules are properly configured."""
        try:
            result = subprocess.run(
                [sys.executable, str(self.workspace_root / "tools_utilities/validate_rules_index.py")],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("   ✅ Rules configuration valid")
                return True
            else:
                print("   ⚠️ Rules configuration issues found")
                return False
        except:
            print("   ⚠️ Could not check rules")
            return False
    
    def _check_evidence_structure(self) -> bool:
        """Check if evidence directory structure exists."""
        if self.evidence_dir.exists():
            print("   ✅ Evidence directory exists")
            return True
        else:
            print("   ⚠️ Evidence directory missing")
            return False
    
    def _show_validation_needs(self) -> None:
        """Show what needs to be validated."""
        print("   📋 Check these items manually:")
        print("   □ All KPIs measured against targets")
        print("   □ Evidence documented")
        print("   □ Rubrics filled out")
        print("   □ Calibration metrics recorded")
        print("   □ Reproducibility info saved")