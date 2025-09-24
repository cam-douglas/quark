#!/usr/bin/env python3
"""
Quark Validation System - Main Entry Point
===========================================
Central orchestrator for all validation activities in the Quark project.
Integrates with CI/CD, Quark State System, and provides interactive guidance.

Usage:
    python quark_validate.py [command] [options]
    
Commands:
    sprint      - Interactive sprint validation guide
    validate    - Run validation for current changes
    verify      - Verify specific KPIs/domains
    metrics     - Display validation metrics
    rubric      - Manage validation rubrics
    evidence    - Manage evidence artifacts
    dashboard   - Generate validation dashboards
    rules       - Validate rules index
    ci          - Run full CI validation pipeline
    
Activation Words (for Quark State System):
    validate, validation, verify, verification, KPI, metrics, rubric
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional

# Add validation core modules to path
validation_core_path = Path(__file__).parent.parent.parent / "tasks/validation/core"
sys.path.insert(0, str(validation_core_path))

# Import core validator
try:
    from validator import QuarkValidator
except ImportError as e:
    print(f"⚠️ Core validation modules not found: {e}")
    sys.exit(1)


def show_help():
    """Display comprehensive help information."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          🚀 QUARK VALIDATION SYSTEM - COMMAND REFERENCE          ║
╚══════════════════════════════════════════════════════════════════╝

📝 IMPORTANT: All validation is MANUAL - commands show what needs validation

═══ INTERACTIVE GUIDES ═══════════════════════════════════════════════

  python quark_validate.py sprint
  make validate
    → Interactive sprint guide - walks through validation step-by-step
    
  make validate-sprint  
    → Same as above (alias for consistency)

═══ SHOW VALIDATION REQUIREMENTS ══════════════════════════════════════

  python quark_validate.py validate [--domain DOMAIN] [--stage N]
  make validate-quick [domain words]
    → Show what needs validation for current git changes
    → Optional: specify domain or stage to check specific requirements
    
  python quark_validate.py verify --domain "foundation layer"
  make validate foundation layer tasks
    → Show requirements for specific domain (smart matching enabled)
    → Examples: "foundation", "integration", "stage 1", "embryonic"
    
  python quark_validate.py verify --stage 2
    → Show requirements for biological stage (1-6)

═══ REPORTS & METRICS ═════════════════════════════════════════════════

  python quark_validate.py metrics
  make validate-metrics
    → Display metrics from previous manual validation runs
    
  python quark_validate.py dashboard
  make validate-dashboard
    → Generate HTML dashboard from manual validation results
    → Output: state/tasks/validation/dashboards/validation_dashboard.html

═══ TEMPLATES & RUBRICS ═══════════════════════════════════════════════

  python quark_validate.py rubric [--action generate]
  make validate-rubrics
    → Generate rubric templates for manual validation
    
  python quark_validate.py evidence
    → Manage evidence artifacts and templates

═══ SYSTEM & CI/CD ════════════════════════════════════════════════════

  python quark_validate.py ci
  make validate-ci
    → Show CI validation checklist (returns exit code for automation)
    
  python quark_validate.py rules [--action sync]
  make validate-rules
    → Validate rules configuration
    
  make validate-sync
    → Sync rules between .cursor and .quark directories

═══ OTHER COMMANDS ════════════════════════════════════════════════════

  python quark_validate.py help
  make help
  make validate-help
    → Show this help message
    
  make test
    → Run project tests (pytest)
    
  make lint
    → Run linters (ruff, mypy)
    
  make clean
  make clean-validation
    → Clean generated files (preserves evidence)

═══ QUICK EXAMPLES ════════════════════════════════════════════════════

  # Start interactive validation for current sprint
  make validate
  
  # Check what needs validation for your changes
  make validate-quick
  
  # Check specific domain requirements
  make validate foundation layer
  python quark_validate.py verify --domain "integration tests"
  
  # View previous validation results
  make validate-metrics
  
  # Generate dashboard from manual results
  make validate-dashboard

═══ VALIDATION WORKFLOW ═══════════════════════════════════════════════

  1. Run 'make validate' to start interactive guide
  2. Select your validation scope (stage/domain)
  3. Review requirements and prerequisites
  4. Perform manual validation of KPIs
  5. Record measurements when prompted
  6. Evidence saved to evidence/<timestamp>/
  7. Update checklists manually
  8. Generate dashboard to visualize results

═══ KEY FILES ═════════════════════════════════════════════════════════

  📁 state/tasks/validation/
     ├── VALIDATION_GUIDE.md          - Complete usage documentation
     ├── VALIDATION_MASTER_INDEX.md   - Index of all validation files
     ├── checklists/                  - Domain-specific checklists
     ├── evidence/                    - Your validation results
     ├── templates/                   - Rubric templates
     └── dashboards/                  - Generated visualizations

═══ ACTIVATION WORDS (Quark State System) ════════════════════════════

  validate, validation, verify, verification, KPI, metrics, rubric,
  benchmark, calibration, evidence, checklist, milestone, gate

⚠️ Remember: This system guides you through validation but does NOT
           automatically validate. You must perform all measurements
           and verification manually.
""")


def main():
    """Main entry point for Quark validation system."""
    parser = argparse.ArgumentParser(
        description="Quark Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="help",
        choices=["sprint", "validate", "verify", "metrics", 
                 "rubric", "evidence", "dashboard", "rules", "ci", "help"],
        help="Validation command to execute"
    )
    
    parser.add_argument(
        "--domain",
        help="Specific domain to validate (supports fuzzy matching)"
    )
    
    parser.add_argument(
        "--stage", 
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Biological stage to validate"
    )
    
    parser.add_argument(
        "--run-id",
        help="Specific evidence run ID"
    )
    
    parser.add_argument(
        "--action",
        choices=["list", "generate", "sync"],
        default="list",
        help="Action for rubric/rules commands"
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = QuarkValidator()
    
    # Route commands
    if args.command == "sprint":
        validator.sprint_mode()
        
    elif args.command == "validate":
        # Show validation requirements - no auto-validation
        validator.show_validation_requirements(scope=args.domain, stage=args.stage)
        
    elif args.command == "verify":
        # Show what needs verification - no auto-validation
        if args.domain:
            validator.show_validation_requirements(scope=args.domain)
        elif args.stage:
            validator.show_validation_requirements(stage=args.stage)
        else:
            print("⚠️ verify command requires --domain or --stage")
            print("Examples:")
            print("  python quark_validate.py verify --domain MAIN_INTEGRATIONS")
            print("  python quark_validate.py verify --domain 'foundation layer'")
            print("  python quark_validate.py verify --stage 1")
            print("\n📝 This will show what needs manual validation")
            
    elif args.command == "metrics":
        validator.show_metrics()
        
    elif args.command == "rubric":
        validator.manage_rubrics(action=args.action)
        
    elif args.command == "dashboard":
        validator.generate_dashboard()
        
    elif args.command == "rules":
        validator.validate_rules()
        if args.action == "sync":
            # Also sync rules between cursor and quark
            try:
                from rules_validator import RulesValidator
                rules_val = RulesValidator(Path.cwd())
                result = rules_val.sync_rules()
                if result["synced"]:
                    print("\n✅ Rules synced:")
                    for msg in result["synced"]:
                        print(f"   - {msg}")
                if result["failed"]:
                    print("\n❌ Sync failures:")
                    for msg in result["failed"]:
                        print(f"   - {msg}")
            except ImportError:
                print("⚠️ Rules sync not available")
                
    elif args.command == "ci":
        # Show CI validation checklist - no auto-validation
        exit_code = validator.run_validation_checklist()
        sys.exit(exit_code)
        
    elif args.command == "help":
        show_help()
        
    else:
        print(f"Unknown command: {args.command}")
        show_help()


if __name__ == "__main__":
    # Check for activation words in command line
    activation_words = ["validate", "validation", "verify", 
                       "verification", "KPI", "metrics", "rubric"]
    
    if len(sys.argv) > 1 and any(word in " ".join(sys.argv).lower() 
                                  for word in activation_words):
        print("🔍 Validation context activated")
    
    main()