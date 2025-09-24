#!/usr/bin/env python3
"""
Validation Integration for Quark State System
==============================================
Integrates validation system with Quark's autonomous agent and state management.
Activates on validation-related keywords and provides automated validation workflows.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add validation system to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import validation components
from state.tasks.validation.core.sprint_guide import SprintGuide
from state.tasks.validation.core.scope_selector import ScopeSelector
from state.tasks.validation.core.kpi_runner import KPIRunner
from state.tasks.validation.core.evidence_collector import EvidenceCollector
from state.tasks.validation.core.checklist_parser import ChecklistParser


class ValidationIntegration:
    """Integrates validation with Quark State System."""
    
    # Activation keywords that trigger validation context
    ACTIVATION_WORDS = [
        "validate", "validation", "verify", "verification",
        "kpi", "metrics", "rubric", "benchmark", "calibration",
        "evidence", "checklist", "milestone", "gate"
    ]
    
    def __init__(self, workspace_root: str):
        """Initialize validation integration."""
        self.workspace_root = Path(workspace_root)
        self.validation_root = self.workspace_root / "state/tasks/validation"
        
        # Initialize core components
        self.sprint_guide = SprintGuide(self.validation_root)
        self.scope_selector = ScopeSelector(self.validation_root)
        self.kpi_runner = KPIRunner(self.validation_root)
        self.evidence_collector = EvidenceCollector(self.validation_root)
        self.checklist_parser = ChecklistParser(self.validation_root)
        
        self.active_validation = None
        
    def should_activate(self, prompt_text: str) -> bool:
        """Check if validation context should be activated."""
        prompt_lower = prompt_text.lower()
        return any(word in prompt_lower for word in self.ACTIVATION_WORDS)
    
    def process_validation_prompt(self, prompt_text: str) -> Dict[str, Any]:
        """Process a validation-related prompt."""
        prompt_lower = prompt_text.lower()
        
        # Determine validation action
        if "sprint" in prompt_lower or "guide" in prompt_lower:
            return self._run_sprint_validation()
        
        elif "quick" in prompt_lower or "current" in prompt_lower:
            return self._run_quick_validation()
        
        elif "kpi" in prompt_lower or "metric" in prompt_lower:
            return self._run_kpi_measurement()
        
        elif "evidence" in prompt_lower:
            return self._collect_evidence()
        
        elif "dashboard" in prompt_lower:
            return self._generate_dashboard()
        
        elif "checklist" in prompt_lower:
            return self._process_checklist()
        
        elif "gate" in prompt_lower or "ci" in prompt_lower:
            return self._run_validation_gate()
        
        else:
            # Default: suggest validation scope
            return self._suggest_validation()
    
    def _run_sprint_validation(self) -> Dict[str, Any]:
        """Run interactive sprint validation."""
        print("\n🚀 VALIDATION: Starting Sprint Validation Workflow")
        
        # Use quick validate for non-interactive context
        result = self.sprint_guide.quick_validate()
        
        return {
            "status": "completed",
            "action": "sprint_validation",
            "run_id": result["run_id"],
            "scope": result["scope"],
            "gate_passed": result["gate_passed"],
            "message": f"Sprint validation {'passed' if result['gate_passed'] else 'failed'}"
        }
    
    def _run_quick_validation(self) -> Dict[str, Any]:
        """Quick validation of current changes."""
        print("\n⚡ VALIDATION: Quick validation of current changes")
        
        # Detect scope from git diff
        suggestions = self.scope_selector.suggest_from_git_diff()
        scope = suggestions.pop() if suggestions else "MAIN_INTEGRATIONS_CHECKLIST"
        
        # Run validation
        result = self.sprint_guide.quick_validate(scope)
        
        return {
            "status": "completed",
            "action": "quick_validation",
            "scope": scope,
            "run_id": result["run_id"],
            "gate_passed": result["gate_passed"],
            "suggestions": list(suggestions) if suggestions else []
        }
    
    def _run_kpi_measurement(self) -> Dict[str, Any]:
        """Run KPI measurements."""
        print("\n📊 VALIDATION: Running KPI measurements")
        
        # Get current scope
        scope = self.active_validation or "MAIN_INTEGRATIONS_CHECKLIST"
        
        # Run measurements
        results = self.kpi_runner.run_all_kpis(scope)
        
        # Create evidence
        run_id = self.evidence_collector.create_run_id()
        run_dir = self.evidence_collector.setup_evidence_directory(run_id)
        self.evidence_collector.collect_metrics(run_dir, results)
        
        return {
            "status": "completed",
            "action": "kpi_measurement",
            "scope": scope,
            "run_id": run_id,
            "kpis": results.get("kpis", {}),
            "evidence_path": str(run_dir)
        }
    
    def _collect_evidence(self) -> Dict[str, Any]:
        """Collect validation evidence."""
        print("\n📁 VALIDATION: Collecting evidence artifacts")
        
        run_id = self.evidence_collector.create_run_id()
        run_dir = self.evidence_collector.setup_evidence_directory(run_id)
        
        # Collect all evidence types
        self.evidence_collector.collect_config(run_dir)
        self.evidence_collector.collect_seeds(run_dir)
        self.evidence_collector.collect_environment(run_dir)
        
        # Log collection
        log_content = f"Evidence collection\\nRun ID: {run_id}\\nTimestamp: {datetime.now()}"
        self.evidence_collector.collect_logs(run_dir, log_content)
        
        # Validate completeness
        completeness = self.evidence_collector.validate_evidence_completeness(run_dir)
        
        return {
            "status": "completed",
            "action": "evidence_collection",
            "run_id": run_id,
            "evidence_path": str(run_dir),
            "completeness": completeness,
            "all_complete": all(completeness.values())
        }
    
    def _generate_dashboard(self) -> Dict[str, Any]:
        """Generate validation dashboard."""
        print("\n📈 VALIDATION: Generating dashboard")
        
        # Import dashboard generator
        from state.tasks.validation.core.dashboard_generator import DashboardGenerator
        dashboard_gen = DashboardGenerator(self.validation_root)
        
        # Generate HTML dashboard
        dashboard_path = dashboard_gen.generate_html_dashboard()
        
        # Generate Grafana config
        grafana_path = dashboard_gen.generate_grafana_config()
        
        # Get trend analysis
        trends = dashboard_gen.generate_trend_analysis()
        
        return {
            "status": "completed",
            "action": "dashboard_generation",
            "html_dashboard": str(dashboard_path),
            "grafana_config": str(grafana_path),
            "trends": trends
        }
    
    def _process_checklist(self) -> Dict[str, Any]:
        """Process validation checklists."""
        print("\n📋 VALIDATION: Processing checklists")
        
        # Generate completion report
        report = self.checklist_parser.generate_completion_report()
        
        # Validate links
        invalid_links = self.checklist_parser.validate_links()
        
        return {
            "status": "completed",
            "action": "checklist_processing",
            "completion_report": report,
            "invalid_links": len([l for l in invalid_links if not l[2]]),
            "message": "Checklist processing complete"
        }
    
    def _run_validation_gate(self) -> Dict[str, Any]:
        """Run validation gate for CI."""
        print("\n🚦 VALIDATION: Running validation gate")
        
        gate_script = self.workspace_root / "tools_utilities/validation_gate.py"
        
        if not gate_script.exists():
            return {
                "status": "error",
                "action": "validation_gate",
                "message": "Validation gate script not found"
            }
        
        # Run gate script
        result = subprocess.run(
            [sys.executable, str(gate_script)],
            capture_output=True,
            text=True
        )
        
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "action": "validation_gate",
            "gate_passed": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr if result.returncode != 0 else None
        }
    
    def _suggest_validation(self) -> Dict[str, Any]:
        """Suggest validation actions based on current state."""
        print("\n💡 VALIDATION: Suggesting validation actions")
        
        # Get git diff suggestions
        suggestions = self.scope_selector.suggest_from_git_diff()
        
        # Check for incomplete validations
        incomplete = []
        for checklist_file in (self.validation_root / "checklists").glob("*.md"):
            parsed = self.checklist_parser.parse_checklist(checklist_file)
            if "error" not in parsed:
                status = parsed["completion_status"]
                if status["gates_percentage"] < 100:
                    incomplete.append(checklist_file.stem)
        
        return {
            "status": "suggestion",
            "action": "validation_suggestion",
            "suggested_scopes": list(suggestions),
            "incomplete_checklists": incomplete,
            "recommended_action": "sprint" if incomplete else "quick"
        }
    
    def integrate_with_agent(self, agent) -> None:
        """Integrate validation with autonomous agent."""
        # Add validation as a compliance check
        if hasattr(agent, 'compliance'):
            original_validate = agent.compliance.validate_action_legality
            
            def enhanced_validate(action_type: str) -> bool:
                # Check if action needs validation
                if any(word in action_type.lower() for word in self.ACTIVATION_WORDS):
                    print("🔍 Validation context detected in action")
                    
                    # Run quick validation
                    result = self._run_quick_validation()
                    
                    if not result.get("gate_passed", False):
                        print("⚠️ Validation gate failed - action blocked")
                        return False
                
                # Continue with original validation
                return original_validate(action_type)
            
            agent.compliance.validate_action_legality = enhanced_validate
            print("✅ Validation integrated with agent compliance")
    
    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation status."""
        # Find most recent run
        evidence_runs = sorted(self.validation_root.glob("evidence/*/metrics.json"))
        
        if not evidence_runs:
            return {
                "status": "no_runs",
                "message": "No validation runs found"
            }
        
        latest_metrics = evidence_runs[-1]
        run_id = latest_metrics.parent.name
        
        # Load metrics
        import json
        with open(latest_metrics) as f:
            metrics = json.load(f)
        
        # Count passed KPIs
        passed = 0
        total = 0
        if "kpis" in metrics:
            for kpi_data in metrics["kpis"].values():
                if isinstance(kpi_data, dict):
                    total += 1
                    if kpi_data.get("status") == "success":
                        passed += 1
        
        return {
            "status": "ready",
            "latest_run": run_id,
            "kpis_passed": f"{passed}/{total}",
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "timestamp": metrics.get("timestamp", "unknown")
        }


# Hook for Quark Driver integration
def enhance_quark_driver(driver):
    """Enhance QuarkDriver with validation capabilities."""
    validation = ValidationIntegration(driver.workspace_root)
    
    # Store original process_prompt
    original_process = driver.process_prompt
    
    def enhanced_process(prompt_text: str):
        """Enhanced prompt processing with validation."""
        # Check for validation activation
        if validation.should_activate(prompt_text):
            print("\n🔍 VALIDATION CONTEXT ACTIVATED")
            result = validation.process_validation_prompt(prompt_text)
            
            print(f"\nValidation Result: {result.get('status', 'unknown')}")
            if "message" in result:
                print(f"Message: {result['message']}")
            
            # Continue with original processing if validation passed
            if result.get("gate_passed", True):
                return original_process(prompt_text)
            else:
                print("⚠️ Validation failed - blocking further execution")
                return
        
        # Normal processing
        return original_process(prompt_text)
    
    # Replace process_prompt
    driver.process_prompt = enhanced_process
    
    # Integrate with agent
    if hasattr(driver, 'agent'):
        validation.integrate_with_agent(driver.agent)
    
    print("✅ Validation system integrated with QuarkDriver")
    return driver


if __name__ == "__main__":
    # Test validation integration
    workspace = "/Users/camdouglas/quark"
    validation = ValidationIntegration(workspace)
    
    # Test activation detection
    test_prompts = [
        "validate the current changes",
        "show me validation metrics",
        "run KPI measurements",
        "generate dashboard",
        "check validation gate",
        "proceed with next task"  # Should not activate
    ]
    
    print("Testing validation activation:")
    for prompt in test_prompts:
        activated = validation.should_activate(prompt)
        print(f"  '{prompt}' -> {'✅ Activated' if activated else '❌ Not activated'}")
    
    # Test validation status
    print("\nCurrent validation status:")
    status = validation.get_validation_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
