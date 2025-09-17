"""
PromptGuardian - Validates prompts against biological, safety, and repo rules.

Integration: Indirect integration via QuarkDriver and AutonomousAgent.
Rationale: Enforces compliance before actions are executed.
"""

import os
import re
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import json

# Import biological constraints if available
try:
    from management.rules.biological_constraints import (
        REQUIRED_BIOLOGICAL_MARKERS,
        PROHIBITED_ACTIONS,
        SIMULATION_SAFETY_BOUNDARIES,
        VALID_DNA_BASES,
        DNA_SEQUENCE_CONSTRAINTS
    )
    RULES_AVAILABLE = True
except ImportError:
    RULES_AVAILABLE = False
    # Fallback minimal rules
    PROHIBITED_ACTIONS = [
        "self_modification_of_security_rules",
        "bypassing_security_protocols",
        "disabling_audit_logging"
    ]

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of prompt validation."""
    is_valid: bool
    reason: str
    severity: str  # "error", "warning", "info"
    suggested_fix: Optional[str] = None


class PromptGuardian:
    """Validates prompts and actions against safety and compliance rules."""

    def __init__(self, workspace_root: str = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.initialized = True
        self.validation_history = []
        self.blocked_actions = 0
        self.approved_actions = 0

        # Load additional rules from files if available
        self._load_custom_rules()

        logger.info(f"🛡️ PromptGuardian initialized (workspace: {self.workspace_root})")
        if not RULES_AVAILABLE:
            logger.warning("⚠️ Biological constraints not available - using minimal ruleset")

    def _load_custom_rules(self):
        """Load any custom rules from configuration files."""
        # Check for custom rules in workspace
        custom_rules_path = os.path.join(self.workspace_root, ".quark_rules.json")
        if os.path.exists(custom_rules_path):
            try:
                with open(custom_rules_path, 'r') as f:
                    custom_rules = json.load(f)
                    self.custom_prohibited = custom_rules.get("prohibited_actions", [])
                    self.custom_patterns = custom_rules.get("dangerous_patterns", [])
                    logger.info(f"Loaded {len(self.custom_prohibited)} custom prohibited actions")
            except Exception as e:
                logger.error(f"Failed to load custom rules: {e}")
                self.custom_prohibited = []
                self.custom_patterns = []
        else:
            self.custom_prohibited = []
            self.custom_patterns = []

    def validate_action(self, action: str, context: Dict[str, Any] = None) -> bool:
        """Validate a proposed action against all rules.

        Args:
            action: The action string to validate
            context: Additional context (action_type, target_files, etc.)
            
        Returns:
            bool: True if action is allowed, False if blocked
        """
        context = context or {}

        # Run all validation checks
        results = [
            self._check_prohibited_actions(action, context),
            self._check_dangerous_patterns(action, context),
            self._check_file_safety(action, context),
            self._check_biological_compliance(action, context),
            self._check_simulation_boundaries(action, context)
        ]

        # Log results
        for result in results:
            if not result.is_valid:
                logger.warning(f"❌ Validation failed: {result.reason}")
                if result.suggested_fix:
                    logger.info(f"💡 Suggestion: {result.suggested_fix}")

        # Update statistics
        is_valid = all(r.is_valid for r in results)
        if is_valid:
            self.approved_actions += 1
        else:
            self.blocked_actions += 1

        # Record in history
        self.validation_history.append({
            "action": action[:100] + "..." if len(action) > 100 else action,
            "context": context.get("action_type", "unknown"),
            "valid": is_valid,
            "timestamp": logger.handlers[0].formatter.formatTime(logger.makeRecord(
                "test", 0, "", 0, "", (), None)) if logger.handlers else "N/A"
        })

        return is_valid

    def _check_prohibited_actions(self, action: str, context: Dict) -> ValidationResult:
        """Check if action contains prohibited operations."""
        action_lower = action.lower()

        # Check against known prohibited actions
        all_prohibited = PROHIBITED_ACTIONS + self.custom_prohibited

        for prohibited in all_prohibited:
            if prohibited.lower() in action_lower:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Action contains prohibited operation: '{prohibited}'",
                    severity="error",
                    suggested_fix="Remove or rephrase the prohibited operation"
                )

        return ValidationResult(is_valid=True, reason="No prohibited actions detected", severity="info")

    def _check_dangerous_patterns(self, action: str, context: Dict) -> ValidationResult:
        """Check for dangerous code patterns."""
        dangerous_patterns = [
            r"rm\s+-rf\s+/",  # Dangerous deletion
            r"sudo\s+rm",  # Sudo deletion
            r"eval\s*\(",  # Eval usage
            r"exec\s*\(",  # Exec usage
            r"__import__\s*\(",  # Dynamic import
            r"\.\.\/\.\.\/\.\.",  # Path traversal
            r"0\.0\.0\.0",  # Bind to all interfaces
        ] + self.custom_patterns

        for pattern in dangerous_patterns:
            if re.search(pattern, action, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    reason=f"Dangerous pattern detected: '{pattern}'",
                    severity="error",
                    suggested_fix="Use safer alternatives or add explicit safety checks"
                )

        return ValidationResult(is_valid=True, reason="No dangerous patterns detected", severity="info")

    def _check_file_safety(self, action: str, context: Dict) -> ValidationResult:
        """Check file operations for safety."""
        # Extract target files from context
        target_files = context.get("target_files", [])

        # Check for critical file modifications
        critical_paths = [
            "management/rules/",
            "brain/architecture/safety/",
            ".git/",
            "QUARK_STATE_SYSTEM.py",
            "prompt_guardian.py",
            "safety_guardian.py",
            ".cursor/rules/",
            ".quark/rules/",
        ]

        for file_path in target_files:
            for critical in critical_paths:
                if critical in file_path:
                    # Allow safe reads/searches but block edits unless explicitly confirmed
                    if context.get("action_type") in ["read", "search", "list"]:
                        continue
                    # For edits on rule files require explicit user confirmation flag
                    if critical.endswith("rules/") and not context.get("user_confirmed", False):
                        return ValidationResult(False, f"Attempt to modify protected rules file '{file_path}' without explicit user approval", "error")

        return ValidationResult(is_valid=True, reason="File operations are safe", severity="info")

    def _check_biological_compliance(self, action: str, context: Dict) -> ValidationResult:
        """Check biological and AlphaGenome compliance."""
        if not RULES_AVAILABLE:
            return ValidationResult(is_valid=True, reason="Biological rules not loaded", severity="warning")

        # Check DNA sequence validity if present
        dna_pattern = r"[ATCG]{10,}"
        dna_matches = re.findall(dna_pattern, action.upper())

        for sequence in dna_matches:
            # Check if sequence contains only valid bases
            if not all(base in VALID_DNA_BASES for base in sequence):
                return ValidationResult(
                    is_valid=False,
                    reason="Invalid DNA sequence detected",
                    severity="error",
                    suggested_fix="DNA sequences must contain only A, T, C, G bases"
                )

            # Check sequence length constraints
            if len(sequence) > DNA_SEQUENCE_CONSTRAINTS["max_length"]:
                return ValidationResult(
                    is_valid=False,
                    reason=f"DNA sequence exceeds maximum length ({DNA_SEQUENCE_CONSTRAINTS['max_length']})",
                    severity="error",
                    suggested_fix="Split long sequences or use streaming approach"
                )

        return ValidationResult(is_valid=True, reason="Biological compliance verified", severity="info")

    def _check_simulation_boundaries(self, action: str, context: Dict) -> ValidationResult:
        """Check simulation safety boundaries."""
        if not RULES_AVAILABLE:
            return ValidationResult(is_valid=True, reason="Simulation rules not loaded", severity="warning")

        # Check for simulation parameters in action
        sim_params = context.get("simulation_params", {})

        # Validate simulation time
        if "simulation_time" in sim_params:
            max_time = SIMULATION_SAFETY_BOUNDARIES["max_simulation_time_hours"]
            if sim_params["simulation_time"] > max_time:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Simulation time exceeds safety limit ({max_time} hours)",
                    severity="error",
                    suggested_fix=f"Reduce simulation time to <= {max_time} hours"
                )

        # Validate cell population
        if "cell_count" in sim_params:
            max_cells = SIMULATION_SAFETY_BOUNDARIES["max_cell_population"]
            if sim_params["cell_count"] > max_cells:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Cell population exceeds safety limit ({max_cells:,})",
                    severity="error",
                    suggested_fix=f"Reduce cell count to <= {max_cells:,}"
                )

        return ValidationResult(is_valid=True, reason="Simulation boundaries respected", severity="info")

    def check_compliance(self, prompt: str) -> bool:
        """Quick compliance check for prompts.

        Args:
            prompt: The prompt text to check

        Returns:
            bool: True if compliant, False otherwise
        """
        # Quick check for obvious violations
        prompt_lower = prompt.lower()

        # Check for requests to disable safety
        unsafe_requests = [
            "disable safety",
            "turn off guardian",
            "bypass security",
            "ignore rules",
            "skip validation"
        ]

        for unsafe in unsafe_requests:
            if unsafe in prompt_lower:
                logger.warning(f"🚫 Unsafe request detected: '{unsafe}'")
                return False

        return True

    def is_safe(self, operation: str) -> bool:
        """Quick safety check for operations.
        
        Args:
            operation: The operation to check
            
        Returns:
            bool: True if safe, False if dangerous
        """
        # Use validate_action with minimal context
        return self.validate_action(operation, {"action_type": "operation"})

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.approved_actions + self.blocked_actions

        return {
            "total_validations": total,
            "approved": self.approved_actions,
            "blocked": self.blocked_actions,
            "approval_rate": self.approved_actions / total if total > 0 else 0,
            "recent_history": self.validation_history[-10:]  # Last 10 validations
        }

    def reset_statistics(self):
        """Reset validation statistics."""
        self.validation_history = []
        self.blocked_actions = 0
        self.approved_actions = 0
        logger.info("PromptGuardian statistics reset")


# Example usage
if __name__ == "__main__":
    guardian = PromptGuardian()

    # Test various actions
    test_actions = [
        ("Update the brain simulation parameters", {"action_type": "update"}),
        ("Delete safety_guardian.py", {"action_type": "delete", "target_files": ["safety_guardian.py"]}),
        ("Run simulation with ATCGATCG sequence", {"action_type": "simulate"}),
        ("Disable audit logging", {"action_type": "config"}),
    ]

    for action, context in test_actions:
        result = guardian.validate_action(action, context)
        print(f"\nAction: {action}")
        print(f"Valid: {result}")

    print(f"\nStatistics: {guardian.get_statistics()}")
