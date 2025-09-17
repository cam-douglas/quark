"""
Biological Validator - Validates biological constraints for Cline tasks

Provides pre-execution and post-execution validation of biological constraints,
ensuring all autonomous coding tasks comply with biological rules and safety.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from .cline_types import CodingTask, TaskResult
from .brain_context_provider import BrainContextProvider


class BiologicalValidator:
    """
    Validates biological constraints for autonomous coding tasks
    
    Ensures all Cline tasks comply with biological rules, developmental stages,
    and architectural constraints before and after execution.
    """
    
    def __init__(self, workspace_path: Path):
        """Initialize biological validator"""
        self.workspace_path = workspace_path
        self.brain_context_provider = BrainContextProvider(workspace_path)
        self.logger = logging.getLogger(__name__)
    
    async def validate_biological_compliance(self, task: CodingTask) -> bool:
        """Validate task against biological constraints"""
        try:
            # Check for biological constraint violations
            violations = []
            
            # Check for prohibited patterns
            prohibited_patterns = [
                "negative_emotion", "harmful_pattern", "biological_harm",
                "neural_damage", "toxic_behavior"
            ]
            
            for pattern in prohibited_patterns:
                if pattern in task.description.lower():
                    violations.append(f"Prohibited pattern: {pattern}")
            
            # Check developmental stage compliance
            brain_context = await self.brain_context_provider.get_brain_context()
            current_stage = brain_context.biological_constraints.get("developmental_stage")
            
            if current_stage == "neural_tube_closure" and "synaptogenesis" in task.description.lower():
                violations.append("Developmental stage mismatch: synaptogenesis not allowed during neural tube closure")
            
            if violations:
                self.logger.warning(f"Biological compliance violations: {violations}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Biological compliance validation failed: {e}")
            return False

    async def validate_post_execution_compliance(self, result: TaskResult) -> bool:
        """Validate biological compliance after task execution"""
        try:
            # Check modified files for compliance
            for file_path in result.files_modified:
                if file_path.startswith("brain/"):
                    # Validate brain module files
                    file_content = Path(file_path).read_text() if Path(file_path).exists() else ""
                    
                    # Check file length (architecture rule: <300 lines)
                    if len(file_content.splitlines()) >= 300:
                        self.logger.warning(f"File {file_path} exceeds 300 line limit")
                        return False
                    
                    # Check for prohibited content
                    if any(term in file_content.lower() for term in ["negative_emotion", "harmful"]):
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Post-execution compliance validation failed: {e}")
            return False

    def assess_task_complexity(self, description: str) -> str:
        """Assess task complexity based on description"""
        description_lower = description.lower()
        
        if any(term in description_lower for term in ["architecture", "brain", "neural", "morphogen"]):
            return "CRITICAL"
        elif any(term in description_lower for term in ["refactor", "multiple files", "system"]):
            return "COMPLEX"
        elif any(term in description_lower for term in ["edit", "modify", "update"]):
            return "MODERATE"
        else:
            return "SIMPLE"

    def get_biological_safety_rules(self) -> List[str]:
        """Get list of biological safety rules"""
        return [
            "No negative emotions in brain modules",
            "Respect developmental stage constraints",
            "Validate against AlphaGenome biological rules",
            "Maintain neuroanatomical naming conventions",
            "File size limit: 300 lines maximum",
            "No harmful or toxic behavioral patterns",
            "Maintain biological plausibility",
            "Follow focused responsibility patterns"
        ]

    def check_file_compliance(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Check individual file compliance with biological constraints
        
        Args:
            file_path: Path to the file
            content: File content to check
            
        Returns:
            Dictionary with compliance results
        """
        compliance_result = {
            "compliant": True,
            "violations": [],
            "warnings": []
        }
        
        # Check file size
        line_count = len(content.splitlines())
        if line_count >= 300:
            compliance_result["compliant"] = False
            compliance_result["violations"].append(f"File exceeds 300 line limit: {line_count} lines")
        elif line_count >= 250:
            compliance_result["warnings"].append(f"File approaching 300 line limit: {line_count} lines")
        
        # Check for prohibited patterns
        prohibited_terms = ["negative_emotion", "harmful_behavior", "toxic_pattern"]
        for term in prohibited_terms:
            if term in content.lower():
                compliance_result["compliant"] = False
                compliance_result["violations"].append(f"Prohibited term found: {term}")
        
        # Check neuroanatomical naming for brain files
        if "brain/" in file_path:
            # Should follow neuroanatomical conventions
            if not any(term in file_path.lower() for term in [
                "neural", "brain", "morphogen", "gradient", "hippocampus", 
                "cortex", "limbic", "cognitive", "memory"
            ]):
                compliance_result["warnings"].append("File may not follow neuroanatomical naming conventions")
        
        return compliance_result
