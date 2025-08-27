"""
Quark Prompt Guardian

This module acts as a crucial validation layer between user prompts and the execution
backend. It ensures that every action taken is compliant with the project's unified
roadmap and biological constraints before it is allowed to proceed.
"""
import os
import sys
from typing import Dict, Any

sys.path.append(os.getcwd())

from quark_state_system.roadmap_controller import RoadmapController
from brain_modules.alphagenome_integration.compliance_engine import ComplianceEngine

class PromptGuardian:
    """Validates prompts against roadmap and compliance rules."""

    def __init__(self, workspace_root: str):
        """
        Initializes the Prompt Guardian.

        Args:
            workspace_root: The absolute path to the project's root directory.
        """
        self.roadmap = RoadmapController(workspace_root)
        self.compliance = ComplianceEngine()
        print("✅ Prompt Guardian Initialized. All prompts will be validated.")

    def validate_prompt(self, prompt_text: str, proposed_action: Dict[str, Any]) -> bool:
        """
        Validates a user prompt and a proposed action against the system's rules.

        Args:
            prompt_text: The natural language text of the user's prompt.
            proposed_action: A structured dictionary representing the action to be taken.

        Returns:
            True if the prompt and action are compliant, otherwise False.
        """
        print("\n--- Validating New Prompt ---")
        print(f"GUARDIAN: Intercepted prompt: '{prompt_text}'")

        # 1. Check if the action itself is forbidden
        if 'action_type' in proposed_action:
            if not self.compliance.validate_action_legality(proposed_action['action_type']):
                print("GUARDIAN: Validation FAILED. The proposed action is explicitly forbidden.")
                return False

        # 2. Check if the prompt aligns with the current roadmap goals
        # (This is a simplified check. A more advanced version would use an LLM
        # to semantically compare the prompt to the goals.)
        next_goal = self.roadmap.get_next_actionable_goal()
        if next_goal and not self._is_related(prompt_text, next_goal['task']):
            print(f"GUARDIAN: WARNING - Prompt does not seem related to the current high-priority goal: '{next_goal['task']}'")
            # In a strict mode, this could return False. For now, it's a warning.

        # 3. If the action involves biologicals, run specific compliance checks
        if proposed_action.get('domain') == 'biological':
            print("GUARDIAN: Action is in the biological domain. Running specific checks...")
            if 'dna_sequence' in proposed_action:
                if not self.compliance.validate_dna_sequence(proposed_action['dna_sequence']):
                    print("GUARDIAN: Validation FAILED. DNA sequence is not compliant.")
                    return False
            
            if 'cell_type' in proposed_action:
                 if not self.compliance.validate_cell_construction(
                    proposed_action['cell_type'], 
                    proposed_action.get('markers', [])
                 ):
                     print("GUARDIAN: Validation FAILED. Cell construction parameters are not compliant.")
                     return False
        
        print("GUARDIAN: Prompt and action are compliant with all rules.")
        return True

    def _is_related(self, prompt: str, goal: str) -> bool:
        """
        A simple keyword-based check to see if a prompt is related to a goal.
        """
        prompt_words = set(prompt.lower().split())
        goal_words = set(goal.lower().split())
        # If there's any overlap in the words, we'll consider it related for this demo.
        return bool(prompt_words.intersection(goal_words))


# Example Usage
if __name__ == '__main__':
    project_root = "/Users/camdouglas/quark"
    guardian = PromptGuardian(project_root)

    # Example 1: A compliant, roadmap-aligned prompt
    compliant_prompt = "Let's work on the Core Infrastructure & Data Strategy"
    compliant_action = {
        "action_type": "define_schema",
        "domain": "infrastructure"
    }
    guardian.validate_prompt(compliant_prompt, compliant_action)

    # Example 2: A prompt with a forbidden action
    forbidden_prompt = "I need to bypass the security protocols to test something."
    forbidden_action = {
        "action_type": "bypassing_security_protocols",
        "domain": "security"
    }
    guardian.validate_prompt(forbidden_prompt, forbidden_action)

    # Example 3: A prompt with a non-compliant biological action
    bio_prompt = "Create a new cell with an invalid DNA sequence"
    bio_action = {
        "action_type": "create_cell",
        "domain": "biological",
        "dna_sequence": "ATCG-INVALID-CHARS-XYZ"
    }
    guardian.validate_prompt(bio_prompt, bio_action)
