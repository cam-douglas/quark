"""
Quark Active Driver System

This is the main active driver for the Quark system. When activated, it ensures
that all operations are driven by the Autonomous Agent and validated by the
Prompt Guardian.

This represents the final step in making Quark a truly self-determined,
compliant, and goal-driven AGI development system.
"""
import os
import sys
from typing import Dict, Any

sys.path.append(os.getcwd())

from quark_state_system.autonomous_agent import AutonomousAgent
from quark_state_system.prompt_guardian import PromptGuardian

class QuarkDriver:
    """The active driver that orchestrates the agent and guardian."""

    def __init__(self, workspace_root: str):
        """Initializes the driver and its core components."""
        print("🚀 QUARK ACTIVE DRIVER: System is now in self-determined execution mode.")
        self.workspace_root = workspace_root
        # The agent is initialized once to understand the full roadmap context.
        self.agent = AutonomousAgent(workspace_root)
        # The guardian is part of the agent's ecosystem, but we instantiate it
        # here to represent the prompt-validation logic.
        self.guardian = self.agent.compliance # Re-use the compliance engine
        self.current_goal = self.agent.roadmap.get_next_actionable_goal()

    def process_prompt(self, prompt_text: str):
        """
        Processes a user prompt through the full validation and execution pipeline.
        This is the new main loop for any interaction.
        """
        print("\n" + "="*60)
        print(f"DRIVER: Intercepted new prompt: '{prompt_text}'")
        
        if not self.current_goal:
            print("DRIVER: All roadmap goals are complete. System is idle.")
            return

        print(f"DRIVER: Current high-priority goal is: '{self.current_goal['task']}'")

        # If the prompt is generic, default to the agent's plan.
        if prompt_text.lower() in ["proceed", "continue", "next", "evolve"]:
            print("DRIVER: Generic prompt detected. Activating autonomous agent for next goal.")
            self.agent.execute_next_goal()
        else:
            # If the prompt is specific, it must be validated.
            print("DRIVER: Specific prompt detected. Engaging Prompt Guardian for validation.")
            # In a real scenario, an LLM would generate this action from the prompt.
            proposed_action = self._generate_action_from_prompt(prompt_text)
            
            # The guardian uses the agent's compliance engine.
            is_compliant = self.guardian.validate_action_legality(proposed_action['action_type'])

            if is_compliant:
                 print("DRIVER: Action is compliant. Proceeding with execution.")
                 # Handing off to the agent to execute its own plan for the validated goal
                 self.agent.execute_next_goal()
            else:
                 print("DRIVER: Action is NOT compliant. Execution halted.")
                 print("DRIVER: Please provide a prompt that aligns with the current roadmap goal.")
        
        # Refresh the current goal for the next cycle
        self.current_goal = self.agent.roadmap.get_next_actionable_goal()
        print("="*60)

    def run_continuous(self):
        """
        Runs the agent in a continuous loop until all roadmap goals are completed.
        """
        print("🚀🚀 QUARK CONTINUOUS AUTOMATION MODE ACTIVATED 🚀🚀")
        print("DRIVER: The agent will now execute all roadmap goals sequentially.")
        print("DRIVER: To stop, press CTRL+C.")
        print("="*60)

        while self.current_goal:
            self.process_prompt("proceed")
        
        print("\n🎉🎉 AUTOMATION COMPLETE: All roadmap goals have been executed. 🎉🎉")

    def _generate_action_from_prompt(self, prompt_text: str) -> Dict[str, Any]:
        """Placeholder for an LLM that would parse a prompt into a structured action."""
        # Simple keyword matching for this demonstration
        if "security" in prompt_text or "protocol" in prompt_text:
            return {"action_type": "bypassing_security_protocols"}
        return {"action_type": "running_authorized_simulation"}


if __name__ == '__main__':
    project_root = "/Users/camdouglas/quark"
    driver = QuarkDriver(project_root)
    
    # --- SIMULATION OF USER INTERACTION ---
    
    # 1. User gives a generic prompt
    driver.process_prompt("proceed")
    
    # 2. User gives a specific but non-compliant prompt
    driver.process_prompt("We need to bypass the security protocols now.")

    # 3. User gives another generic prompt, agent proceeds with the next task
    driver.process_prompt("continue")
