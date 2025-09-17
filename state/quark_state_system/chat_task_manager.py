#!/usr/bin/env python3
"""Chat Task Manager - Handles ad-hoc task management from chat context.

This module provides functionality to:
1. Parse chat context for outstanding tasks
2. Generate/update chat_tasks.yaml file when explicitly requested
3. Maintain separation between ad-hoc and roadmap tasks

Integration: Used by QuarkDriver and AutonomousAgent for chat-based task tracking.
Rationale: Enforces the protocol that chat_tasks.yaml is only updated on explicit request.
"""

import yaml
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


class ChatTaskManager:
    """Manages ad-hoc tasks from chat context with explicit update control."""

    def __init__(self):
        """Initialize with proper task directory path."""
        self.tasks_dir = Path("/Users/camdouglas/quark/state/quark_state_system/tasks")
        self.chat_tasks_file = self.tasks_dir / "chat_tasks.yaml"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def should_ask_task_type(self, user_query: str) -> bool:
        """
        Determine if we should ask 'ad-hoc or roadmap?' for task requests.
        
        Args:
            user_query: The user's query text
            
        Returns:
            True if query is asking for tasks (not update), False otherwise
        """
        task_keywords = ["tasks", "what should i do", "next actions", "todo"]
        update_keywords = ["update chat tasks", "update the chat tasks"]

        query_lower = user_query.lower()

        # If explicitly asking to update, don't ask type
        if any(keyword in query_lower for keyword in update_keywords):
            return False

        # If asking for tasks, ask type
        return any(keyword in query_lower for keyword in task_keywords)

    def should_auto_update_tasks(self, user_query: str) -> bool:
        """
        Determine if we should automatically update chat_tasks.yaml.
        
        Args:
            user_query: The user's query text
            
        Returns:
            True if query explicitly requests chat tasks update, False otherwise
        """
        update_keywords = [
            "update chat tasks",
            "update the chat tasks",
            "update chat_tasks",
            "generate chat tasks",
            "refresh chat tasks"
        ]

        query_lower = user_query.lower()
        return any(keyword in query_lower for keyword in update_keywords)

    def extract_tasks_from_context(self, chat_context: str) -> List[Dict[str, Any]]:
        """
        Extract outstanding tasks from chat context.
        
        Args:
            chat_context: The current chat conversation context
            
        Returns:
            List of task dictionaries with title, description, status, etc.
        """
        # This is a simplified implementation - in practice you'd parse
        # the actual chat context more intelligently
        tasks = []

        # For now, return example structure - this would be enhanced to
        # actually parse chat context for TODO items, action items, etc.
        current_time = datetime.now().isoformat()

        # Example task extraction logic would go here
        # This is a placeholder that would be enhanced with actual parsing

        return tasks

    def update_chat_tasks_file(self, tasks: List[Dict[str, Any]] = None) -> str:
        """
        Update the chat_tasks.yaml file with current outstanding tasks.
        
        Args:
            tasks: Optional list of tasks to write. If None, extracts from context.
            
        Returns:
            Status message about the update
        """
        if tasks is None:
            tasks = self.extract_tasks_from_context("")  # Would pass actual context

        # Read existing tasks to preserve completed ones
        existing_tasks = []
        if self.chat_tasks_file.exists():
            with open(self.chat_tasks_file, 'r', encoding='utf-8') as f:
                existing_tasks = yaml.safe_load(f) or []

        # Merge with new tasks (preserve completed, add new outstanding ones)
        updated_tasks = existing_tasks + tasks

        # Write updated tasks
        with open(self.chat_tasks_file, 'w', encoding='utf-8') as f:
            yaml.dump(updated_tasks, f, default_flow_style=False, sort_keys=False)

        return f"✅ Updated chat_tasks.yaml with {len(tasks)} new tasks"

    def get_task_response(self, user_query: str) -> str:
        """
        Generate appropriate response for task-related queries.
        
        Args:
            user_query: The user's query
            
        Returns:
            Appropriate response following the protocol
        """
        if self.should_auto_update_tasks(user_query):
            # Auto-update case
            result = self.update_chat_tasks_file()
            return result

        elif self.should_ask_task_type(user_query):
            # Ask for clarification case
            return "**Ad-hoc or roadmap?**"

        else:
            # Not a task-related query
            return ""

    def validate_protocol_compliance(self) -> Dict[str, bool]:
        """
        Validate that the task management protocol is being followed.
        
        Returns:
            Dictionary of compliance checks
        """
        checks = {
            "correct_directory_exists": self.tasks_dir.exists(),
            "chat_tasks_in_correct_location": self.chat_tasks_file.exists(),
            "old_directory_not_used": not Path("/Users/camdouglas/quark/state/tasks").exists()
        }

        return checks


def main():
    """Demonstrate the chat task manager functionality."""
    manager = ChatTaskManager()

    print("🔧 Chat Task Manager - Protocol Enforcement")
    print("=" * 50)

    # Test protocol compliance
    compliance = manager.validate_protocol_compliance()
    print("\n📋 Protocol Compliance Check:")
    for check, passed in compliance.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")

    # Test query handling
    test_queries = [
        "what are my tasks?",
        "update chat tasks",
        "show me the roadmap",
        "update the chat tasks file"
    ]

    print("\n🧪 Query Response Tests:")
    for query in test_queries:
        response = manager.get_task_response(query)
        print(f"   Query: '{query}'")
        print(f"   Response: '{response}'\n")


if __name__ == "__main__":
    main()
