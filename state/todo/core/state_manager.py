"""
State Manager Module
====================
Manages persistent state for TODO system.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List


class StateManager:
    """Manages persistent state and history."""
    
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.state_dir / 'todo_state.json'
        self.history_file = self.state_dir / 'todo_history.json'
        
        self.state = self._load_state()
        self.history = self._load_history()
    
    def _load_state(self) -> Dict:
        """Load persistent state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'current_task': None,
            'last_command': None,
            'last_validation': None,
            'session_start': datetime.now().isoformat(),
            'preferences': {}
        }
    
    def _load_history(self) -> List[Dict]:
        """Load command history."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except:
                pass
        
        return []
    
    def save_state(self) -> None:
        """Save current state."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save state: {e}")
    
    def save_history(self) -> None:
        """Save command history."""
        try:
            # Keep only last 100 commands
            self.history = self.history[-100:]
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save history: {e}")
    
    def record_command(self, command: str, system: str, action: str) -> None:
        """Record a command in history."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'system': system,
            'action': action
        }
        
        self.history.append(entry)
        self.state['last_command'] = command
        
        self.save_history()
        self.save_state()
    
    def set_current_task(self, task_id: Optional[str]) -> None:
        """Set the current working task."""
        self.state['current_task'] = task_id
        self.save_state()
    
    def get_current_task(self) -> Optional[str]:
        """Get current working task."""
        return self.state.get('current_task')
    
    def update_validation_time(self) -> None:
        """Update last validation timestamp."""
        self.state['last_validation'] = datetime.now().isoformat()
        self.save_state()
    
    def get_recent_commands(self, count: int = 5) -> List[str]:
        """Get recent commands."""
        return [h['command'] for h in self.history[-count:]]
    
    def get_suggestions(self) -> List[str]:
        """Get command suggestions based on history."""
        suggestions = []
        
        # Check if validation is stale (>1 day)
        if self.state.get('last_validation'):
            try:
                last_val = datetime.fromisoformat(self.state['last_validation'])
                if (datetime.now() - last_val).days > 0:
                    suggestions.append("validate current changes")
            except:
                pass
        
        # Suggest continuing current task
        if self.state.get('current_task'):
            suggestions.append(f"work on {self.state['current_task']}")
        
        # Analyze recent commands for patterns
        recent = self.get_recent_commands(10)
        if recent:
            # If lots of edits, suggest testing
            if any('edit' in cmd or 'write' in cmd for cmd in recent):
                suggestions.append("run tests")
            
            # If validation done, suggest next task
            if any('validate' in cmd for cmd in recent):
                suggestions.append("list pending tasks")
        
        return suggestions[:3]  # Return top 3 suggestions
