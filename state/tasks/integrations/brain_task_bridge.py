#!/usr/bin/env python3
"""
🧠 Brain-Task Integration Bridge

This module provides the integration between the brain's automatic goal management system
and the central task management system. It enables bidirectional synchronization of goals,
tasks, and consciousness states.

Features:
- Real-time brain state monitoring
- Automatic goal generation from consciousness
- Task creation from brain-generated goals
- Consciousness state updates from task progress
- Priority mapping between brain and task systems
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainStateMonitor:
    """Monitors the brain's consciousness state and goal generation"""
    
    def __init__(self, brain_architecture_path: str = "brain_architecture"):
        self.brain_path = Path(brain_architecture_path)
        self.consciousness_state = {}
        self.goal_generation = {}
        self.attention_state = {}
        self.last_update = datetime.now()
        
        # Initialize default states
        self._initialize_default_states()
    
    def _initialize_default_states(self):
        """Initialize default brain states"""
        self.consciousness_state = {
            "awake": True,
            "attention_focus": "general",
            "emotional_state": "neutral",
            "cognitive_load": 0.5,
            "memory_consolidation": False,
            "learning_mode": "active"
        }
        
        self.attention_state = {
            "task_bias": 0.6,
            "internal_bias": 0.4,
            "focus_target": "balanced"
        }
        
        self.goal_generation = {
            "active_goals": [],
            "goal_priority": "medium",
            "goal_type": "learning"
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current brain consciousness state"""
        return {
            "consciousness": self.consciousness_state.copy(),
            "goals": self.goal_generation.copy(),
            "attention": self.attention_state.copy(),
            "timestamp": datetime.now().isoformat()
        }
    
    def update_consciousness(self, task_status: Dict[str, Any]):
        """Update consciousness state based on task status"""
        # Update cognitive load based on active tasks
        active_tasks = task_status.get("active_tasks", 0)
        completed_tasks = task_status.get("completed_tasks", 0)
        
        # Adjust cognitive load based on task activity
        if active_tasks > 3:
            self.consciousness_state["cognitive_load"] = min(1.0, 0.5 + (active_tasks * 0.1))
        elif completed_tasks > 0:
            self.consciousness_state["cognitive_load"] = max(0.2, 0.5 - (completed_tasks * 0.05))
        
        # Update learning mode based on task completion
        if completed_tasks > 0:
            self.consciousness_state["learning_mode"] = "active"
        
        # Update attention focus based on task priority
        high_priority_tasks = task_status.get("high_priority_tasks", 0)
        if high_priority_tasks > 0:
            self.attention_state["task_bias"] = min(1.0, 0.6 + (high_priority_tasks * 0.1))
            self.attention_state["focus_target"] = "external_tasks"
        else:
            self.attention_state["task_bias"] = max(0.3, 0.6 - 0.1)
            self.attention_state["focus_target"] = "balanced"
        
        self.last_update = datetime.now()
        logger.info(f"Updated consciousness state: {self.consciousness_state}")

class BrainGoalGenerator:
    """Generates goals based on brain consciousness state"""
    
    def __init__(self):
        self.goal_templates = self._load_goal_templates()
        self.priority_weights = {
            "homeostasis": 1.0,
            "learning": 0.8,
            "adaptation": 0.7,
            "exploration": 0.5
        }
    
    def _load_goal_templates(self) -> Dict[str, List[str]]:
        """Load goal templates for different types"""
        return {
            "homeostasis": [
                "Reduce cognitive load through task prioritization",
                "Maintain optimal consciousness state",
                "Balance internal and external focus",
                "Optimize resource allocation"
            ],
            "learning": [
                "Explore new knowledge domains",
                "Develop new skills and capabilities",
                "Integrate learned information",
                "Apply knowledge to current tasks"
            ],
            "adaptation": [
                "Adapt to changing task requirements",
                "Optimize task execution strategies",
                "Improve system integration",
                "Enhance coordination between systems"
            ],
            "exploration": [
                "Discover new task opportunities",
                "Research innovative approaches",
                "Explore system capabilities",
                "Investigate potential improvements"
            ]
        }
    
    def generate_goals(self, consciousness_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate goals based on consciousness state"""
        goals = []
        
        # Homeostasis goals (high priority)
        if consciousness_state.get("cognitive_load", 0) > 0.8:
            goals.append({
                "id": f"homeostasis_{int(time.time())}",
                "type": "homeostasis",
                "priority": "high",
                "description": "Reduce cognitive load through task prioritization",
                "brain_origin": True,
                "urgency": "immediate"
            })
        
        # Learning goals (medium priority)
        if consciousness_state.get("learning_mode") == "active":
            goals.append({
                "id": f"learning_{int(time.time())}",
                "type": "learning",
                "priority": "medium",
                "description": "Explore new knowledge domains",
                "brain_origin": True,
                "urgency": "soon"
            })
        
        # Adaptation goals (medium priority)
        if consciousness_state.get("attention_focus") == "task-positive":
            goals.append({
                "id": f"adaptation_{int(time.time())}",
                "type": "adaptation",
                "priority": "medium",
                "description": "Optimize task execution strategies",
                "brain_origin": True,
                "urgency": "soon"
            })
        
        # Exploration goals (low priority)
        if consciousness_state.get("cognitive_load", 0) < 0.4:
            goals.append({
                "id": f"exploration_{int(time.time())}",
                "description": "Discover new task opportunities",
                "brain_origin": True,
                "urgency": "when_available"
            })
        
        logger.info(f"Generated {len(goals)} brain goals")
        return goals

class BrainGoalTranslator:
    """Translates brain-generated goals to tasks"""
    
    def __init__(self, task_system_path: str = "tasks"):
        self.task_system_path = Path(task_system_path)
        self.goal_mapping = {}
        
    def translate_brain_goals(self, brain_goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translate brain-generated goals to tasks"""
        tasks = []
        
        for goal in brain_goals:
            task = {
                "title": goal["description"],
                "priority": self.map_priority(goal["priority"]),
                "source": "brain_generated",
                "brain_goal_id": goal.get("id"),
                "acceptance_criteria": self.generate_criteria(goal),
                "estimated_effort": self.estimate_effort(goal),
                "dependencies": [],
                "tags": ["brain_generated", goal.get("type", "general")],
                "created_at": datetime.now().isoformat(),
                "due_date": self.calculate_due_date(goal)
            }
            tasks.append(task)
        
        logger.info(f"Translated {len(tasks)} brain goals to tasks")
        return tasks
    
    def map_priority(self, brain_priority: str) -> str:
        """Map brain priority to task priority"""
        priority_mapping = {
            "high": "high",
            "medium": "medium",
            "low": "low"
        }
        return priority_mapping.get(brain_priority, "medium")
    
    def generate_criteria(self, goal: Dict[str, Any]) -> List[str]:
        """Generate acceptance criteria for brain goals"""
        goal_type = goal.get("type", "general")
        
        if goal_type == "homeostasis":
            return [
                "Cognitive load reduced to optimal levels",
                "Consciousness state stabilized",
                "Resource allocation optimized"
            ]
        elif goal_type == "learning":
            return [
                "New knowledge acquired and integrated",
                "Skills developed and demonstrated",
                "Learning outcomes documented"
            ]
        elif goal_type == "adaptation":
            return [
                "Task execution strategies optimized",
                "System integration improved",
                "Coordination enhanced"
            ]
        else:  # exploration
            return [
                "New opportunities identified",
                "Innovative approaches researched",
                "Potential improvements documented"
            ]
    
    def estimate_effort(self, goal: Dict[str, Any]) -> str:
        """Estimate effort required for brain goal"""
        goal_type = goal.get("type", "general")
        
        effort_mapping = {
            "homeostasis": "low",
            "learning": "medium",
            "adaptation": "medium",
            "exploration": "low"
        }
        return effort_mapping.get(goal_type, "medium")
    
    def calculate_due_date(self, goal: Dict[str, Any]) -> str:
        """Calculate due date based on goal urgency"""
        urgency = goal.get("urgency", "soon")
        now = datetime.now()
        
        if urgency == "immediate":
            due_date = now.replace(hour=now.hour + 2)  # 2 hours from now
        elif urgency == "soon":
            due_date = now.replace(day=now.day + 1)  # Tomorrow
        else:  # when_available
            due_date = now.replace(day=now.day + 7)  # Next week
        
        return due_date.isoformat()

class BrainTaskSynchronizer:
    """Synchronizes brain state with task system"""
    
    def __init__(self, brain_monitor: BrainStateMonitor, task_system_path: str = "tasks"):
        self.brain_monitor = brain_monitor
        self.task_system_path = Path(task_system_path)
        self.goal_generator = BrainGoalGenerator()
        self.goal_translator = BrainGoalTranslator(task_system_path)
        self.sync_interval = 5.0  # seconds
        self.running = False
        self.sync_thread = None
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for brain-generated tasks"""
        brain_tasks_dir = self.task_system_path / "active_tasks" / "brain_generated"
        brain_tasks_dir.mkdir(parents=True, exist_ok=True)
        
        # Create README for brain-generated tasks
        readme_path = brain_tasks_dir / "README.md"
        if not readme_path.exists():
            self._create_brain_tasks_readme(readme_path)
    
    def _create_brain_tasks_readme(self, readme_path: Path):
        """Create README for brain-generated tasks directory"""
        readme_content = """# 🧠 Brain-Generated Tasks

This directory contains tasks that were automatically generated by the brain's consciousness system.

## Task Types

### Homeostasis Tasks
- **Priority**: High
- **Purpose**: Maintain optimal brain state and cognitive load
- **Examples**: Reduce cognitive load, optimize resource allocation

### Learning Tasks
- **Priority**: Medium
- **Purpose**: Acquire new knowledge and develop skills
- **Examples**: Explore new domains, integrate information

### Adaptation Tasks
- **Priority**: Medium
- **Purpose**: Optimize task execution and system integration
- **Examples**: Improve strategies, enhance coordination

### Exploration Tasks
- **Priority**: Low
- **Purpose**: Discover opportunities and research improvements
- **Examples**: Identify new approaches, investigate capabilities

## Integration

These tasks are automatically synchronized with the central task system and can be managed like any other task.
"""
        readme_path.write_text(readme_content)
    
    def start_synchronization(self):
        """Start bidirectional synchronization"""
        if self.running:
            logger.warning("Synchronization already running")
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._synchronization_loop, daemon=True)
        self.sync_thread.start()
        logger.info("Started brain-task synchronization")
    
    def stop_synchronization(self):
        """Stop synchronization"""
        self.running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5.0)
        logger.info("Stopped brain-task synchronization")
    
    def _synchronization_loop(self):
        """Main synchronization loop"""
        while self.running:
            try:
                # Brain → Task System
                brain_state = self.brain_monitor.get_current_state()
                self.update_task_system(brain_state)
                
                # Task System → Brain
                task_status = self.get_task_system_status()
                self.brain_monitor.update_consciousness(task_status)
                
                time.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Error in synchronization loop: {e}")
                time.sleep(self.sync_interval)
    
    def update_task_system(self, brain_state: Dict[str, Any]):
        """Update task system with brain state"""
        try:
            # Generate goals from brain state
            goals = self.goal_generator.generate_goals(brain_state["consciousness"])
            
            if goals:
                # Translate goals to tasks
                tasks = self.goal_translator.translate_brain_goals(goals)
                
                # Save brain-generated tasks
                self._save_brain_tasks(tasks)
                
                # Update central task system
                self._update_central_task_system(tasks)
                
                logger.info(f"Updated task system with {len(tasks)} brain-generated tasks")
        
        except Exception as e:
            logger.error(f"Error updating task system: {e}")
    
    def _save_brain_tasks(self, tasks: List[Dict[str, Any]]):
        """Save brain-generated tasks to file system"""
        brain_tasks_dir = self.task_system_path / "active_tasks" / "brain_generated"
        
        for task in tasks:
            task_id = task.get("brain_goal_id", f"brain_task_{int(time.time())}")
            task_file = brain_tasks_dir / f"{task_id}.json"
            
            # Save task as JSON
            with open(task_file, 'w') as f:
                json.dump(task, f, indent=2)
    
    def _update_central_task_system(self, tasks: List[Dict[str, Any]]):
        """Update central task system with brain-generated tasks"""
        # This would integrate with your existing task management system
        # For now, we'll create a summary file
        
        summary_file = self.task_system_path / "active_tasks" / "brain_generated" / "TASK_SUMMARY.md"
        
        summary_content = self._generate_task_summary(tasks)
        summary_file.write_text(summary_content)
    
    def _generate_task_summary(self, tasks: List[Dict[str, Any]]) -> str:
        """Generate summary of brain-generated tasks"""
        summary = f"""# 🧠 Brain-Generated Tasks Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Tasks**: {len(tasks)}

## Task Overview

"""
        
        for task in tasks:
            summary += f"""### {task['title']}
- **Priority**: {task['priority']}
- **Type**: {task.get('tags', ['general'])[-1]}
- **Effort**: {task.get('estimated_effort', 'medium')}
- **Due**: {task.get('due_date', 'Not specified')}
- **Source**: Brain-generated (ID: {task.get('brain_goal_id', 'Unknown')})

"""
        
        return summary
    
    def get_task_system_status(self) -> Dict[str, Any]:
        """Get current task system status"""
        try:
            # Read task status from central system
            status_file = self.task_system_path / "TASK_STATUS.md"
            
            if status_file.exists():
                # Parse task status (simplified)
                return {
                    "active_tasks": 6,  # From your current system
                    "high_priority_tasks": 3,
                    "completed_tasks": 0,
                    "system_health": "integrating"
                }
            else:
                return {
                    "active_tasks": 0,
                    "high_priority_tasks": 0,
                    "completed_tasks": 0,
                    "system_health": "unknown"
                }
        
        except Exception as e:
            logger.error(f"Error getting task system status: {e}")
            return {
                "active_tasks": 0,
                "high_priority_tasks": 0,
                "completed_tasks": 0,
                "system_health": "error"
            }

class BrainTaskIntegrationManager:
    """Main manager for brain-task integration"""
    
    def __init__(self, brain_architecture_path: str = "brain_architecture", task_system_path: str = "tasks"):
        self.brain_monitor = BrainStateMonitor(brain_architecture_path)
        self.synchronizer = BrainTaskSynchronizer(self.brain_monitor, task_system_path)
        self.integration_status = "initialized"
    
    def start_integration(self):
        """Start the brain-task integration"""
        try:
            self.synchronizer.start_synchronization()
            self.integration_status = "running"
            logger.info("Brain-task integration started successfully")
            
        except Exception as e:
            logger.error(f"Error starting integration: {e}")
            self.integration_status = "error"
    
    def stop_integration(self):
        """Stop the brain-task integration"""
        try:
            self.synchronizer.stop_synchronization()
            self.integration_status = "stopped"
            logger.info("Brain-task integration stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping integration: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        brain_state = self.brain_monitor.get_current_state()
        
        return {
            "integration_status": self.integration_status,
            "brain_state": brain_state,
            "synchronization_active": self.synchronizer.running,
            "last_sync": self.brain_monitor.last_update.isoformat(),
            "brain_tasks_directory": str(self.synchronizer.task_system_path / "active_tasks" / "brain_generated")
        }
    
    def generate_test_goals(self) -> List[Dict[str, Any]]:
        """Generate test goals for demonstration"""
        test_consciousness_state = {
            "awake": True,
            "attention_focus": "task-positive",
            "emotional_state": "focused",
            "cognitive_load": 0.8,
            "learning_mode": "active"
        }
        
        goal_generator = BrainGoalGenerator()
        goals = goal_generator.generate_goals(test_consciousness_state)
        
        goal_translator = BrainGoalTranslator()
        tasks = goal_translator.translate_brain_goals(goals)
        
        return tasks

def main():
    """Main function for testing the integration"""
    logger.info("Starting Brain-Task Integration System")
    
    # Create integration manager
    integration_manager = BrainTaskIntegrationManager()
    
    # Start integration
    integration_manager.start_integration()
    
    try:
        # Keep running for demonstration
        while True:
            status = integration_manager.get_integration_status()
            logger.info(f"Integration Status: {status['integration_status']}")
            logger.info(f"Brain State: {status['brain_state']['consciousness']}")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Stopping integration...")
        integration_manager.stop_integration()
        logger.info("Integration stopped")

if __name__ == "__main__":
    main()
