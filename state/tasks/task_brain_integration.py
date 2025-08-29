#!/usr/bin/env python3
"""
🔗 Task-Brain Integration System
Connects the biological brain agent with the actual task management system
to enable intelligent task prioritization and execution decisions.
"""

import sys
import os
# Add brain architecture modules to path
current_dir = os.path.dirname(os.path.abspath(__file__))
brain_path = os.path.join(current_dir, '..', 'brain_architecture', 'neural_core')
sys.path.append(brain_path)

from biological_brain_agent import BiologicalBrainAgent
import re
from typing import Dict, List, Any, Optional

class TaskBrainIntegration:
    """Integrates task management with biological brain agent"""
    
    def __init__(self, task_file_path: str = "current_tasks.md"):
        self.task_file_path = task_file_path
        self.brain_agent = BiologicalBrainAgent()
        self.tasks = {}
        self.task_updates = []
        
        print("🔗 Task-Brain Integration System initialized")
    
    def load_and_parse_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Load and parse tasks from the markdown file"""
        try:
            with open(self.task_file_path, 'r') as f:
                content = f.read()
            
            # Parse tasks using regex patterns
            tasks = self._parse_markdown_tasks(content)
            self.tasks = tasks
            
            # Update brain agent with real tasks
            self.brain_agent.active_tasks = tasks
            
            print(f"✅ Loaded {len(tasks)} tasks from {self.task_file_path}")
            return tasks
            
        except Exception as e:
            print(f"❌ Error loading tasks: {e}")
            return {}
    
    def _parse_markdown_tasks(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Parse markdown content to extract structured task data"""
        tasks = {}
        
        # Split content into sections
        sections = content.split('### **')
        
        for section in sections[1:]:  # Skip first empty section
            lines = section.split('\n')
            
            # Extract task ID and title from first line
            first_line = lines[0].strip()
            if '**' in first_line:
                task_id = first_line.split('**')[1].split(':')[0].strip()
                title = first_line.split(':')[1].strip() if ':' in first_line else task_id
                
                # Initialize task
                task = {
                    "id": task_id,
                    "title": title,
                    "status": "unknown",
                    "priority": "unknown",
                    "objective": "",
                    "estimated_effort": "",
                    "dependencies": [],
                    "deliverables": []
                }
                
                # Parse task details from remaining lines
                for line in lines[1:]:
                    line = line.strip()
                    
                    if line.startswith('**Status**:'):
                        task["status"] = line.split('**Status**:')[1].strip()
                    elif line.startswith('**Priority**:'):
                        task["priority"] = line.split('**Priority**:')[1].strip()
                    elif line.startswith('**Objective**:'):
                        task["objective"] = line.split('**Objective**:')[1].strip()
                    elif line.startswith('**Estimated Effort**:'):
                        task["estimated_effort"] = line.split('**Estimated Effort**:')[1].strip()
                    elif line.startswith('**Dependencies**:'):
                        deps_text = line.split('**Dependencies**:')[1].strip()
                        task["dependencies"] = [d.strip() for d in deps_text.split(',') if d.strip()]
                    elif line.startswith('**Deliverables**:'):
                        deliv_text = line.split('**Deliverables**:')[1].strip()
                        task["deliverables"] = [d.strip() for d in deliv_text.split(',') if d.strip()]
                
                tasks[task_id] = task
        
        return tasks
    
    def get_brain_agent_recommendations(self) -> Dict[str, Any]:
        """Get task recommendations from the biological brain agent"""
        # Ensure tasks are loaded
        if not self.tasks:
            self.load_and_parse_tasks()
        
        # Get brain agent decisions
        brain_output = self.brain_agent.step()
        
        # Extract recommendations
        recommendations = {
            "task_decisions": brain_output["task_decisions"],
            "execution_plan": brain_output["execution_results"],
            "resource_assessment": brain_output["agent_status"]["resource_state"],
            "brain_status": brain_output["agent_status"]["brain_modules"]
        }
        
        return recommendations
    
    def update_task_status(self, task_id: str, new_status: str, progress: Optional[float] = None):
        """Update task status and track changes"""
        if task_id in self.tasks:
            old_status = self.tasks[task_id]["status"]
            self.tasks[task_id]["status"] = new_status
            
            if progress is not None:
                self.tasks[task_id]["progress"] = progress
            
            # Record update
            update = {
                "task_id": task_id,
                "old_status": old_status,
                "new_status": new_status,
                "progress": progress,
                "timestamp": time.time()
            }
            self.task_updates.append(update)
            
            # Update brain agent
            self.brain_agent.active_tasks[task_id]["status"] = new_status
            
            print(f"✅ Updated task {task_id}: {old_status} → {new_status}")
        else:
            print(f"❌ Task {task_id} not found")
    
    def get_task_analytics(self) -> Dict[str, Any]:
        """Get comprehensive task analytics"""
        if not self.tasks:
            self.load_and_parse_tasks()
        
        # Calculate basic metrics
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if "COMPLETED" in t.get("status", "")])
        in_progress_tasks = len([t for t in self.tasks.values() if "IN PROGRESS" in t.get("status", "")])
        not_started_tasks = len([t for t in self.tasks.values() if "NOT STARTED" in t.get("status", "")])
        
        # Priority distribution
        high_priority = len([t for t in self.tasks.values() if "HIGH" in t.get("priority", "")])
        medium_priority = len([t for t in self.tasks.values() if "MEDIUM" in t.get("priority", "")])
        low_priority = len([t for t in self.tasks.values() if "LOW" in t.get("priority", "")])
        
        # Progress calculation
        overall_progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        analytics = {
            "task_counts": {
                "total": total_tasks,
                "completed": completed_tasks,
                "in_progress": in_progress_tasks,
                "not_started": not_started_tasks
            },
            "priority_distribution": {
                "high": high_priority,
                "medium": medium_priority,
                "low": low_priority
            },
            "progress": {
                "overall_percentage": overall_progress,
                "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0
            },
            "recent_updates": self.task_updates[-5:] if self.task_updates else []
        }
        
        return analytics
    
    def generate_execution_report(self) -> str:
        """Generate a human-readable execution report"""
        recommendations = self.get_brain_agent_recommendations()
        analytics = self.get_task_analytics()
        
        report = []
        report.append("# 🧠 QUARK Task Execution Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Task Analytics
        report.append("## 📊 Task Analytics")
        report.append(f"- **Total Tasks**: {analytics['task_counts']['total']}")
        report.append(f"- **Completed**: {analytics['task_counts']['completed']}")
        report.append(f"- **In Progress**: {analytics['task_counts']['in_progress']}")
        report.append(f"- **Not Started**: {analytics['task_counts']['not_started']}")
        report.append(f"- **Overall Progress**: {analytics['progress']['overall_percentage']:.1f}%")
        report.append("")
        
        # Brain Agent Recommendations
        report.append("## 🧠 Brain Agent Recommendations")
        
        executed_tasks = recommendations["execution_plan"]["executed_tasks"]
        deferred_tasks = recommendations["execution_plan"]["deferred_tasks"]
        
        report.append(f"### 🚀 Tasks Recommended for Execution ({len(executed_tasks)})")
        for task in executed_tasks:
            report.append(f"- **{task['task_id']}**: Priority {task['priority_score']:.2f}")
        
        if deferred_tasks:
            report.append(f"### ⏸️ Tasks Deferred ({len(deferred_tasks)})")
            for task_id in deferred_tasks:
                report.append(f"- **{task_id}**: Resource constraints or lower priority")
        
        # Resource Assessment
        report.append("")
        report.append("## 🔋 Resource Assessment")
        resources = recommendations["resource_assessment"]
        report.append(f"- **Cognitive Load**: {resources['cognitive_load']:.2f}")
        report.append(f"- **Working Memory**: {resources['working_memory_available']:.2f}")
        report.append(f"- **Energy Level**: {resources['energy_level']:.2f}")
        report.append(f"- **Motivation**: {resources['emotional_state']['motivation']:.2f}")
        
        return "\n".join(report)
    
    def save_execution_report(self, filename: str = "task_execution_report.md"):
        """Save execution report to file"""
        report = self.generate_execution_report()
        
        try:
            with open(filename, 'w') as f:
                f.write(report)
            print(f"✅ Execution report saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def step(self) -> Dict[str, Any]:
        """Step the integration system forward"""
        # Load and parse tasks
        tasks = self.load_and_parse_tasks()
        
        # Get brain agent recommendations
        recommendations = self.get_brain_agent_recommendations()
        
        # Get task analytics
        analytics = self.get_task_analytics()
        
        # Generate execution report
        report = self.generate_execution_report()
        
        return {
            "tasks": tasks,
            "recommendations": recommendations,
            "analytics": analytics,
            "report": report
        }

def main():
    """Main function to demonstrate task-brain integration"""
    print("🔗 Task-Brain Integration System Demonstration")
    print("=" * 50)
    
    # Initialize integration system
    integration = TaskBrainIntegration()
    
    # Step the system forward
    print("\n🔄 Stepping Task-Brain Integration System...")
    results = integration.step()
    
    # Display results
    print(f"\n📊 Loaded {len(results['tasks'])} tasks")
    print(f"🧠 Brain agent made {len(results['recommendations']['task_decisions'])} decisions")
    print(f"📈 Overall progress: {results['analytics']['progress']['overall_percentage']:.1f}%")
    
    # Show some recommendations
    print("\n🎯 Top Task Recommendations:")
    for i, decision in enumerate(results['recommendations']['task_decisions'][:3]):
        print(f"   {i+1}. {decision['task_id']}: {decision['decision']} (Priority: {decision['priority_score']:.2f})")
    
    # Save execution report
    print("\n💾 Saving execution report...")
    integration.save_execution_report()
    
    print("\n✅ Task-Brain Integration demonstration complete!")
    return integration

if __name__ == "__main__":
    import time
    try:
        integration = main()
    except Exception as e:
        print(f"❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
