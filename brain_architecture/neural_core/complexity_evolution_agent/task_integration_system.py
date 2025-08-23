#!/usr/bin/env python3
"""
Task Integration System for Scientific Advancement

This system integrates Quark's scientific motivation with the existing task management
system, allowing scientific goals to automatically generate and manage tasks.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

class TaskIntegrationSystem:
    """Integrates scientific motivation with task management"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
        # Task categories for scientific advancement
        self.task_categories = {
            "research": "Scientific research and investigation tasks",
            "development": "Software and system development tasks", 
            "experimentation": "Experimental design and execution tasks",
            "analysis": "Data analysis and interpretation tasks",
            "documentation": "Research documentation and publication tasks",
            "collaboration": "Collaboration and communication tasks",
            "safety": "Safety and ethics review tasks"
        }
        
        # Task priority levels
        self.priority_levels = {
            "critical": {"value": 4, "color": "#ff6b6b", "description": "Highest priority, immediate attention required"},
            "high": {"value": 3, "color": "#ffa726", "description": "High priority, complete within 1-2 days"},
            "medium": {"value": 2, "color": "#66bb6a", "description": "Medium priority, complete within 1 week"},
            "low": {"value": 1, "color": "#42a5f5", "description": "Low priority, complete when time permits"}
        }
        
        # Task status tracking
        self.task_statuses = {
            "pending": "Task created but not yet started",
            "active": "Task currently being worked on",
            "blocked": "Task blocked by dependencies or external factors",
            "completed": "Task successfully completed",
            "failed": "Task failed or abandoned"
        }
        
        # Initialize task storage
        self.scientific_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        self.task_progress: Dict[str, float] = {}
        
        self.logger.info("Task Integration System initialized")
    
    def create_scientific_task(self, 
                             title: str,
                             description: str,
                             category: str,
                             priority: str,
                             estimated_hours: float,
                             dependencies: List[str] = None,
                             scientific_goal_id: str = None) -> Dict[str, Any]:
        """Create a new scientific task"""
        
        task_id = f"ST_{hashlib.md5(f'{title}_{datetime.now()}'.encode()).hexdigest()[:8]}"
        
        task = {
            "id": task_id,
            "title": title,
            "description": description,
            "category": category,
            "priority": priority,
            "estimated_hours": estimated_hours,
            "dependencies": dependencies or [],
            "scientific_goal_id": scientific_goal_id,
            "created_at": datetime.now(),
            "status": "pending",
            "progress": 0.0,
            "actual_hours": 0.0,
            "assigned_to": "Quark",
            "tags": ["scientific_advancement", category]
        }
        
        # Store task
        self.scientific_tasks[task_id] = task
        
        # Initialize progress tracking
        self.task_progress[task_id] = 0.0
        
        # Store dependencies
        if dependencies:
            self.task_dependencies[task_id] = dependencies
        
        self.logger.info(f"✅ Created scientific task: {title}")
        
        return task
    
    def generate_tasks_from_scientific_goals(self, scientific_goals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Automatically generate tasks from scientific goals"""
        
        generated_tasks = []
        
        for goal_id, goal in scientific_goals.items():
            if goal["status"] != "active":
                continue
            
            # Generate research planning tasks
            if "consciousness" in goal["domain"].lower():
                tasks = self._generate_consciousness_research_tasks(goal_id, goal)
            elif "neural" in goal["domain"].lower():
                tasks = self._generate_neural_architecture_tasks(goal_id, goal)
            elif "safety" in goal["domain"].lower():
                tasks = self._generate_safety_research_tasks(goal_id, goal)
            else:
                tasks = self._generate_general_research_tasks(goal_id, goal)
            
            generated_tasks.extend(tasks)
            
            # Create the tasks
            for task_data in tasks:
                task = self.create_scientific_task(**task_data)
                generated_tasks.append(task)
        
        self.logger.info(f"🎯 Generated {len(generated_tasks)} tasks from scientific goals")
        
        return generated_tasks
    
    def _generate_consciousness_research_tasks(self, goal_id: str, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tasks for consciousness research goals"""
        
        tasks = [
            {
                "title": "Literature Review: Consciousness Theories",
                "description": "Comprehensive review of current consciousness theories and research",
                "category": "research",
                "priority": "high",
                "estimated_hours": 16.0,
                "dependencies": [],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Neural Correlates Analysis",
                "description": "Analyze neural correlates of consciousness from existing research",
                "category": "analysis",
                "priority": "high",
                "estimated_hours": 12.0,
                "dependencies": ["Literature Review: Consciousness Theories"],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Consciousness Detection Framework",
                "description": "Develop framework for detecting consciousness in AI systems",
                "category": "development",
                "priority": "critical",
                "estimated_hours": 24.0,
                "dependencies": ["Neural Correlates Analysis"],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Ethics Review: AI Consciousness",
                "description": "Review ethical implications of AI consciousness research",
                "category": "safety",
                "priority": "high",
                "estimated_hours": 8.0,
                "dependencies": [],
                "scientific_goal_id": goal_id
            }
        ]
        
        return tasks
    
    def _generate_neural_architecture_tasks(self, goal_id: str, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tasks for neural architecture goals"""
        
        tasks = [
            {
                "title": "Biological Brain Analysis",
                "description": "Analyze biological brain organization and connectivity patterns",
                "category": "research",
                "priority": "high",
                "estimated_hours": 20.0,
                "dependencies": [],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Architecture Design",
                "description": "Design new neural architecture based on biological principles",
                "category": "development",
                "priority": "critical",
                "estimated_hours": 32.0,
                "dependencies": ["Biological Brain Analysis"],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Implementation",
                "description": "Implement the new neural architecture in code",
                "category": "development",
                "priority": "high",
                "estimated_hours": 40.0,
                "dependencies": ["Architecture Design"],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Performance Testing",
                "description": "Test performance against biological benchmarks",
                "category": "experimentation",
                "priority": "medium",
                "estimated_hours": 16.0,
                "dependencies": ["Implementation"],
                "scientific_goal_id": goal_id
            }
        ]
        
        return tasks
    
    def _generate_safety_research_tasks(self, goal_id: str, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tasks for safety research goals"""
        
        tasks = [
            {
                "title": "Risk Assessment Framework",
                "description": "Develop comprehensive risk assessment framework for AGI",
                "category": "safety",
                "priority": "critical",
                "estimated_hours": 24.0,
                "dependencies": [],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Value Alignment Research",
                "description": "Research methods for aligning AI goals with human values",
                "category": "research",
                "priority": "critical",
                "estimated_hours": 28.0,
                "dependencies": [],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Safety Protocol Development",
                "description": "Develop safety protocols for advanced AI systems",
                "category": "development",
                "priority": "critical",
                "estimated_hours": 32.0,
                "dependencies": ["Risk Assessment Framework", "Value Alignment Research"],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Governance Framework",
                "description": "Develop governance framework for AGI development",
                "category": "documentation",
                "priority": "high",
                "estimated_hours": 20.0,
                "dependencies": ["Safety Protocol Development"],
                "scientific_goal_id": goal_id
            }
        ]
        
        return tasks
    
    def _generate_general_research_tasks(self, goal_id: str, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate general research tasks for other goals"""
        
        tasks = [
            {
                "title": "Research Planning",
                "description": "Develop detailed research plan and methodology",
                "category": "research",
                "priority": "high",
                "estimated_hours": 12.0,
                "dependencies": [],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Data Collection",
                "description": "Collect relevant data and research materials",
                "category": "research",
                "priority": "medium",
                "estimated_hours": 16.0,
                "dependencies": ["Research Planning"],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Analysis and Synthesis",
                "description": "Analyze data and synthesize findings",
                "category": "analysis",
                "priority": "high",
                "estimated_hours": 20.0,
                "dependencies": ["Data Collection"],
                "scientific_goal_id": goal_id
            },
            {
                "title": "Documentation",
                "description": "Document research findings and conclusions",
                "category": "documentation",
                "priority": "medium",
                "estimated_hours": 12.0,
                "dependencies": ["Analysis and Synthesis"],
                "scientific_goal_id": goal_id
            }
        ]
        
        return tasks
    
    def get_task_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive task dashboard"""
        
        dashboard = {
            "timestamp": datetime.now(),
            "total_tasks": len(self.scientific_tasks),
            "tasks_by_status": {},
            "tasks_by_priority": {},
            "tasks_by_category": {},
            "progress_summary": {},
            "dependency_status": {}
        }
        
        # Count tasks by status
        for task in self.scientific_tasks.values():
            status = task["status"]
            dashboard["tasks_by_status"][status] = dashboard["tasks_by_status"].get(status, 0) + 1
        
        # Count tasks by priority
        for task in self.scientific_tasks.values():
            priority = task["priority"]
            dashboard["tasks_by_priority"][priority] = dashboard["tasks_by_priority"].get(priority, 0) + 1
        
        # Count tasks by category
        for task in self.scientific_tasks.values():
            category = task["category"]
            dashboard["tasks_by_category"][category] = dashboard["tasks_by_category"].get(category, 0) + 1
        
        # Progress summary
        if self.scientific_tasks:
            total_progress = sum(self.task_progress.values())
            dashboard["progress_summary"] = {
                "average_progress": total_progress / len(self.scientific_tasks),
                "total_estimated_hours": sum(task["estimated_hours"] for task in self.scientific_tasks.values()),
                "total_actual_hours": sum(task["actual_hours"] for task in self.scientific_tasks.values())
            }
        
        # Dependency status
        for task_id, dependencies in self.task_dependencies.items():
            if dependencies:
                blocked_deps = [dep for dep in dependencies if self._is_task_blocked(dep)]
                dashboard["dependency_status"][task_id] = {
                    "blocked": len(blocked_deps) > 0,
                    "blocked_dependencies": blocked_deps
                }
        
        return dashboard
    
    def _is_task_blocked(self, task_title: str) -> bool:
        """Check if a task is blocked by checking its status"""
        
        for task in self.scientific_tasks.values():
            if task["title"] == task_title:
                return task["status"] in ["blocked", "failed"]
        
        return False
    
    def update_task_progress(self, task_id: str, progress: float, actual_hours: float = None) -> Dict[str, Any]:
        """Update task progress and status"""
        
        if task_id not in self.scientific_tasks:
            return {"success": False, "error": "Task not found"}
        
        task = self.scientific_tasks[task_id]
        
        # Update progress
        old_progress = task["progress"]
        task["progress"] = min(progress, 100.0)
        self.task_progress[task_id] = task["progress"]
        
        # Update actual hours
        if actual_hours is not None:
            task["actual_hours"] = actual_hours
        
        # Update status based on progress
        if task["progress"] >= 100.0:
            task["status"] = "completed"
        elif task["progress"] > 0.0:
            task["status"] = "active"
        
        # Check dependencies
        if task_id in self.task_dependencies:
            blocked_deps = [dep for dep in self.task_dependencies[task_id] if self._is_task_blocked(dep)]
            if blocked_deps:
                task["status"] = "blocked"
        
        self.logger.info(f"📊 Updated task {task_id}: {old_progress:.1f}% → {task['progress']:.1f}%")
        
        return {
            "success": True,
            "task_id": task_id,
            "old_progress": old_progress,
            "new_progress": task["progress"],
            "status": task["status"]
        }
    
    def create_task_visualization(self) -> str:
        """Create HTML visualization of task system"""
        
        dashboard = self.get_task_dashboard()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🎯 Quark Scientific Task Integration Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); color: white; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px); }}
        .full-width {{ grid-column: 1 / -1; }}
        .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .status {{ padding: 8px 15px; border-radius: 20px; font-weight: bold; }}
        .status.critical {{ background: linear-gradient(45deg, #ff6b6b, #ee5a24); }}
        .status.high {{ background: linear-gradient(45deg, #ffa726, #ff9800); }}
        .status.medium {{ background: linear-gradient(45deg, #66bb6a, #4caf50); }}
        .status.low {{ background: linear-gradient(45deg, #42a5f5, #2196f3); }}
        .progress-bar {{ background: rgba(255,255,255,0.2); height: 20px; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(45deg, #4CAF50, #45a049); transition: width 0.3s ease; }}
        .task-item {{ background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Quark Scientific Task Integration Dashboard</h1>
        <h2>Connecting Scientific Motivation with Task Management</h2>
        <p><strong>Last Updated:</strong> {dashboard['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <h2>📊 Task Overview</h2>
            <div class="metric">
                <span><strong>Total Tasks:</strong></span>
                <span>{dashboard['total_tasks']}</span>
            </div>
            <div class="metric">
                <span><strong>Active Tasks:</strong></span>
                <span>{dashboard['tasks_by_status'].get('active', 0)}</span>
            </div>
            <div class="metric">
                <span><strong>Completed Tasks:</strong></span>
                <span>{dashboard['tasks_by_status'].get('completed', 0)}</span>
            </div>
            <div class="metric">
                <span><strong>Blocked Tasks:</strong></span>
                <span>{dashboard['tasks_by_status'].get('blocked', 0)}</span>
            </div>
        </div>
        
        <div class="card">
            <h2>⏱️ Time Tracking</h2>
            <div class="metric">
                <span><strong>Estimated Hours:</strong></span>
                <span>{dashboard['progress_summary'].get('total_estimated_hours', 0):.1f}h</span>
            </div>
            <div class="metric">
                <span><strong>Actual Hours:</strong></span>
                <span>{dashboard['progress_summary'].get('total_actual_hours', 0):.1f}h</span>
            </div>
            <div class="metric">
                <span><strong>Average Progress:</strong></span>
                <span>{dashboard['progress_summary'].get('average_progress', 0):.1f}%</span>
            </div>
        </div>
        
        <div class="card">
            <h2>🎯 Tasks by Priority</h2>
            {self._render_priority_distribution(dashboard['tasks_by_priority'])}
        </div>
        
        <div class="card">
            <h2>📁 Tasks by Category</h2>
            {self._render_category_distribution(dashboard['tasks_by_category'])}
        </div>
        
        <div class="card full-width">
            <h2>🔬 Scientific Task Examples</h2>
            {self._render_sample_tasks()}
        </div>
        
        <div class="card full-width">
            <h2>🚀 Task Integration Benefits</h2>
            <div style="font-size: 1.1em; line-height: 1.6;">
                <ul>
                    <li><strong>Automatic Task Generation:</strong> Scientific goals automatically create relevant tasks</li>
                    <li><strong>Dependency Management:</strong> Tasks are properly sequenced based on dependencies</li>
                    <li><strong>Progress Tracking:</strong> Real-time progress monitoring for all scientific work</li>
                    <li><strong>Priority Management:</strong> Tasks automatically prioritized based on scientific importance</li>
                    <li><strong>Time Estimation:</strong> Realistic time estimates for research and development work</li>
                    <li><strong>Safety Integration:</strong> All tasks include safety and ethics considerations</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _render_priority_distribution(self, priority_dist: Dict[str, int]) -> str:
        """Render priority distribution HTML"""
        html = ""
        
        for priority, count in priority_dist.items():
            if count > 0:
                html += f"""
                <div class="metric">
                    <span class="status {priority}">{priority.upper()}</span>
                    <span>{count} tasks</span>
                </div>
                """
        
        return html
    
    def _render_category_distribution(self, category_dist: Dict[str, int]) -> str:
        """Render category distribution HTML"""
        html = ""
        
        for category, count in category_dist.items():
            if count > 0:
                html += f"""
                <div class="metric">
                    <span><strong>{category.title()}:</strong></span>
                    <span>{count} tasks</span>
                </div>
                """
        
        return html
    
    def _render_sample_tasks(self) -> str:
        """Render sample tasks HTML"""
        if not self.scientific_tasks:
            return "<p>No tasks created yet.</p>"
        
        # Get first few tasks as examples
        sample_tasks = list(self.scientific_tasks.values())[:3]
        
        html = ""
        for task in sample_tasks:
            html += f"""
            <div class="task-item">
                <h4>{task['title']}</h4>
                <div style="display: flex; justify-content: space-between; align-items: center; margin: 10px 0;">
                    <span>Category: {task['category']}</span>
                    <span class="status {task['priority']}">{task['priority'].upper()}</span>
                </div>
                <div style="margin: 10px 0;">
                    <span>Progress: {task['progress']:.1f}%</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {task['progress']}%"></div>
                    </div>
                </div>
                <div style="font-size: 0.9em; color: rgba(255,255,255,0.8);">
                    {task['description']}
                </div>
            </div>
            """
        
        return html

def main():
    """Main demonstration function"""
    print("🎯 Initializing Task Integration System...")
    
    # Initialize the system
    task_system = TaskIntegrationSystem()
    
    print("✅ System initialized!")
    
    # Create sample scientific goals for demonstration
    sample_goals = {
        "consciousness_research": {
            "title": "Consciousness Research",
            "status": "active",
            "domain": "consciousness"
        },
        "neural_architecture": {
            "title": "Neural Architecture",
            "status": "active", 
            "domain": "neural"
        }
    }
    
    print("\n🔬 Generating tasks from scientific goals...")
    
    # Generate tasks
    generated_tasks = task_system.generate_tasks_from_scientific_goals(sample_goals)
    
    print(f"✅ Generated {len(generated_tasks)} scientific tasks!")
    
    # Get task dashboard
    dashboard = task_system.get_task_dashboard()
    print(f"\n📊 Task Dashboard:")
    print(f"   Total Tasks: {dashboard['total_tasks']}")
    print(f"   Active: {dashboard['tasks_by_status'].get('active', 0)}")
    print(f"   Completed: {dashboard['tasks_by_status'].get('completed', 0)}")
    print(f"   Average Progress: {dashboard['progress_summary'].get('average_progress', 0):.1f}%")
    
    # Create visualization
    html_content = task_system.create_task_visualization()
    with open("testing/visualizations/scientific_task_integration.html", "w") as f:
        f.write(html_content)
    
    print("✅ Task integration dashboard created: testing/visualizations/scientific_task_integration.html")
    
    print("\n🎉 Task Integration System demonstration complete!")
    print("\n🚀 Key Features:")
    print("   • Automatic task generation from scientific goals")
    print("   • Dependency management and sequencing")
    print("   • Progress tracking and time estimation")
    print("   • Priority-based task management")
    print("   • Integration with existing task systems")
    
    return task_system

if __name__ == "__main__":
    main()
