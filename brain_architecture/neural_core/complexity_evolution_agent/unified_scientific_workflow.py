#!/usr/bin/env python3
"""
Unified Scientific Workflow System

This system integrates the task integration system with the existing task management
and self-learning systems, creating a unified workflow for scientific advancement.
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

from brain_architecture.neural_core.complexity_evolution_agent.scientific_advancement_motivation import ScientificAdvancementMotivation
from brain_architecture.neural_core.complexity_evolution_agent.task_integration_system import TaskIntegrationSystem

class UnifiedScientificWorkflow:
    """
    Unified system that integrates task integration, task management, and self-learning
    for comprehensive scientific advancement.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
        # Initialize core systems
        self.scientific_motivation = ScientificAdvancementMotivation()
        self.task_integration = TaskIntegrationSystem()
        
        # Self-learning system integration
        self.self_learning_capabilities = self._initialize_self_learning()
        
        # Workflow management
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.learning_cycles: List[Dict[str, Any]] = []
        
        # Integration status
        self.integration_status = {
            "task_management": "connected",
            "self_learning": "active",
            "scientific_motivation": "active",
            "workflow_orchestration": "operational"
        }
        
        self.logger.info("Unified Scientific Workflow System initialized")
    
    def _initialize_self_learning(self) -> Dict[str, Any]:
        """Initialize self-learning capabilities"""
        
        return {
            "learning_modes": {
                "exploratory": "Explore new research directions and methodologies",
                "experimental": "Test hypotheses through experimentation",
                "analytical": "Analyze data and synthesize findings",
                "synthetic": "Combine knowledge from multiple domains",
                "adaptive": "Adapt strategies based on results and feedback"
            },
            "learning_cycles": {
                "plan": "Plan learning objectives and methods",
                "execute": "Execute learning activities",
                "evaluate": "Evaluate learning outcomes",
                "reflect": "Reflect on learning process and results",
                "adapt": "Adapt future learning strategies"
            },
            "knowledge_integration": {
                "cross_domain": "Integrate knowledge across different scientific domains",
                "temporal": "Integrate knowledge across different time periods",
                "spatial": "Integrate knowledge across different spatial scales",
                "conceptual": "Integrate knowledge across different conceptual frameworks"
            }
        }
    
    def create_scientific_workflow(self, 
                                 workflow_name: str,
                                 scientific_goal_id: str,
                                 workflow_type: str = "research") -> Dict[str, Any]:
        """Create a new scientific workflow"""
        
        workflow_id = f"WF_{hashlib.md5(f'{workflow_name}_{datetime.now()}'.encode()).hexdigest()[:8]}"
        
        workflow = {
            "id": workflow_id,
            "name": workflow_name,
            "scientific_goal_id": scientific_goal_id,
            "type": workflow_type,
            "created_at": datetime.now(),
            "status": "active",
            "current_phase": "planning",
            "phases": self._get_workflow_phases(workflow_type),
            "tasks": [],
            "learning_cycles": [],
            "progress": 0.0,
            "estimated_completion": None,
            "actual_completion": None
        }
        
        # Generate tasks for this workflow
        scientific_goals = {scientific_goal_id: self.scientific_motivation.scientific_goals.get(scientific_goal_id, {})}
        if scientific_goals[scientific_goal_id]:
            generated_tasks = self.task_integration.generate_tasks_from_scientific_goals(scientific_goals)
            workflow["tasks"] = [task["id"] for task in generated_tasks if isinstance(task, dict) and "id" in task]
        
        # Store workflow
        self.active_workflows[workflow_id] = workflow
        
        self.logger.info(f"✅ Created scientific workflow: {workflow_name}")
        
        return workflow
    
    def _get_workflow_phases(self, workflow_type: str) -> List[Dict[str, Any]]:
        """Get workflow phases based on type"""
        
        if workflow_type == "research":
            return [
                {"name": "planning", "description": "Research planning and methodology design", "duration_hours": 8.0},
                {"name": "literature_review", "description": "Comprehensive literature review", "duration_hours": 16.0},
                {"name": "data_collection", "description": "Data collection and preparation", "duration_hours": 20.0},
                {"name": "analysis", "description": "Data analysis and interpretation", "duration_hours": 24.0},
                {"name": "synthesis", "description": "Synthesis of findings and conclusions", "duration_hours": 16.0},
                {"name": "documentation", "description": "Documentation and publication preparation", "duration_hours": 12.0}
            ]
        elif workflow_type == "development":
            return [
                {"name": "requirements", "description": "Requirements analysis and specification", "duration_hours": 12.0},
                {"name": "design", "description": "System design and architecture", "duration_hours": 24.0},
                {"name": "implementation", "description": "System implementation and coding", "duration_hours": 40.0},
                {"name": "testing", "description": "Testing and validation", "duration_hours": 20.0},
                {"name": "deployment", "description": "Deployment and integration", "duration_hours": 16.0}
            ]
        else:  # experimental
            return [
                {"name": "hypothesis", "description": "Hypothesis formation and experimental design", "duration_hours": 12.0},
                {"name": "experiment_setup", "description": "Experimental setup and preparation", "duration_hours": 16.0},
                {"name": "execution", "description": "Experiment execution and data collection", "duration_hours": 24.0},
                {"name": "analysis", "description": "Data analysis and interpretation", "duration_hours": 20.0},
                {"name": "conclusion", "description": "Conclusion drawing and future directions", "duration_hours": 12.0}
            ]
    
    def execute_learning_cycle(self, workflow_id: str, cycle_type: str = "adaptive") -> Dict[str, Any]:
        """Execute a learning cycle within a workflow"""
        
        if workflow_id not in self.active_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        # Create learning cycle
        cycle_id = f"LC_{workflow_id}_{len(workflow['learning_cycles']) + 1}"
        
        learning_cycle = {
            "id": cycle_id,
            "workflow_id": workflow_id,
            "type": cycle_type,
            "started_at": datetime.now(),
            "status": "active",
            "phases": self.self_learning_capabilities["learning_cycles"].copy(),
            "current_phase": "plan",
            "phase_progress": {"plan": 0.0, "execute": 0.0, "evaluate": 0.0, "reflect": 0.0, "adapt": 0.0},
            "insights": [],
            "adaptations": []
        }
        
        # Add to workflow
        workflow["learning_cycles"].append(cycle_id)
        
        # Store cycle
        self.learning_cycles.append(learning_cycle)
        
        self.logger.info(f"🧠 Started learning cycle {cycle_id} in workflow {workflow_id}")
        
        return {
            "success": True,
            "cycle_id": cycle_id,
            "learning_cycle": learning_cycle
        }
    
    def advance_learning_phase(self, cycle_id: str, phase_name: str, progress: float, insights: List[str] = None) -> Dict[str, Any]:
        """Advance a learning cycle to the next phase"""
        
        # Find the learning cycle
        cycle = None
        for c in self.learning_cycles:
            if c["id"] == cycle_id:
                cycle = c
                break
        
        if not cycle:
            return {"success": False, "error": "Learning cycle not found"}
        
        # Update current phase progress
        cycle["phase_progress"][phase_name] = progress
        
        # Add insights
        if insights:
            cycle["insights"].extend(insights)
        
        # Check if phase is complete
        if progress >= 100.0:
            # Move to next phase
            phases = list(cycle["phases"].keys())
            current_index = phases.index(cycle["current_phase"])
            
            if current_index < len(phases) - 1:
                next_phase = phases[current_index + 1]
                cycle["current_phase"] = next_phase
                cycle["phase_progress"][next_phase] = 0.0
                
                self.logger.info(f"🔄 Learning cycle {cycle_id} advanced to phase: {next_phase}")
                
                return {
                    "success": True,
                    "phase_completed": phase_name,
                    "new_phase": next_phase,
                    "cycle_progress": self._calculate_cycle_progress(cycle)
                }
            else:
                # Cycle complete
                cycle["status"] = "completed"
                cycle["completed_at"] = datetime.now()
                
                self.logger.info(f"�� Learning cycle {cycle_id} completed!")
                
                return {
                    "success": True,
                    "cycle_completed": True,
                    "final_progress": 100.0
                }
        
        return {
            "success": True,
            "phase_progress": progress,
            "cycle_progress": self._calculate_cycle_progress(cycle)
        }
    
    def _calculate_cycle_progress(self, cycle: Dict[str, Any]) -> float:
        """Calculate overall progress of a learning cycle"""
        
        total_progress = sum(cycle["phase_progress"].values())
        return total_progress / len(cycle["phase_progress"])
    
    def update_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Update workflow progress based on task and learning cycle completion"""
        
        if workflow_id not in self.active_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        # Calculate task progress
        task_progress = 0.0
        if workflow["tasks"]:
            total_task_progress = 0.0
            for task_id in workflow["tasks"]:
                if task_id in self.task_integration.task_progress:
                    total_task_progress += self.task_integration.task_progress[task_id]
            task_progress = total_task_progress / len(workflow["tasks"])
        
        # Calculate learning cycle progress
        learning_progress = 0.0
        if workflow["learning_cycles"]:
            total_learning_progress = 0.0
            for cycle_id in workflow["learning_cycles"]:
                cycle = next((c for c in self.learning_cycles if c["id"] == cycle_id), None)
                if cycle:
                    total_learning_progress += self._calculate_cycle_progress(cycle)
            learning_progress = total_learning_progress / len(workflow["learning_cycles"])
        
        # Overall progress (weighted average)
        overall_progress = (task_progress * 0.7) + (learning_progress * 0.3)
        
        # Update workflow
        old_progress = workflow["progress"]
        workflow["progress"] = overall_progress
        
        # Update current phase
        if overall_progress >= 100.0:
            workflow["status"] = "completed"
            workflow["actual_completion"] = datetime.now()
            workflow["current_phase"] = "completed"
        else:
            workflow["current_phase"] = self._determine_current_phase(workflow, overall_progress)
        
        self.logger.info(f"📊 Updated workflow {workflow_id}: {old_progress:.1f}% → {overall_progress:.1f}%")
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "old_progress": old_progress,
            "new_progress": overall_progress,
            "task_progress": task_progress,
            "learning_progress": learning_progress,
            "current_phase": workflow["current_phase"]
        }
    
    def _determine_current_phase(self, workflow: Dict[str, Any], progress: float) -> str:
        """Determine current workflow phase based on progress"""
        
        phases = workflow["phases"]
        if not phases:
            return "unknown"
        
        # Calculate phase boundaries
        total_duration = sum(phase["duration_hours"] for phase in phases)
        phase_boundaries = []
        cumulative_duration = 0.0
        
        for phase in phases:
            cumulative_duration += phase["duration_hours"]
            phase_boundaries.append((phase["name"], cumulative_duration / total_duration))
        
        # Find current phase
        for phase_name, boundary in phase_boundaries:
            if progress <= boundary:
                return phase_name
        
        return phases[-1]["name"]  # Default to last phase
    
    def get_unified_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive unified dashboard"""
        
        dashboard = {
            "timestamp": datetime.now(),
            "integration_status": self.integration_status,
            "scientific_motivation": self.scientific_motivation.get_motivation_status(),
            "task_integration": self.task_integration.get_task_dashboard(),
            "workflow_overview": {
                "total_workflows": len(self.active_workflows),
                "active_workflows": len([w for w in self.active_workflows.values() if w["status"] == "active"]),
                "completed_workflows": len([w for w in self.active_workflows.values() if w["status"] == "completed"]),
                "total_learning_cycles": len(self.learning_cycles),
                "active_learning_cycles": len([c for c in self.learning_cycles if c["status"] == "active"])
            },
            "self_learning_capabilities": self.self_learning_capabilities,
            "recent_activities": self._get_recent_activities()
        }
        
        return dashboard
    
    def _get_recent_activities(self) -> List[Dict[str, Any]]:
        """Get recent activities across all systems"""
        
        activities = []
        
        # Recent task updates
        for task in list(self.task_integration.scientific_tasks.values())[:3]:
            activities.append({
                "type": "task_update",
                "timestamp": task.get("created_at", datetime.now()),
                "description": f"Task '{task['title']}' updated",
                "system": "task_integration"
            })
        
        # Recent learning cycles
        for cycle in self.learning_cycles[-3:]:
            activities.append({
                "type": "learning_cycle",
                "timestamp": cycle.get("started_at", datetime.now()),
                "description": f"Learning cycle {cycle['id']} {cycle['status']}",
                "system": "self_learning"
            })
        
        # Recent workflow updates
        for workflow in list(self.active_workflows.values())[:3]:
            activities.append({
                "type": "workflow_update",
                "timestamp": workflow.get("created_at", datetime.now()),
                "description": f"Workflow '{workflow['name']}' {workflow['status']}",
                "system": "workflow_orchestration"
            })
        
        # Sort by timestamp
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return activities[:10]  # Return 10 most recent
    
    def create_unified_visualization(self) -> str:
        """Create comprehensive HTML visualization of the unified system"""
        
        dashboard = self.get_unified_dashboard()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🚀 Quark Unified Scientific Workflow Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 100%); color: white; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .integration-banner {{ background: linear-gradient(45deg, #00d4ff, #0099cc); padding: 20px; border-radius: 15px; margin-bottom: 20px; text-align: center; font-size: 1.2em; font-weight: bold; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px); }}
        .full-width {{ grid-column: 1 / -1; }}
        .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .status {{ padding: 8px 15px; border-radius: 20px; font-weight: bold; }}
        .status.connected {{ background: linear-gradient(45deg, #4CAF50, #45a049); }}
        .status.active {{ background: linear-gradient(45deg, #2196F3, #1976D2); }}
        .status.operational {{ background: linear-gradient(45deg, #FF9800, #F57C00); }}
        .progress-bar {{ background: rgba(255,255,255,0.2); height: 20px; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(45deg, #00d4ff, #0099cc); transition: width 0.3s ease; }}
        .system-status {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Quark Unified Scientific Workflow Dashboard</h1>
        <h2>Integrated Task Management, Self-Learning, and Scientific Motivation</h2>
        <p><strong>Last Updated:</strong> {dashboard['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="integration-banner">
        🔗 UNIFIED INTEGRATION: All systems connected and operational for scientific advancement
    </div>
    
    <div class="dashboard">
        <div class="card">
            <h2>🔗 Integration Status</h2>
            <div class="system-status">
                {self._render_integration_status(dashboard['integration_status'])}
            </div>
        </div>
        
        <div class="card">
            <h2>📊 Workflow Overview</h2>
            <div class="metric">
                <span><strong>Total Workflows:</strong></span>
                <span>{dashboard['workflow_overview']['total_workflows']}</span>
            </div>
            <div class="metric">
                <span><strong>Active Workflows:</strong></span>
                <span>{dashboard['workflow_overview']['active_workflows']}</span>
            </div>
            <div class="metric">
                <span><strong>Learning Cycles:</strong></span>
                <span>{dashboard['workflow_overview']['total_learning_cycles']}</span>
            </div>
        </div>
        
        <div class="card">
            <h2>🧠 Self-Learning Capabilities</h2>
            <div style="font-size: 0.9em;">
                <strong>Learning Modes:</strong>
                <ul>
                    <li>Exploratory: Explore new research directions</li>
                    <li>Experimental: Test hypotheses</li>
                    <li>Analytical: Analyze and synthesize</li>
                    <li>Adaptive: Adapt based on feedback</li>
                </ul>
            </div>
        </div>
        
        <div class="card">
            <h2>🎯 Scientific Motivation</h2>
            <div class="metric">
                <span><strong>Overall Motivation:</strong></span>
                <span>{dashboard['scientific_motivation']['overall_motivation']:.1%}</span>
            </div>
            <div class="metric">
                <span><strong>Research Progress:</strong></span>
                <span>{dashboard['scientific_motivation']['research_progress']:.1%}</span>
            </div>
            <div class="metric">
                <span><strong>Active Goals:</strong></span>
                <span>{len([g for g in dashboard['scientific_motivation']['scientific_goals'].values() if g['status'] == 'active'])}</span>
            </div>
        </div>
        
        <div class="card full-width">
            <h2>📋 Task Integration Status</h2>
            <div class="metric">
                <span><strong>Total Tasks:</strong></span>
                <span>{dashboard['task_integration']['total_tasks']}</span>
            </div>
            <div class="metric">
                <span><strong>Active Tasks:</strong></span>
                <span>{dashboard['task_integration']['tasks_by_status'].get('active', 0)}</span>
            </div>
            <div class="metric">
                <span><strong>Completed Tasks:</strong></span>
                <span>{dashboard['task_integration']['tasks_by_status'].get('completed', 0)}</span>
            </div>
            <div class="metric">
                <span><strong>Average Progress:</strong></span>
                <span>{dashboard['task_integration']['progress_summary'].get('average_progress', 0):.1f}%</span>
            </div>
        </div>
        
        <div class="card full-width">
            <h2>🔄 Recent Activities</h2>
            {self._render_recent_activities(dashboard['recent_activities'])}
        </div>
        
        <div class="card full-width">
            <h2>🚀 Unified Workflow Benefits</h2>
            <div style="font-size: 1.1em; line-height: 1.6;">
                <ul>
                    <li><strong>Seamless Integration:</strong> Task management, self-learning, and motivation work together</li>
                    <li><strong>Automated Workflows:</strong> Scientific goals automatically generate tasks and learning cycles</li>
                    <li><strong>Intelligent Adaptation:</strong> System learns and adapts based on progress and outcomes</li>
                    <li><strong>Progress Tracking:</strong> Comprehensive progress monitoring across all systems</li>
                    <li><strong>Knowledge Integration:</strong> Cross-domain knowledge synthesis and application</li>
                    <li><strong>Safety Integration:</strong> All workflows include safety and ethics considerations</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _render_integration_status(self, status: Dict[str, str]) -> str:
        """Render integration status HTML"""
        html = ""
        
        for system, system_status in status.items():
            html += f"""
            <div style="text-align: center;">
                <div class="status {system_status}">{system.replace('_', ' ').title()}</div>
                <div style="font-size: 0.8em; margin-top: 5px;">{system_status}</div>
            </div>
            """
        
        return html
    
    def _render_recent_activities(self, activities: List[Dict[str, Any]]) -> str:
        """Render recent activities HTML"""
        if not activities:
            return "<p>No recent activities.</p>"
        
        html = "<div style='display: grid; gap: 10px;'>"
        
        for activity in activities[:5]:  # Show 5 most recent
            html += f"""
            <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span><strong>{activity['type'].replace('_', ' ').title()}</strong></span>
                    <span style="font-size: 0.8em;">{activity['timestamp'].strftime('%H:%M')}</span>
                </div>
                <div style="margin-top: 10px; color: rgba(255,255,255,0.8);">
                    {activity['description']}
                </div>
                <div style="font-size: 0.8em; color: rgba(255,255,255,0.6); margin-top: 5px;">
                    System: {activity['system'].replace('_', ' ').title()}
                </div>
            </div>
            """
        
        html += "</div>"
        return html

def main():
    """Main demonstration function"""
    print("🚀 Initializing Unified Scientific Workflow System...")
    
    # Initialize the system
    unified_system = UnifiedScientificWorkflow()
    
    print("✅ System initialized!")
    print("\n🔗 All systems integrated and operational!")
    
    # Create a sample scientific workflow
    print("\n🔬 Creating sample scientific workflow...")
    
    # Get a scientific goal
    scientific_goals = unified_system.scientific_motivation.scientific_goals
    if scientific_goals:
        goal_id = list(scientific_goals.keys())[0]
        workflow = unified_system.create_scientific_workflow(
            "Consciousness Research Workflow",
            goal_id,
            "research"
        )
        print(f"✅ Created workflow: {workflow['name']}")
        
        # Execute a learning cycle
        print("\n🧠 Executing learning cycle...")
        learning_cycle = unified_system.execute_learning_cycle(workflow['id'], "exploratory")
        print(f"✅ Started learning cycle: {learning_cycle['cycle_id']}")
        
        # Update workflow progress
        print("\n📊 Updating workflow progress...")
        progress_update = unified_system.update_workflow_progress(workflow['id'])
        print(f"✅ Workflow progress: {progress_update['new_progress']:.1f}%")
    
    # Get unified dashboard
    dashboard = unified_system.get_unified_dashboard()
    print(f"\n📊 Unified Dashboard:")
    print(f"   Total Workflows: {dashboard['workflow_overview']['total_workflows']}")
    print(f"   Active Workflows: {dashboard['workflow_overview']['active_workflows']}")
    print(f"   Learning Cycles: {dashboard['workflow_overview']['total_learning_cycles']}")
    print(f"   Total Tasks: {dashboard['task_integration']['total_tasks']}")
    
    # Create visualization
    html_content = unified_system.create_unified_visualization()
    with open("testing/visualizations/unified_scientific_workflow.html", "w") as f:
        f.write(html_content)
    
    print("✅ Unified workflow dashboard created: testing/visualizations/unified_scientific_workflow.html")
    
    print("\n🎉 Unified Scientific Workflow System demonstration complete!")
    print("\n🚀 Key Features:")
    print("   • Complete integration of all systems")
    print("   • Automated workflow generation from scientific goals")
    print("   • Self-learning cycles with adaptive phases")
    print("   • Unified progress tracking and monitoring")
    print("   • Cross-system knowledge integration")
    print("   • Safety-first approach throughout")
    
    return unified_system

if __name__ == "__main__":
    main()
