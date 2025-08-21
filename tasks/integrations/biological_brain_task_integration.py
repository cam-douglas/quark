#!/usr/bin/env python3
"""
🧠 Biological Brain Agent - Task Management Integration

This module provides deep integration between the biological brain agent and the central task management system.
It ensures that the brain agent can read, analyze, and update tasks while maintaining biological constraints
and cognitive resource management.
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiologicalBrainTaskIntegration:
    """Integrates biological brain agent with central task management system"""
    
    def __init__(self, brain_agent_path: str = "brain_architecture/neural_core/biological_brain_agent.py", 
                 task_system_path: str = "tasks"):
        self.brain_agent_path = Path(brain_agent_path)
        self.task_system_path = Path(task_system_path)
        self.integration_status = "initialized"
        self.last_sync = datetime.now()
        
        # Task management state
        self.central_tasks = {}
        self.brain_task_mapping = {}
        self.task_priorities = {}
        self.task_dependencies = {}
        
        # Biological constraints
        self.biological_constraints = {
            "max_cognitive_load": 0.8,
            "min_working_memory": 0.3,
            "min_energy_level": 0.4,
            "max_concurrent_tasks": 3,
            "task_switching_cost": 0.1
        }
        
        # Integration metrics
        self.integration_metrics = {
            "tasks_analyzed": 0,
            "brain_decisions_made": 0,
            "task_updates_sent": 0,
            "biological_constraints_violated": 0,
            "last_health_check": datetime.now()
        }
        
        logger.info("🧠 Biological Brain Task Integration initialized")
    
    def load_central_task_system(self) -> Dict[str, Any]:
        """Load the central task management system"""
        try:
            # Load main task status
            task_status_file = self.task_system_path / "TASK_STATUS.md"
            if task_status_file.exists():
                self.central_tasks = self._parse_task_status_markdown(task_status_file.read_text())
                logger.info(f"Loaded {len(self.central_tasks)} tasks from central system")
            else:
                logger.warning("Central task status file not found")
                self.central_tasks = {}
            
            # Load active tasks
            active_tasks_dir = self.task_system_path / "active_tasks"
            if active_tasks_dir.exists():
                self._load_active_tasks(active_tasks_dir)
            
            # Load dependencies
            dependencies_dir = self.task_system_path / "dependencies"
            if dependencies_dir.exists():
                self._load_task_dependencies(dependencies_dir)
            
            return self.central_tasks
            
        except Exception as e:
            logger.error(f"Error loading central task system: {e}")
            return {}
    
    def _parse_task_status_markdown(self, markdown_content: str) -> Dict[str, Any]:
        """Parse the main TASK_STATUS.md file to extract task information"""
        tasks = {}
        current_task = None
        
        lines = markdown_content.split('\n')
        for line in lines:
            # Look for task headers
            if line.startswith('### **Task #'):
                if current_task:
                    tasks[current_task['id']] = current_task
                
                # Extract task ID and title
                task_match = re.search(r'Task #(\d+): (.+)', line)
                if task_match:
                    current_task = {
                        'id': f"task_{task_match.group(1)}",
                        'title': task_match.group(2).strip(),
                        'status': 'unknown',
                        'priority': 'unknown',
                        'progress': 0,
                        'due_date': None,
                        'location': None,
                        'owner': None,
                        'acceptance_criteria': [],
                        'next_steps': []
                    }
            
            # Extract task details
            if current_task:
                if '**Status**' in line:
                    status_match = re.search(r'**Status**:\s*([^\\n]+)', line)
                    if status_match:
                        current_task['status'] = status_match.group(1).strip()
                
                elif '**Due Date**' in line:
                    due_match = re.search(r'**Due Date**:\s*([^\\n]+)', line)
                    if due_match:
                        current_task['due_date'] = due_match.group(1).strip()
                
                elif '**Progress**' in line:
                    progress_match = re.search(r'**Progress**:\s*(\d+)%', line)
                    if progress_match:
                        current_task['progress'] = int(progress_match.group(1))
                
                elif '**Location**' in line:
                    location_match = re.search(r'**Location**:\s*`([^`]+)`', line)
                    if location_match:
                        current_task['location'] = location_match.group(1)
                
                elif '**Owner**' in line:
                    owner_match = re.search(r'**Owner**:\s*([^\\n]+)', line)
                    if owner_match:
                        current_task['owner'] = owner_match.group(1).strip()
                
                elif line.strip().startswith('- [ ]'):
                    current_task['acceptance_criteria'].append(line.strip())
                
                elif line.strip().startswith('**Next Steps**'):
                    # Start collecting next steps
                    pass
                elif line.strip().startswith('1.') and current_task.get('next_steps') is not None:
                    current_task['next_steps'].append(line.strip())
        
        # Add the last task
        if current_task:
            tasks[current_task['id']] = current_task
        
        return tasks
    
    def _load_active_tasks(self, active_tasks_dir: Path):
        """Load active tasks from the active_tasks directory"""
        for priority_dir in ['high_priority', 'medium_priority', 'low_priority']:
            priority_path = active_tasks_dir / priority_dir
            if priority_path.exists():
                for task_file in priority_path.glob('*.md'):
                    if task_file.name != 'README.md':
                        self._load_single_active_task(task_file, priority_dir)
    
    def _load_single_active_task(self, task_file: Path, priority: str):
        """Load a single active task file"""
        try:
            content = task_file.read_text()
            task_id = task_file.stem
            
            # Parse task content
            task_info = {
                'id': task_id,
                'priority': priority,
                'file_path': str(task_file),
                'content': content
            }
            
            # Extract additional metadata
            if '**Status**' in content:
                status_match = re.search(r'**Status**:\s*([^\\n]+)', content)
                if status_match:
                    task_info['status'] = status_match.group(1).strip()
            
            self.central_tasks[task_id] = task_info
            
        except Exception as e:
            logger.error(f"Error loading active task {task_file}: {e}")
    
    def _load_task_dependencies(self, dependencies_dir: Path):
        """Load task dependencies and blockers"""
        blockers_file = dependencies_dir / "blockers.md"
        if blockers_file.exists():
            self._parse_blockers_file(blockers_file)
    
    def _parse_blockers_file(self, blockers_file: Path):
        """Parse the blockers.md file to understand task dependencies"""
        try:
            content = blockers_file.read_text()
            # Extract blocker information
            # This is a simplified parser - can be enhanced based on actual format
            logger.info(f"Loaded blockers from {blockers_file}")
        except Exception as e:
            logger.error(f"Error parsing blockers file: {e}")
    
    def analyze_tasks_for_brain(self) -> Dict[str, Any]:
        """Analyze central tasks and prepare them for brain agent consumption"""
        brain_task_analysis = {
            "task_summary": {},
            "priority_distribution": {},
            "resource_requirements": {},
            "biological_constraints": {},
            "recommendations": []
        }
        
        # Analyze each task
        for task_id, task in self.central_tasks.items():
            task_analysis = self._analyze_single_task(task)
            brain_task_analysis["task_summary"][task_id] = task_analysis
            
            # Update priority distribution
            priority = task_analysis.get('priority', 'unknown')
            if priority not in brain_task_analysis["priority_distribution"]:
                brain_task_analysis["priority_distribution"][priority] = 0
            brain_task_analysis["priority_distribution"][priority] += 1
        
        # Generate biological constraint analysis
        brain_task_analysis["biological_constraints"] = self._analyze_biological_constraints()
        
        # Generate recommendations
        brain_task_analysis["recommendations"] = self._generate_brain_recommendations()
        
        self.integration_metrics["tasks_analyzed"] = len(self.central_tasks)
        logger.info(f"Analyzed {len(self.central_tasks)} tasks for brain agent")
        
        return brain_task_analysis
    
    def _analyze_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single task for brain consumption"""
        analysis = {
            "id": task.get('id', 'unknown'),
            "title": task.get('title', 'Unknown Task'),
            "priority": self._extract_priority(task),
            "status": task.get('status', 'unknown'),
            "progress": task.get('progress', 0),
            "due_date": task.get('due_date'),
            "location": task.get('location'),
            "owner": task.get('owner'),
            "estimated_effort": self._estimate_task_effort(task),
            "cognitive_load": self._estimate_cognitive_load(task),
            "working_memory_required": self._estimate_working_memory(task),
            "biological_compatibility": self._assess_biological_compatibility(task)
        }
        
        return analysis
    
    def _extract_priority(self, task: Dict[str, Any]) -> str:
        """Extract priority from task data"""
        # Check multiple sources for priority
        if 'priority' in task:
            return task['priority']
        
        # Check title for priority indicators
        title = task.get('title', '').upper()
        if 'HIGH' in title or 'URGENT' in title or 'CRITICAL' in title:
            return 'high'
        elif 'MEDIUM' in title or 'NORMAL' in title:
            return 'medium'
        elif 'LOW' in title or 'OPTIONAL' in title:
            return 'low'
        
        # Default to medium
        return 'medium'
    
    def _estimate_task_effort(self, task: Dict[str, Any]) -> str:
        """Estimate effort required for task"""
        title = task.get('title', '').lower()
        content = task.get('content', '').lower()
        
        # Simple heuristic based on keywords
        if any(word in title or word in content for word in ['complete', 'implement', 'deploy', 'establish']):
            return 'high'
        elif any(word in title or word in content for word in ['update', 'modify', 'enhance']):
            return 'medium'
        else:
            return 'low'
    
    def _estimate_cognitive_load(self, task: Dict[str, Any]) -> float:
        """Estimate cognitive load for task"""
        effort = self._estimate_task_effort(task)
        effort_scores = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        
        base_load = effort_scores.get(effort, 0.5)
        
        # Adjust based on complexity indicators
        title = task.get('title', '').lower()
        if any(word in title for word in ['integration', 'framework', 'system']):
            base_load += 0.1
        
        return min(1.0, base_load)
    
    def _estimate_working_memory(self, task: Dict[str, Any]) -> float:
        """Estimate working memory required for task"""
        effort = self._estimate_task_effort(task)
        memory_scores = {'low': 0.1, 'medium': 0.3, 'high': 0.6}
        
        return memory_scores.get(effort, 0.3)
    
    def _assess_biological_compatibility(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Assess task compatibility with biological constraints"""
        cognitive_load = self._estimate_cognitive_load(task)
        working_memory = self._estimate_working_memory(task)
        
        compatibility = {
            "cognitive_load_ok": cognitive_load <= self.biological_constraints["max_cognitive_load"],
            "working_memory_ok": working_memory <= (1.0 - self.biological_constraints["min_working_memory"]),
            "overall_compatible": True
        }
        
        if not compatibility["cognitive_load_ok"] or not compatibility["working_memory_ok"]:
            compatibility["overall_compatible"] = False
            self.integration_metrics["biological_constraints_violated"] += 1
        
        return compatibility
    
    def _analyze_biological_constraints(self) -> Dict[str, Any]:
        """Analyze overall biological constraint status"""
        total_tasks = len(self.central_tasks)
        high_priority_tasks = len([t for t in self.central_tasks.values() 
                                 if self._extract_priority(t) == 'high'])
        
        return {
            "total_tasks": total_tasks,
            "high_priority_tasks": high_priority_tasks,
            "max_concurrent_tasks": self.biological_constraints["max_concurrent_tasks"],
            "current_cognitive_load": min(1.0, high_priority_tasks * 0.3),
            "constraints_met": total_tasks <= self.biological_constraints["max_concurrent_tasks"]
        }
    
    def _generate_brain_recommendations(self) -> List[str]:
        """Generate recommendations for the brain agent"""
        recommendations = []
        
        total_tasks = len(self.central_tasks)
        high_priority_tasks = len([t for t in self.central_tasks.values() 
                                 if self._extract_priority(t) == 'high'])
        
        if total_tasks > self.biological_constraints["max_concurrent_tasks"]:
            recommendations.append("Reduce concurrent task load to maintain biological constraints")
        
        if high_priority_tasks > 2:
            recommendations.append("Focus on highest priority tasks to avoid cognitive overload")
        
        if self.integration_metrics["biological_constraints_violated"] > 0:
            recommendations.append("Review tasks that violate biological constraints")
        
        if not recommendations:
            recommendations.append("Current task load is within biological constraints")
        
        return recommendations
    
    def send_brain_analysis_to_tasks(self, brain_analysis: Dict[str, Any]):
        """Send brain analysis results back to the task management system"""
        try:
            # Create brain analysis summary
            analysis_summary = self._create_brain_analysis_summary(brain_analysis)
            
            # Save to brain analysis directory
            brain_analysis_dir = self.task_system_path / "integrations" / "brain_analysis"
            brain_analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed analysis
            analysis_file = brain_analysis_dir / "brain_task_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(brain_analysis, f, indent=2, default=str)
            
            # Save summary
            summary_file = brain_analysis_dir / "BRAIN_ANALYSIS_SUMMARY.md"
            summary_file.write_text(analysis_summary)
            
            # Update integration status
            self.integration_metrics["task_updates_sent"] += 1
            self.last_sync = datetime.now()
            
            logger.info("Brain analysis sent to task management system")
            
        except Exception as e:
            logger.error(f"Error sending brain analysis to tasks: {e}")
    
    def _create_brain_analysis_summary(self, brain_analysis: Dict[str, Any]) -> str:
        """Create a markdown summary of brain analysis"""
        summary = f"""# 🧠 Brain Task Analysis Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Integration Status**: {self.integration_status}
**Last Sync**: {self.last_sync.strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Task Overview

**Total Tasks Analyzed**: {brain_analysis.get('task_summary', {}).__len__()}

### Priority Distribution
"""
        
        for priority, count in brain_analysis.get('priority_distribution', {}).items():
            summary += f"- **{priority.title()} Priority**: {count} tasks\n"
        
        summary += f"""
## 🧬 Biological Constraints Analysis

**Current Cognitive Load**: {brain_analysis.get('biological_constraints', {}).get('current_cognitive_load', 0):.2f}
**Max Concurrent Tasks**: {brain_analysis.get('biological_constraints', {}).get('max_concurrent_tasks', 0)}
**Constraints Met**: {brain_analysis.get('biological_constraints', {}).get('constraints_met', False)}

## 💡 Brain Recommendations

"""
        
        for recommendation in brain_analysis.get('recommendations', []):
            summary += f"- {recommendation}\n"
        
        summary += f"""
## 📈 Integration Metrics

- **Tasks Analyzed**: {self.integration_metrics['tasks_analyzed']}
- **Brain Decisions Made**: {self.integration_metrics['brain_decisions_made']}
- **Task Updates Sent**: {self.integration_metrics['task_updates_sent']}
- **Biological Constraints Violated**: {self.integration_metrics['biological_constraints_violated']}
- **Last Health Check**: {self.integration_metrics['last_health_check'].strftime('%Y-%m-%d %H:%M:%S')}

---
*Generated by Biological Brain Task Integration System*
"""
        
        return summary
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            "integration_status": self.integration_status,
            "last_sync": self.last_sync.isoformat(),
            "total_tasks": len(self.central_tasks),
            "integration_metrics": self.integration_metrics.copy(),
            "biological_constraints": self.biological_constraints.copy(),
            "brain_analysis_directory": str(self.task_system_path / "integrations" / "brain_analysis")
        }
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform health check on the integration"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "issues": [],
            "recommendations": []
        }
        
        # Check file accessibility
        if not self.brain_agent_path.exists():
            health_status["issues"].append("Biological brain agent file not found")
            health_status["overall_health"] = "degraded"
        
        if not self.task_system_path.exists():
            health_status["issues"].append("Task system directory not found")
            health_status["overall_health"] = "critical"
        
        # Check integration metrics
        if self.integration_metrics["biological_constraints_violated"] > 5:
            health_status["issues"].append("High number of biological constraint violations")
            health_status["overall_health"] = "degraded"
        
        # Check sync frequency
        time_since_sync = (datetime.now() - self.last_sync).total_seconds()
        if time_since_sync > 3600:  # 1 hour
            health_status["issues"].append("Integration sync is stale")
            health_status["recommendations"].append("Perform manual sync or check integration loop")
        
        # Update last health check
        self.integration_metrics["last_health_check"] = datetime.now()
        
        return health_status
    
    def start_integration_loop(self, interval_seconds: int = 30):
        """Start continuous integration loop"""
        def integration_loop():
            while True:
                try:
                    # Load central task system
                    self.load_central_task_system()
                    
                    # Analyze tasks for brain
                    brain_analysis = self.analyze_tasks_for_brain()
                    
                    # Send analysis back to task system
                    self.send_brain_analysis_to_tasks(brain_analysis)
                    
                    # Health check
                    health = self.perform_health_check()
                    if health["overall_health"] != "healthy":
                        logger.warning(f"Integration health check failed: {health['issues']}")
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in integration loop: {e}")
                    time.sleep(interval_seconds)
        
        # Start integration thread
        integration_thread = threading.Thread(target=integration_loop, daemon=True)
        integration_thread.start()
        
        self.integration_status = "running"
        logger.info(f"Integration loop started with {interval_seconds}s interval")
        
        return integration_thread

def main():
    """Main function to demonstrate the integration"""
    logger.info("🧠 Starting Biological Brain Task Integration")
    
    # Create integration
    integration = BiologicalBrainTaskIntegration()
    
    # Load central task system
    logger.info("Loading central task system...")
    tasks = integration.load_central_task_system()
    logger.info(f"Loaded {len(tasks)} tasks")
    
    # Analyze tasks for brain
    logger.info("Analyzing tasks for brain agent...")
    brain_analysis = integration.analyze_tasks_for_brain()
    
    # Send analysis to task system
    logger.info("Sending brain analysis to task system...")
    integration.send_brain_analysis_to_tasks(brain_analysis)
    
    # Get integration status
    status = integration.get_integration_status()
    logger.info(f"Integration Status: {status['integration_status']}")
    
    # Health check
    health = integration.perform_health_check()
    logger.info(f"Health Status: {health['overall_health']}")
    
    logger.info("✅ Biological Brain Task Integration demonstration complete!")
    
    return integration

if __name__ == "__main__":
    try:
        integration = main()
    except Exception as e:
        logger.error(f"❌ Integration demonstration failed: {e}")
        import traceback
        traceback.print_exc()
