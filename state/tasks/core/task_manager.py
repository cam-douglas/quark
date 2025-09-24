"""
Task Manager Module
===================
Main orchestrator for task management operations.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from .roadmap_parser import RoadmapParser
from .task_generator import TaskGenerator
from .task_tracker import TaskTracker
from .task_executor import TaskExecutor


class TaskManager:
    """Main task management orchestrator."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tasks_root = self.project_root / "state" / "tasks"
        self.roadmap_tasks = self.tasks_root / "roadmap_tasks"
        self.validation_root = self.tasks_root / "validation"
        self.template_path = self.validation_root / "templates" / "TASK_TEMPLATE.md"
        self.roadmap_rules = self.project_root / "management" / "rules" / "roadmap"
        
        # Initialize components
        self.parser = RoadmapParser(self.roadmap_rules)
        self.generator = TaskGenerator(self.template_path, self.roadmap_tasks)
        self.tracker = TaskTracker(self.roadmap_tasks)
        self.executor = TaskExecutor()
        
        # Task categories
        self.categories = [
            "foundation_layer",
            "cerebellum",
            "developmental_biology",
            "brainstem_segmentation"
        ]
    
    def generate_from_roadmap(self, stage: Optional[int] = None, 
                            category: Optional[str] = None) -> None:
        """Generate tasks from roadmap milestones."""
        print("\n" + "=" * 60)
        print("🚀 GENERATING TASKS FROM ROADMAP")
        print("=" * 60)
        
        # Get roadmap file
        roadmap_file = self.parser.get_roadmap_file(stage, category)
        
        if not roadmap_file.exists():
            print(f"⚠️ Roadmap file not found: {roadmap_file}")
            return
        
        print(f"\n📖 Parsing: {roadmap_file.name}")
        
        # Parse roadmap
        milestones, kpis = self.parser.parse_roadmap_file(roadmap_file)
        
        if not milestones:
            print("⚠️ No milestones found in roadmap")
            return
        
        print(f"\n📋 Found {len(milestones)} milestones and {len(kpis)} KPIs")
        
        # Generate tasks
        generated_count = 0
        for milestone in milestones:
            task_file = self.generator.generate_from_milestone(milestone, kpis)
            if task_file:
                print(f"✅ Generated: {task_file.name}")
                generated_count += 1
        
        print(f"\n🎯 Generated {generated_count} task files in {self.roadmap_tasks}")
        self._print_next_steps()
    
    def sync_roadmap_status(self, update: bool = False) -> None:
        """Sync task status with roadmap documents."""
        print("\n" + "=" * 60)
        print("🔄 SYNCING WITH ROADMAP")
        print("=" * 60)
        
        task_statuses = self.tracker.sync_with_roadmap(update)
        
        if not task_statuses:
            print("📭 No task files found to sync")
            return
        
        print(f"\n📋 Found {len(task_statuses)} task files")
        
        # Display summary
        summary = self._get_status_summary(task_statuses)
        print("\n📊 Task Status Summary:")
        print(f"  ✅ Completed: {summary['completed']}")
        print(f"  🔄 In Progress: {summary['in_progress']}")
        print(f"  ⏳ Pending: {summary['pending']}")
        
        if update:
            print("\n✅ Status report created: STATUS_REPORT.md")
    
    def plan_task(self, category: Optional[str] = None) -> None:
        """Interactive task planning guide."""
        print("\n" + "=" * 60)
        print("📋 QUARK TASK PLANNING GUIDE")
        print("=" * 60)
        
        # Collect task information
        if not category:
            category = self._select_category()
        
        task_info = self._collect_task_info()
        kpis = self._select_kpis()
        dependencies = self._collect_dependencies()
        risks = self._collect_risks()
        
        # Generate task
        task_path = self.generator.generate_from_input(
            category, task_info, kpis, dependencies, risks
        )
        
        print(f"\n✅ Task created: {task_path}")
        print("\n📝 Task planning complete!")
    
    def execute_task(self, task_id: Optional[str] = None) -> None:
        """Guide for executing current tasks."""
        print("\n" + "=" * 60)
        print("🚀 TASK EXECUTION GUIDE")
        print("=" * 60)
        
        if not task_id:
            task_id = self._select_active_task()
        
        print(f"\n📌 Executing Task: {task_id}")
        
        # Show requirements
        print("\n📋 Task Requirements:")
        print("-" * 40)
        for req in self.executor.get_task_requirements():
            print(f"  • {req}")
        
        # Guide through P/F/A/O
        print("\n🔄 P/F/A/O Execution Stages:")
        print("-" * 40)
        for stage, info in self.executor.get_pfao_guidance().items():
            print(f"\n{stage}: {info['name']}")
            for step in info['steps']:
                print(f"  □ {step}")
        
        # Link validation
        print("\n🔗 Validation Integration:")
        print("-" * 40)
        print(f"\nTo validate task '{task_id}':")
        for step in self.executor.get_validation_steps(task_id):
            print(f"  {step}")
        
        # Reminders
        print("\n📝 Remember to:")
        for reminder in self.executor.get_execution_reminders():
            print(f"  {reminder}")
    
    def track_progress(self, status: Optional[str] = None) -> None:
        """Track progress on active tasks."""
        print("\n" + "=" * 60)
        print("📊 TASK PROGRESS TRACKER")
        print("=" * 60)
        
        tasks = self.tracker.load_tasks(status)
        
        if not tasks:
            print("\n📭 No active tasks found.")
            return
        
        print(f"\n📋 Active Tasks ({len(tasks)} total):\n")
        
        for task in tasks:
            self._display_task_progress(task)
        
        # Summary
        summary = self.tracker.get_task_summary(tasks)
        print(f"\n📊 Overall Progress: {summary['avg_progress']}%")
        print(f"   Tasks: {len(tasks)} active")
    
    def review_tasks(self) -> None:
        """Review completed tasks."""
        print("\n" + "=" * 60)
        print("✅ COMPLETED TASK REVIEW")
        print("=" * 60)
        
        completed_tasks = self.tracker.load_tasks("COMPLETED")
        
        if not completed_tasks:
            print("\n📭 No completed tasks to review.")
            return
        
        print(f"\n🎯 Completed Tasks ({len(completed_tasks)} total):\n")
        
        for task in completed_tasks:
            print(f"✅ {task['id']}: {task['name']}")
            print(f"   Category: {task['category']} | Priority: {task['priority']}")
        
        print(f"\n📊 Validation Summary:")
        print(f"   Total Completed: {len(completed_tasks)}")
        print(f"   Ready for validation: make validate-quick")
    
    def create_task(self, template: Optional[str] = None) -> None:
        """Create new task from template."""
        print("\n" + "=" * 60)
        print("📝 CREATE NEW TASK")
        print("=" * 60)
        
        # Interactive creation
        print("\nFill in the following fields:")
        
        fields = {
            "category": input("Category: ").strip(),
            "task_id": input("Task ID: ").strip(),
            "name": input("Task Name: ").strip(),
            "description": input("Description: ").strip(),
            "priority": input("Priority (Critical/High/Medium/Low): ").strip(),
            "estimated_effort": input("Estimated Effort: ").strip()
        }
        
        # Generate task
        task_path = self.generator.generate_from_input(
            fields["category"],
            fields,
            [],  # KPIs
            {"upstream": [], "downstream": []},  # Dependencies
            []  # Risks
        )
        
        print(f"\n✅ Task created: {task_path}")
        self._print_next_steps()
    
    def list_tasks(self, filter_by: Optional[str] = None) -> None:
        """List all tasks by category/status."""
        print("\n" + "=" * 60)
        print("📚 TASK INVENTORY")
        print("=" * 60)
        
        all_tasks = self.tracker.load_tasks()
        
        if filter_by:
            all_tasks = self._filter_tasks(all_tasks, filter_by)
        
        # Group by category
        by_category = self.tracker.group_by_category(all_tasks)
        
        # Display
        for category, tasks in by_category.items():
            print(f"\n📁 {category.upper()} ({len(tasks)} tasks)")
            print("-" * 40)
            
            for task in tasks:
                icon = self._get_status_icon(task.get("status", "PENDING"))
                print(f"{icon} {task['id']}: {task['name']}")
                print(f"   Phase: {task.get('phase', 'N/A')} | Priority: {task.get('priority', 'N/A')}")
        
        # Summary
        summary = self.tracker.get_task_summary(all_tasks)
        print(f"\n📊 Summary: {summary['total']} total | "
              f"{summary['pending']} pending | {summary['in_progress']} in progress | "
              f"{summary['completed']} completed")
    
    # Helper methods
    def _select_category(self) -> str:
        """Interactive category selection."""
        print("\n📂 Select Task Category:")
        for i, cat in enumerate(self.categories, 1):
            print(f"  {i}. {cat.replace('_', ' ').title()}")
        
        while True:
            try:
                choice = int(input("\nEnter choice (1-{}): ".format(len(self.categories))))
                if 1 <= choice <= len(self.categories):
                    return self.categories[choice - 1]
            except (ValueError, KeyboardInterrupt):
                print("\n⚠️ Cancelled")
                sys.exit(0)
    
    def _collect_task_info(self) -> Dict:
        """Collect basic task information."""
        print("\n📝 Define Task:")
        return {
            "name": input("Task Name: ").strip(),
            "description": input("Brief Description: ").strip(),
            "priority": input("Priority (Critical/High/Medium/Low): ").strip(),
            "estimated_effort": input("Estimated Effort (hours/days/weeks): ").strip()
        }
    
    def _select_kpis(self) -> List[Dict]:
        """Select KPIs for task."""
        print("\n📊 Select KPIs:")
        
        available_kpis = self.parser.get_available_kpis()
        
        for i, kpi in enumerate(available_kpis, 1):
            print(f"  {i}. {kpi['name']}: {kpi['target']} - {kpi['description']}")
        
        print("\n  0. Skip KPI selection")
        selected = input("\nSelect KPIs (comma-separated numbers): ").strip()
        
        if selected == "0" or not selected:
            return []
        
        kpis = []
        try:
            indices = [int(x.strip()) for x in selected.split(",")]
            for idx in indices:
                if 1 <= idx <= len(available_kpis):
                    kpi = available_kpis[idx - 1]
                    kpis.append({
                        "name": kpi["name"],
                        "target": kpi["target"],
                        "measurement": kpi["description"]
                    })
        except ValueError:
            print("⚠️ Invalid selection")
        
        return kpis
    
    def _collect_dependencies(self) -> Dict[str, List[str]]:
        """Collect task dependencies."""
        print("\n🔗 Dependencies (press Enter to skip):")
        
        deps = {"upstream": [], "downstream": []}
        
        print("\nUpstream (what this needs):")
        while True:
            dep = input("  Add dependency: ").strip()
            if not dep:
                break
            deps["upstream"].append(dep)
        
        print("\nDownstream (what needs this):")
        while True:
            dep = input("  Add consumer: ").strip()
            if not dep:
                break
            deps["downstream"].append(dep)
        
        return deps
    
    def _collect_risks(self) -> List[Dict]:
        """Collect risk assessment."""
        print("\n⚠️ Risks (press Enter to skip):")
        
        risks = []
        while True:
            risk = input("\nRisk description: ").strip()
            if not risk:
                break
            
            risks.append({
                "risk": risk,
                "probability": input("Probability (H/M/L): ").strip(),
                "impact": input("Impact (H/M/L): ").strip(),
                "mitigation": input("Mitigation: ").strip()
            })
        
        return risks
    
    def _select_active_task(self) -> str:
        """Select from active tasks."""
        tasks = self.tracker.load_tasks("IN_PROGRESS")
        
        if not tasks:
            return input("\nEnter task ID: ").strip()
        
        print("\n📌 Select Active Task:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task['id']}: {task['name']}")
        
        try:
            choice = int(input("\nEnter number: "))
            if 1 <= choice <= len(tasks):
                return tasks[choice - 1]["id"]
        except ValueError:
            pass
        
        return input("Enter task ID: ").strip()
    
    def _display_task_progress(self, task: Dict) -> None:
        """Display task progress."""
        icon = self._get_status_icon(task.get("status", "PENDING"))
        progress = task.get("progress", 0)
        progress_bar = self._create_progress_bar(progress)
        
        print(f"{icon} {task['id']}: {task['name']}")
        print(f"   Phase: {task.get('phase', 'N/A')} | Progress: {progress_bar} {progress}%")
    
    def _get_status_icon(self, status: str) -> str:
        """Get status icon."""
        icons = {
            "PENDING": "⏳",
            "IN_PROGRESS": "🔄",
            "COMPLETED": "✅",
            "BLOCKED": "🚫"
        }
        return icons.get(status, "❓")
    
    def _create_progress_bar(self, progress: int) -> str:
        """Create progress bar."""
        filled = "█" * (progress // 10)
        empty = "░" * ((100 - progress) // 10)
        return f"[{filled}{empty}]"
    
    def _filter_tasks(self, tasks: List[Dict], filter_by: str) -> List[Dict]:
        """Filter tasks by criteria."""
        if filter_by in ["PENDING", "IN_PROGRESS", "COMPLETED", "BLOCKED"]:
            return [t for t in tasks if t.get("status") == filter_by]
        elif filter_by in self.categories:
            return [t for t in tasks if filter_by in t.get("category", "")]
        return tasks
    
    def _get_status_summary(self, task_statuses: Dict[str, str]) -> Dict:
        """Get status summary from task statuses."""
        completed = sum(1 for s in task_statuses.values() if s == "COMPLETED")
        in_progress = sum(1 for s in task_statuses.values() if s == "IN_PROGRESS")
        pending = sum(1 for s in task_statuses.values() if s == "PENDING")
        
        return {
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending
        }
    
    def _print_next_steps(self) -> None:
        """Print next steps guidance."""
        print("\n📝 Next steps:")
        print("  1. Review generated tasks in state/tasks/roadmap_tasks/")
        print("  2. Refine task details as needed")
        print("  3. Execute tasks with 'python quark_tasks.py execute'")
        print("  4. Track progress with 'python quark_tasks.py track'")
