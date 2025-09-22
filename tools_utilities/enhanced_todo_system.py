#!/usr/bin/env python3
"""
Enhanced TODO System with Task Completion Confidence

Integrates anti-overconfidence principles into task management.
Never marks tasks as complete without high confidence and clear evidence.

CORE PRINCIPLE: Assume tasks are incomplete until proven otherwise with substantial evidence.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import task completion validation
try:
    from .unified_validation_system import should_task_be_marked_complete, validate_task_completion_with_evidence
    VALIDATION_AVAILABLE = True
except ImportError:
    try:
        from unified_validation_system import should_task_be_marked_complete, validate_task_completion_with_evidence
        VALIDATION_AVAILABLE = True
    except ImportError:
        logger.warning("⚠️ Task completion validation not available - using basic confidence")
        VALIDATION_AVAILABLE = False


class TaskStatus(Enum):
    """Task status with confidence requirements"""
    PENDING = "pending"                    # Not started
    IN_PROGRESS = "in_progress"           # Currently working
    NEEDS_VALIDATION = "needs_validation" # Claimed complete but needs evidence
    LIKELY_COMPLETE = "likely_complete"   # 60-75% confidence of completion
    COMPLETE = "complete"                 # >75% confidence with strong evidence
    VERIFIED = "verified"                 # >85% confidence with external verification
    CANCELLED = "cancelled"               # No longer needed


@dataclass
class TaskEvidence:
    """Evidence supporting task completion"""
    description: str
    evidence_type: str  # 'test_results', 'functional_verification', etc.
    verification_method: str
    confidence_weight: float
    timestamp: datetime
    details: Dict[str, Any]


@dataclass
class EnhancedTask:
    """Enhanced task with completion confidence tracking"""
    id: str
    content: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    
    # Completion confidence tracking
    completion_confidence: float = 0.0
    completion_evidence: List[TaskEvidence] = None
    completion_validation_result: Optional[Dict[str, Any]] = None
    
    # Metadata
    estimated_effort: Optional[str] = None
    dependencies: List[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.completion_evidence is None:
            self.completion_evidence = []
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
    
    def add_evidence(self, evidence: TaskEvidence):
        """Add evidence for task completion"""
        self.completion_evidence.append(evidence)
        self.updated_at = datetime.now()
        logger.info(f"📋 Added evidence for task {self.id}: {evidence.description}")
    
    def calculate_completion_confidence(self) -> float:
        """Calculate completion confidence based on evidence"""
        if not self.completion_evidence:
            return 0.0
        
        # Simple confidence calculation if validation not available
        if not VALIDATION_AVAILABLE:
            total_weight = sum(e.confidence_weight for e in self.completion_evidence)
            return min(total_weight / 3.0, 0.8)  # Cap at 80% without validation
        
        # Use full validation system
        evidence_dicts = []
        for ev in self.completion_evidence:
            evidence_dicts.append({
                'type': ev.evidence_type,
                'description': ev.description,
                'verification_method': ev.verification_method,
                'confidence_weight': ev.confidence_weight,
                'details': ev.details
            })
        
        try:
            result = validate_task_completion_with_evidence(
                task_id=self.id,
                task_description=self.content,
                evidence=evidence_dicts
            )
            self.completion_validation_result = result
            return result.get('completion_confidence', 0.0)
        except Exception as e:
            logger.error(f"❌ Validation failed for task {self.id}: {e}")
            return 0.0
    
    def should_mark_complete(self) -> Tuple[bool, str]:
        """Determine if task should be marked as complete"""
        confidence = self.calculate_completion_confidence()
        self.completion_confidence = confidence
        
        if confidence >= 0.75:
            return True, f"High confidence ({confidence:.1%}) with strong evidence"
        elif confidence >= 0.60:
            return False, f"Moderate confidence ({confidence:.1%}) - need more evidence"
        else:
            return False, f"Low confidence ({confidence:.1%}) - significant work remaining"
    
    def get_status_with_confidence(self) -> str:
        """Get status description with confidence information"""
        confidence = self.completion_confidence
        
        if self.status == TaskStatus.COMPLETE:
            return f"✅ COMPLETE ({confidence:.1%} confidence)"
        elif self.status == TaskStatus.VERIFIED:
            return f"⭐ VERIFIED ({confidence:.1%} confidence)"
        elif self.status == TaskStatus.LIKELY_COMPLETE:
            return f"🟡 LIKELY COMPLETE ({confidence:.1%} confidence)"
        elif self.status == TaskStatus.NEEDS_VALIDATION:
            return f"⚠️ NEEDS VALIDATION ({confidence:.1%} confidence)"
        elif self.status == TaskStatus.IN_PROGRESS:
            return f"🔄 IN PROGRESS"
        elif self.status == TaskStatus.PENDING:
            return f"📋 PENDING"
        else:
            return f"❌ {self.status.value.upper()}"


class EnhancedTodoSystem:
    """
    Enhanced TODO system with anti-overconfidence principles
    
    NEVER marks tasks as complete without substantial evidence.
    Applies skeptical validation to all completion claims.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("state/tasks/enhanced_todos.json")
        self.tasks: Dict[str, EnhancedTask] = {}
        self.load_tasks()
    
    def create_task(self, 
                   task_id: str, 
                   content: str, 
                   estimated_effort: str = None,
                   dependencies: List[str] = None,
                   tags: List[str] = None) -> EnhancedTask:
        """Create a new task with anti-overconfidence tracking"""
        task = EnhancedTask(
            id=task_id,
            content=content,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            estimated_effort=estimated_effort,
            dependencies=dependencies or [],
            tags=tags or []
        )
        
        self.tasks[task_id] = task
        self.save_tasks()
        
        logger.info(f"📝 Created task: {task_id}")
        return task
    
    def update_task_status(self, 
                          task_id: str, 
                          new_status: TaskStatus,
                          evidence: List[TaskEvidence] = None) -> bool:
        """
        Update task status with anti-overconfidence validation
        
        Returns True if status change was allowed, False if rejected due to insufficient confidence
        """
        if task_id not in self.tasks:
            logger.error(f"❌ Task not found: {task_id}")
            return False
        
        task = self.tasks[task_id]
        old_status = task.status
        
        # Add any provided evidence
        if evidence:
            for ev in evidence:
                task.add_evidence(ev)
        
        # Special validation for completion claims
        if new_status in [TaskStatus.COMPLETE, TaskStatus.VERIFIED]:
            should_complete, reason = task.should_mark_complete()
            
            if not should_complete:
                logger.warning(f"⚠️ COMPLETION REJECTED for task {task_id}: {reason}")
                # Set to needs validation instead
                task.status = TaskStatus.NEEDS_VALIDATION
                task.updated_at = datetime.now()
                self.save_tasks()
                return False
            else:
                logger.info(f"✅ COMPLETION APPROVED for task {task_id}: {reason}")
        
        # Update status
        task.status = new_status
        task.updated_at = datetime.now()
        self.save_tasks()
        
        logger.info(f"🔄 Task {task_id}: {old_status.value} → {new_status.value}")
        return True
    
    def add_task_evidence(self, 
                         task_id: str, 
                         evidence: TaskEvidence) -> bool:
        """Add evidence for task completion"""
        if task_id not in self.tasks:
            logger.error(f"❌ Task not found: {task_id}")
            return False
        
        task = self.tasks[task_id]
        task.add_evidence(evidence)
        
        # Recalculate completion confidence
        confidence = task.calculate_completion_confidence()
        logger.info(f"📊 Task {task_id} completion confidence: {confidence:.1%}")
        
        # Auto-update status based on confidence
        if confidence >= 0.75 and task.status == TaskStatus.NEEDS_VALIDATION:
            task.status = TaskStatus.COMPLETE
            logger.info(f"✅ Auto-promoted task {task_id} to COMPLETE")
        elif confidence >= 0.60 and task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
            task.status = TaskStatus.LIKELY_COMPLETE
            logger.info(f"🟡 Auto-promoted task {task_id} to LIKELY_COMPLETE")
        
        self.save_tasks()
        return True
    
    def get_task_completion_report(self, task_id: str) -> str:
        """Generate detailed completion report for a task"""
        if task_id not in self.tasks:
            return f"❌ Task not found: {task_id}"
        
        task = self.tasks[task_id]
        confidence = task.calculate_completion_confidence()
        
        report = f"""
🔍 TASK COMPLETION REPORT: {task_id}

Task: {task.content}
Status: {task.get_status_with_confidence()}
Completion Confidence: {confidence:.1%}

EVIDENCE PROVIDED ({len(task.completion_evidence)} items):
"""
        
        if task.completion_evidence:
            for i, evidence in enumerate(task.completion_evidence, 1):
                report += f"  {i}. {evidence.description}\n"
                report += f"     Type: {evidence.evidence_type}\n"
                report += f"     Method: {evidence.verification_method}\n"
                report += f"     Weight: {evidence.confidence_weight:.1f}\n"
                report += f"     Time: {evidence.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n"
        else:
            report += "  No evidence provided\n\n"
        
        # Add validation result if available
        if task.completion_validation_result:
            result = task.completion_validation_result
            report += f"VALIDATION ASSESSMENT:\n"
            report += f"  Should mark complete: {'✅ Yes' if result.get('should_mark_complete') else '❌ No'}\n"
            report += f"  Recommendation: {result.get('completion_recommendation', 'N/A')}\n\n"
        
        # Add recommendations
        should_complete, reason = task.should_mark_complete()
        report += f"RECOMMENDATION:\n"
        report += f"  {'✅ Mark as complete' if should_complete else '❌ Do not mark complete'}\n"
        report += f"  Reason: {reason}\n"
        
        if not should_complete:
            report += f"\nTO IMPROVE CONFIDENCE:\n"
            report += f"  • Add test results or functional verification\n"
            report += f"  • Provide external validation\n"
            report += f"  • Document specific metrics achieved\n"
            report += f"  • Include integration verification\n"
        
        return report
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[EnhancedTask]:
        """Get all tasks with a specific status"""
        return [task for task in self.tasks.values() if task.status == status]
    
    def get_incomplete_tasks(self) -> List[EnhancedTask]:
        """Get all tasks that are not confidently complete"""
        incomplete_statuses = [
            TaskStatus.PENDING, 
            TaskStatus.IN_PROGRESS, 
            TaskStatus.NEEDS_VALIDATION,
            TaskStatus.LIKELY_COMPLETE
        ]
        return [task for task in self.tasks.values() if task.status in incomplete_statuses]
    
    def generate_status_summary(self) -> str:
        """Generate summary of all task statuses"""
        status_counts = {}
        total_confidence = 0.0
        completed_tasks = 0
        
        for task in self.tasks.values():
            status = task.status
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
            
            if status in [TaskStatus.COMPLETE, TaskStatus.VERIFIED]:
                total_confidence += task.completion_confidence
                completed_tasks += 1
        
        avg_confidence = total_confidence / completed_tasks if completed_tasks > 0 else 0.0
        
        summary = f"""
📊 ENHANCED TODO SYSTEM STATUS

Total Tasks: {len(self.tasks)}
"""
        
        for status, count in status_counts.items():
            emoji = {
                TaskStatus.PENDING: "📋",
                TaskStatus.IN_PROGRESS: "🔄", 
                TaskStatus.NEEDS_VALIDATION: "⚠️",
                TaskStatus.LIKELY_COMPLETE: "🟡",
                TaskStatus.COMPLETE: "✅",
                TaskStatus.VERIFIED: "⭐",
                TaskStatus.CANCELLED: "❌"
            }.get(status, "❓")
            
            summary += f"{emoji} {status.value.replace('_', ' ').title()}: {count}\n"
        
        if completed_tasks > 0:
            summary += f"\n📈 Average Completion Confidence: {avg_confidence:.1%}"
        
        incomplete = self.get_incomplete_tasks()
        if incomplete:
            summary += f"\n⚠️ {len(incomplete)} tasks still need work or validation"
        
        return summary
    
    def save_tasks(self):
        """Save tasks to storage"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert tasks to serializable format
            tasks_data = {}
            for task_id, task in self.tasks.items():
                task_dict = asdict(task)
                # Convert datetime objects to ISO strings
                task_dict['created_at'] = task.created_at.isoformat()
                task_dict['updated_at'] = task.updated_at.isoformat()
                task_dict['status'] = task.status.value
                
                # Convert evidence timestamps
                for evidence in task_dict['completion_evidence']:
                    evidence['timestamp'] = evidence['timestamp'].isoformat()
                
                tasks_data[task_id] = task_dict
            
            with open(self.storage_path, 'w') as f:
                json.dump(tasks_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to save tasks: {e}")
    
    def load_tasks(self):
        """Load tasks from storage"""
        try:
            if not self.storage_path.exists():
                return
            
            with open(self.storage_path, 'r') as f:
                tasks_data = json.load(f)
            
            for task_id, task_dict in tasks_data.items():
                # Convert ISO strings back to datetime objects
                task_dict['created_at'] = datetime.fromisoformat(task_dict['created_at'])
                task_dict['updated_at'] = datetime.fromisoformat(task_dict['updated_at'])
                task_dict['status'] = TaskStatus(task_dict['status'])
                
                # Convert evidence
                evidence_list = []
                for ev_dict in task_dict['completion_evidence']:
                    evidence = TaskEvidence(
                        description=ev_dict['description'],
                        evidence_type=ev_dict['evidence_type'],
                        verification_method=ev_dict['verification_method'],
                        confidence_weight=ev_dict['confidence_weight'],
                        timestamp=datetime.fromisoformat(ev_dict['timestamp']),
                        details=ev_dict['details']
                    )
                    evidence_list.append(evidence)
                
                task_dict['completion_evidence'] = evidence_list
                
                # Create task object
                task = EnhancedTask(**task_dict)
                self.tasks[task_id] = task
                
            logger.info(f"📂 Loaded {len(self.tasks)} tasks from storage")
            
        except Exception as e:
            logger.error(f"❌ Failed to load tasks: {e}")


# Integration with existing todo_write system
def enhanced_todo_write(todos: List[Dict[str, Any]], merge: bool = True) -> Dict[str, Any]:
    """
    Enhanced todo_write that integrates completion confidence validation
    
    Args:
        todos: List of todo items with id, content, status
        merge: Whether to merge with existing todos
        
    Returns:
        Result with completion confidence analysis
    """
    todo_system = EnhancedTodoSystem()
    
    results = {
        'todos_processed': 0,
        'completion_rejections': 0,
        'confidence_warnings': [],
        'status_changes': []
    }
    
    for todo_item in todos:
        task_id = todo_item['id']
        content = todo_item['content']
        status_str = todo_item['status']
        
        # Convert status string to TaskStatus
        try:
            status = TaskStatus(status_str)
        except ValueError:
            logger.warning(f"⚠️ Unknown status '{status_str}' for task {task_id}")
            status = TaskStatus.PENDING
        
        # Create or update task
        if task_id not in todo_system.tasks:
            todo_system.create_task(task_id, content)
        
        # Try to update status
        old_status = todo_system.tasks[task_id].status
        success = todo_system.update_task_status(task_id, status)
        
        if not success and status in [TaskStatus.COMPLETE, TaskStatus.VERIFIED]:
            results['completion_rejections'] += 1
            results['confidence_warnings'].append(
                f"Task {task_id}: Completion rejected - insufficient evidence"
            )
        
        if old_status != todo_system.tasks[task_id].status:
            results['status_changes'].append(
                f"Task {task_id}: {old_status.value} → {todo_system.tasks[task_id].status.value}"
            )
        
        results['todos_processed'] += 1
    
    return results


if __name__ == "__main__":
    # Example usage
    todo_system = EnhancedTodoSystem()
    
    # Create a test task
    task = todo_system.create_task(
        "test_task_1",
        "Implement user authentication system",
        estimated_effort="2 days",
        tags=["backend", "security"]
    )
    
    print("🔍 TESTING ENHANCED TODO SYSTEM")
    print("="*50)
    
    # Try to mark as complete without evidence (should fail)
    print("\n1. Attempting to mark complete without evidence:")
    success = todo_system.update_task_status("test_task_1", TaskStatus.COMPLETE)
    print(f"   Result: {'✅ Allowed' if success else '❌ Rejected'}")
    print(f"   Status: {task.get_status_with_confidence()}")
    
    # Add some evidence
    print("\n2. Adding evidence:")
    evidence = TaskEvidence(
        description="All authentication tests pass (15/15)",
        evidence_type="test_results",
        verification_method="automated_testing",
        confidence_weight=0.9,
        timestamp=datetime.now(),
        details={"test_count": 15, "pass_rate": 1.0}
    )
    todo_system.add_task_evidence("test_task_1", evidence)
    
    # Add more evidence
    evidence2 = TaskEvidence(
        description="Login/logout functionality verified manually",
        evidence_type="functional_verification",
        verification_method="manual_testing",
        confidence_weight=0.8,
        timestamp=datetime.now(),
        details={"features_tested": ["login", "logout", "session_management"]}
    )
    todo_system.add_task_evidence("test_task_1", evidence2)
    
    # Try to mark complete again
    print("\n3. Attempting to mark complete with evidence:")
    success = todo_system.update_task_status("test_task_1", TaskStatus.COMPLETE)
    print(f"   Result: {'✅ Allowed' if success else '❌ Rejected'}")
    print(f"   Status: {task.get_status_with_confidence()}")
    
    # Generate completion report
    print("\n4. Completion Report:")
    report = todo_system.get_task_completion_report("test_task_1")
    print(report)
    
    # System summary
    print("\n5. System Summary:")
    summary = todo_system.generate_status_summary()
    print(summary)
