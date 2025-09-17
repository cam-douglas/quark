"""
Task Converter - Convert Quark tasks to Cline tasks

Handles conversion between Quark State System task format and Cline CodingTask
format, with biological constraints and Foundation Layer context.
"""

from typing import Dict, List, Any
from brain.modules.cline_integration.cline_adapter import (
    CodingTask, TaskComplexity, ClineTaskType
)


def convert_quark_task_to_cline_task(quark_task: Dict[str, Any]) -> CodingTask:
    """
    Convert a Quark task dictionary to a Cline CodingTask
    
    Args:
        quark_task: Task from existing task loader
        
    Returns:
        CodingTask object for Cline execution
    """
    title = quark_task.get("title", "")
    description = quark_task.get("description", "")
    
    # Determine task type from content
    task_type = _determine_cline_task_type(title, description)
    
    # Assess complexity
    complexity = _assess_complexity_from_quark_task(quark_task)
    
    # Extract files that might be involved
    files_involved = _extract_files_from_task(quark_task)
    
    # Build enhanced description with Quark context
    enhanced_description = _build_enhanced_description(quark_task)
    
    return CodingTask(
        description=enhanced_description,
        task_type=task_type,
        complexity=complexity,
        files_involved=files_involved,
        biological_constraints=True,  # Always true for Quark tasks
        context={
            "quark_task_id": quark_task.get("id"),
            "phase": quark_task.get("phase"),
            "batch": quark_task.get("batch"),
            "step": quark_task.get("step"),
            "priority": quark_task.get("priority"),
            "source": quark_task.get("source"),
            "foundation_layer": "foundation" in title.lower()
        }
    )


def _determine_cline_task_type(title: str, description: str) -> ClineTaskType:
    """Determine Cline task type from Quark task content"""
    content = (title + " " + description).lower()
    
    if any(term in content for term in ["test", "testing", "validate"]):
        return ClineTaskType.TESTING
    elif any(term in content for term in ["document", "readme", "docs"]):
        return ClineTaskType.DOCUMENTATION
    elif any(term in content for term in ["refactor", "restructure"]):
        return ClineTaskType.REFACTORING
    elif any(term in content for term in ["edit", "modify", "update"]):
        return ClineTaskType.FILE_EDITING
    else:
        return ClineTaskType.CODE_GENERATION


def _assess_complexity_from_quark_task(task: Dict[str, Any]) -> TaskComplexity:
    """Assess task complexity from Quark task properties"""
    title = task.get("title", "").lower()
    description = task.get("description", "").lower()
    priority = task.get("priority", "medium").lower()
    
    # High priority usually means high complexity
    if priority in ["critical", "urgent"]:
        return TaskComplexity.CRITICAL
    
    # Check content for complexity indicators
    content = title + " " + description
    
    if any(term in content for term in ["architecture", "system", "integration", "multi"]):
        return TaskComplexity.COMPLEX
    elif any(term in content for term in ["implement", "create", "develop"]):
        return TaskComplexity.MODERATE
    else:
        return TaskComplexity.SIMPLE


def _extract_files_from_task(task: Dict[str, Any]) -> List[str]:
    """Extract likely file paths from task content"""
    files = []
    content = task.get("title", "") + " " + task.get("description", "")
    
    # Common file patterns for Foundation Layer
    if "morphogen" in content.lower():
        files.extend([
            "brain/modules/morphogen_solver/",
            "brain/modules/morphogen_solver/morphogen_solver.py"
        ])
    
    if "bmp" in content.lower():
        files.append("brain/modules/morphogen_solver/bmp_gradient_system.py")
    
    if any(term in content.lower() for term in ["wnt", "fgf"]):
        files.extend([
            "brain/modules/morphogen_solver/wnt_gradient_system.py",
            "brain/modules/morphogen_solver/fgf_gradient_system.py"
        ])
    
    if "ventricular" in content.lower():
        files.append("brain/modules/morphogen_solver/ventricular_system.py")
    
    if "atlas" in content.lower():
        files.extend([
            "brain/modules/morphogen_solver/atlas_validation.py",
            "brain/modules/morphogen_solver/validation_metrics.py"
        ])
    
    return files


def _build_enhanced_description(task: Dict[str, Any]) -> str:
    """Build enhanced description with Quark context for Cline"""
    base_description = task.get("description", task.get("title", ""))
    
    # Add Quark context
    context_parts = [
        f"Quark Task: {task.get('title', 'Unknown')}",
        f"Phase: {task.get('phase', 'Unknown')} | Batch: {task.get('batch', 'Unknown')} | Step: {task.get('step', 'Unknown')}",
        f"Priority: {task.get('priority', 'medium')}",
        f"Source: {task.get('source', 'unknown')}",
        "",
        "Task Description:",
        base_description,
        "",
        "Quark Brain Architecture Context:",
        "- This task is part of the Foundation Layer development",
        "- Must follow biological constraints and AlphaGenome compliance",
        "- All modules must be <300 lines following architecture rules",
        "- Integration with existing morphogen solver system required",
        "- Maintain 1µm³ spatial resolution and neural tube closure stage compatibility"
    ]
    
    return "\n".join(context_parts)


def get_foundation_layer_file_mapping() -> Dict[str, List[str]]:
    """
    Get mapping of Foundation Layer task types to likely files
    
    Returns:
        Dictionary mapping task keywords to file paths
    """
    return {
        "bmp": [
            "brain/modules/morphogen_solver/bmp_gradient_system.py",
            "brain/modules/morphogen_solver/bmp_dynamics_engine.py",
            "brain/modules/morphogen_solver/bmp_parameters.py"
        ],
        "wnt": [
            "brain/modules/morphogen_solver/wnt_gradient_system.py",
            "brain/modules/morphogen_solver/wnt_dynamics_engine.py"
        ],
        "fgf": [
            "brain/modules/morphogen_solver/fgf_gradient_system.py",
            "brain/modules/morphogen_solver/fgf_dynamics_engine.py"
        ],
        "ventricular": [
            "brain/modules/morphogen_solver/ventricular_system.py",
            "brain/modules/morphogen_solver/voxel_excavation.py",
            "brain/modules/morphogen_solver/csf_modeling.py"
        ],
        "meninges": [
            "brain/modules/morphogen_solver/meninges_scaffold.py",
            "brain/modules/morphogen_solver/dura_mater.py",
            "brain/modules/morphogen_solver/arachnoid_membrane.py"
        ],
        "atlas": [
            "brain/modules/morphogen_solver/atlas_validation.py",
            "brain/modules/morphogen_solver/validation_metrics.py",
            "brain/modules/morphogen_solver/parameter_optimization.py"
        ],
        "ml": [
            "brain/modules/morphogen_solver/diffusion_model.py",
            "brain/modules/morphogen_solver/vit3d_encoder.py",
            "brain/modules/morphogen_solver/gnn_integration.py"
        ]
    }
