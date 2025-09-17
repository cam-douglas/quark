#!/usr/bin/env python3
"""Sprint Management Module - Phase/Batch/Step organization following cursor rules.

Implements the sprint-batch-task-management structure with P/F/A/O workflow mapping.

Integration: Provides structured task organization for QuarkDriver and AutonomousAgent.
Rationale: Follows cursor rules for Phase → Batch → Step → P/F/A/O micro-cycle organization.
"""

from typing import Dict, List
import re

def extract_phase_from_roadmap(stage_name: str) -> int:
    """Extract phase number from roadmap stage name."""
    # Extract stage number from names like "stage1_embryonic_rules"
    match = re.search(r'stage(\d+)', stage_name.lower())
    if match:
        return int(match.group(1))
    return 1  # Default to phase 1

def assign_batch_letter(section_name: str) -> str:
    """Assign batch letter based on section type following P/F/A/O model."""
    section_lower = section_name.lower()

    if any(keyword in section_lower for keyword in ["milestone", "goal", "objective", "requirement"]):
        return "A"  # Planning/Discovery phase
    elif any(keyword in section_lower for keyword in ["engineering", "implementation", "development", "technical"]):
        return "B"  # Implementation/Forge phase
    elif any(keyword in section_lower for keyword in ["sota", "practice", "method", "approach"]):
        return "C"  # Assure/Validation phase
    elif any(keyword in section_lower for keyword in ["deliverable", "acceptance", "criteria"]):
        return "D"  # Operate/Deploy phase
    else:
        return "A"  # Default to planning

def assign_pfo_stage(section_name: str, category: str) -> str:
    """Assign P/F/A/O stage based on section and category."""
    section_lower = section_name.lower()
    category_lower = category.lower()

    # Probe stages (P1-P4) - Discovery and planning
    if any(keyword in section_lower for keyword in ["goal", "objective", "requirement"]):
        if any(keyword in category_lower for keyword in ["foundation", "core", "basic"]):
            return "P1"  # Discover
        elif any(keyword in category_lower for keyword in ["research", "analysis"]):
            return "P2"  # Research
        elif any(keyword in category_lower for keyword in ["definition", "scope"]):
            return "P3"  # Define
        else:
            return "P4"  # Reflect

    # Forge stages (F1-F4) - Design and implementation
    elif any(keyword in section_lower for keyword in ["engineering", "implementation", "development"]):
        if any(keyword in category_lower for keyword in ["design", "architecture"]):
            return "F1"  # Design
        elif any(keyword in category_lower for keyword in ["develop", "implement", "build"]):
            return "F2"  # Develop
        elif any(keyword in category_lower for keyword in ["integrate", "refine"]):
            return "F3"  # Integrate
        else:
            return "F4"  # Demo

    # Assure stages (A1-A4) - Testing and validation
    elif any(keyword in section_lower for keyword in ["sota", "practice", "method"]):
        return "A2"  # Test/Practice

    # Operate stages (O1-O4) - Deploy and monitor
    elif any(keyword in section_lower for keyword in ["deliverable", "acceptance"]):
        return "O1"  # Prepare

    # Default based on category
    if any(keyword in category_lower for keyword in ["foundation", "core"]):
        return "P1"  # Discover
    elif any(keyword in category_lower for keyword in ["development", "implementation"]):
        return "F2"  # Develop
    else:
        return "P2"  # Research

def add_sprint_structure_to_task(task: Dict, task_index: int) -> Dict:
    """Add sprint-batch-task-management structure to a task."""
    stage_name = task.get("source", "")
    section_name = task.get("section", "")
    category = task.get("category", "")

    # Calculate sprint structure
    phase_num = extract_phase_from_roadmap(stage_name)
    batch_letter = assign_batch_letter(section_name)
    step_num = (task_index % 5) + 1  # Keep steps 1-5 per batch
    pfo_stage = assign_pfo_stage(section_name, category)

    # Add sprint structure to task
    task.update({
        "phase": phase_num,
        "batch": batch_letter,
        "step": step_num,
        "pfo_stage": pfo_stage,
        "formatted_label": f"Phase {phase_num} ▸ Batch {batch_letter} ▸ Step {step_num}.{pfo_stage}"
    })

    return task

def organize_tasks_by_sprint_structure(tasks: List[Dict]) -> Dict:
    """Organize tasks by Phase → Batch → Step structure."""
    organized = {}

    for task in tasks:
        phase = task.get("phase", 1)
        batch = task.get("batch", "A")

        if phase not in organized:
            organized[phase] = {}
        if batch not in organized[phase]:
            organized[phase][batch] = []

        organized[phase][batch].append(task)

    return organized

def format_task_for_display(task: Dict) -> str:
    """Format task according to sprint-batch-task-management rules."""
    formatted_label = task.get("formatted_label", "")
    title = task.get("title", "No title")
    pfo_stage = task.get("pfo_stage", "")
    status = task.get("status", "pending")

    # Map status to cursor rule format
    status_map = {
        "pending": "To-Do",
        "in_progress": "In-Progress",
        "completed": "Done",
        "review": "Review"
    }

    cursor_status = status_map.get(status, status)

    return f"{formatted_label} [{cursor_status}] {title}"

def get_current_phase_summary(tasks: List[Dict]) -> str:
    """Get summary of current phase/batch/step status."""
    if not tasks:
        return "No active tasks"

    # Group by phase and batch
    by_phase = {}
    for task in tasks:
        phase = task.get("phase", 1)
        batch = task.get("batch", "A")
        status = task.get("status", "pending")

        if phase not in by_phase:
            by_phase[phase] = {}
        if batch not in by_phase[phase]:
            by_phase[phase][batch] = {"total": 0, "completed": 0, "in_progress": 0}

        by_phase[phase][batch]["total"] += 1
        if status == "completed":
            by_phase[phase][batch]["completed"] += 1
        elif status == "in_progress":
            by_phase[phase][batch]["in_progress"] += 1

    # Format summary
    summary_lines = []
    for phase, batches in sorted(by_phase.items()):
        for batch, stats in sorted(batches.items()):
            total = stats["total"]
            completed = stats["completed"]
            in_progress = stats["in_progress"]

            if completed == total:
                status = "completed"
            elif in_progress > 0:
                status = "in progress"
            else:
                status = "to-do"

            summary_lines.append(f"Phase {phase} ▸ Batch {batch}: {completed}/{total} steps completed - {status}")

    return "\n".join(summary_lines)
