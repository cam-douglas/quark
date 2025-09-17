#!/usr/bin/env python3
"""YAML Generator Module - Auto-generate YAML files with categorized in-progress tasks.

Handles automatic generation of YAML task files with proper categorization and sub-headings.

Integration: Called whenever tasks are requested to ensure consistent pipeline.
Rationale: Centralized YAML generation with categorization for better task organization.
"""

from pathlib import Path
from typing import List, Dict
import yaml
from datetime import datetime
from collections import defaultdict

def auto_generate_in_progress_yaml(tasks: List[Dict]) -> str:
    """Auto-generate YAML file with all in-progress tasks, organized by category."""

    if not tasks:
        return "No in-progress tasks found."

    # Group tasks by category with sub-headings
    categorized_tasks = defaultdict(list)

    for task in tasks:
        category = task.get('category', 'General')
        section = task.get('section', 'Uncategorized')
        source = task.get('source', 'Unknown')

        # Create hierarchical category key
        category_key = f"{source} - {section} - {category}"
        categorized_tasks[category_key].append(task)

    # Generate YAML structure with categorization
    yaml_structure = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": len(tasks),
        "categories": {}
    }

    # Add tasks organized by category
    for category_key, category_tasks in categorized_tasks.items():
        yaml_structure["categories"][category_key] = {
            "count": len(category_tasks),
            "tasks": []
        }

        for task in category_tasks:
            task_entry = {
                "id": task.get("id", "unknown"),
                "title": task.get("title", "Untitled Task"),
                "status": task.get("status", "pending"),
                "phase": task.get("phase", 1),
                "batch": task.get("batch", "A"),
                "step": task.get("step", 1),
                "priority": task.get("priority", "medium"),
                "description": task.get("description", ""),
                "source": task.get("source", ""),
                "section": task.get("section", ""),
                "category": task.get("category", "")
            }
            yaml_structure["categories"][category_key]["tasks"].append(task_entry)

    # Write to in-progress tasks file
    yaml_file_path = Path("state/quark_state_system/tasks/in-progress_tasks.yaml")
    yaml_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_structure, f, sort_keys=False, default_flow_style=False, indent=2)

    return f"✅ Generated YAML file with {len(tasks)} tasks in {len(categorized_tasks)} categories at {yaml_file_path}"

def format_tasks_with_subheadings(tasks: List[Dict]) -> str:
    """Format tasks with proper sub-headings for display, completely dynamic from YAML data."""

    # Load the YAML file to get the complete task structure
    yaml_file_path = Path("state/quark_state_system/tasks/in-progress_tasks.yaml")

    if not yaml_file_path.exists():
        return "📋 No YAML task file found. Run task generation first."

    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        return f"❌ Error reading YAML file: {e}"

    if not yaml_data or 'tasks_by_roadmap' not in yaml_data:
        return "📋 No task data found in YAML file."

    output_lines = []
    total_tasks = yaml_data.get('total_tasks', 0)

    output_lines.append(f"📊 **Total Tasks Found:** {total_tasks}")

    # Process each roadmap file dynamically
    for roadmap_file, sections in yaml_data['tasks_by_roadmap'].items():
        output_lines.append(f"\n## 🗂️ {roadmap_file.replace('_', ' ').replace('.md', '').title()}")

        # Process each section dynamically (no hardcoded section names)
        for section_name, section_data in sections.items():
            # Use the original header if available, otherwise use section name
            display_name = section_data.get('original_header', section_name)
            if display_name.startswith('**') and display_name.endswith('**'):
                display_name = display_name[2:-2]  # Remove markdown bold

            output_lines.append(f"\n### 🎯 {display_name}")

            section_tasks = section_data.get('tasks', [])

            # Group tasks by category within each section
            categories = defaultdict(list)
            for task in section_tasks:
                category = task.get('category', 'general').replace('[', '').replace(']', '')
                categories[category].append(task)

            # Display tasks by category
            for category, category_tasks in categories.items():
                if len(categories) > 1:  # Only show category header if multiple categories
                    output_lines.append(f"\n#### 📂 {category}")

                for i, task in enumerate(category_tasks, 1):
                    phase = task.get('phase', 1)
                    batch = task.get('batch', 'A')
                    step = task.get('step', 1)
                    status = task.get('status', 'pending')

                    status_icon = "🔄" if status == "in-progress" else "⏳" if status == "pending" else "✅"

                    # Clean up title display (remove prefixes dynamically)
                    title = task.get('title', 'Untitled')
                    # Remove any stage prefixes
                    if ' ▶ ' in title:
                        title = title.split(' ▶ ', 1)[1]
                    # Remove category brackets
                    if title.startswith('[[') and ']] ' in title:
                        title = title.split(']] ', 1)[1]

                    output_lines.append(
                        f"   {i}. {status_icon} **{title}**"
                        f" [Phase {phase}.{batch}.{step}] ({status})"
                    )

    return "\n".join(output_lines)

# --- NEW CODE: sprint-structured formatter ---

def format_tasks_by_sprint() -> str:
    """Return tasks grouped under Phase ▸ Batch ▸ Step headers, obeying Cursor sprint rules."""
    yaml_file_path = Path("state/quark_state_system/tasks/in-progress_tasks.yaml")
    if not yaml_file_path.exists():
        return "📋 No YAML task file found."
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        return f"❌ Error reading YAML file: {e}"
    tasks = []
    for sections in yaml_data.get("tasks_by_roadmap", {}).values():
        for section in sections.values():
            tasks.extend(section.get("tasks", []))

    if not tasks:
        return "📋 No tasks found."

    # sort tasks by phase, batch, step
    def _key(t):
        return (t.get("phase", 1), t.get("batch", "A"), t.get("step", 1))
    tasks.sort(key=_key)

    output = []
    current_header = (None, None, None)
    for t in tasks:
        phase, batch, step = t.get("phase", 1), t.get("batch", "A"), t.get("step", 1)
        header = (phase, batch, step)
        if header != current_header:
            output.append(f"\n### Phase {phase} ▸ Batch {batch} ▸ Step {step}")
            current_header = header
        # include roadmap section/category context
        section = t.get("section", "").strip()
        category = t.get("category", "").replace("[", "").replace("]", "").strip()
        context = f"[{section}]" if section else ""
        if category:
            context += f" ({category})" if context else category
        title = t.get("title", "Untitled")
        if ' ▶ ' in title:
            title = title.split(' ▶ ', 1)[1]
        if title.startswith('[[') and ']] ' in title:
            title = title.split(']] ', 1)[1]
        output.append(f" • {title} {context}")

    return "\n".join(output)

def ensure_yaml_consistency():
    """Ensure YAML file is always generated when tasks are requested."""
    from .task_loader import get_tasks, generate_tasks_from_active_roadmaps

    # Always regenerate tasks from roadmaps first
    generate_tasks_from_active_roadmaps()

    # Get all in-progress tasks
    in_progress_tasks = get_tasks(status="in-progress")

    # Auto-generate YAML file
    result = auto_generate_in_progress_yaml(in_progress_tasks)

    return result, in_progress_tasks
