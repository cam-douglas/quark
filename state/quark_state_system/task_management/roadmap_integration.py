#!/usr/bin/env python3
"""Roadmap Integration Module - Extract tasks from active roadmap files.

Handles scanning roadmap files for in-progress tasks and converting them to structured task data.

Integration: Feeds roadmap tasks into QuarkDriver and AutonomousAgent execution pipeline.
Rationale: Dynamic task extraction from live roadmap files ensures current priorities.
"""

from pathlib import Path
from typing import List, Dict
import re

def extract_tasks_from_active_roadmaps() -> List[Dict]:
    """Extract all tasks from roadmap files marked as 'In Progress'."""
    roadmap_dir = Path("management/rules/roadmap")
    all_tasks = []
    task_counter = 0

    print("🔍 Scanning ALL roadmap files for 'In Progress' status...")

    # Load Appendix A for context when working with stage roadmaps
    appendix_a_context = _load_appendix_a_context(roadmap_dir)

    # Find ALL roadmap files that are marked as "In Progress" (excluding archives)
    for roadmap_file in roadmap_dir.glob("*.md"):
        # Skip any files in archive directories
        if "archive" in str(roadmap_file).lower() or "backup" in str(roadmap_file).lower():
            continue

        try:
            content = roadmap_file.read_text(encoding='utf-8')

            # Check for any variation of "In Progress" status
            in_progress_patterns = [
                "📋 In Progress",
                "In Progress",
                "IN PROGRESS",
                "🚀 In Progress",
                "Status:** In Progress",
                "Status:** 📋 In Progress"
            ]

            is_in_progress = any(pattern in content for pattern in in_progress_patterns)

            if not is_in_progress:
                continue  # Skip non-active roadmaps

            print(f"   ✅ Found active roadmap: {roadmap_file.name}")
            stage_name = roadmap_file.stem  # e.g., "stage1_embryonic_rules"

            # Check if this is a stage roadmap (stage1-6) to include Appendix A context
            is_stage_roadmap = any(f"stage{i}" in stage_name for i in range(1, 7))

            # Extract tasks from known task-containing sections
            tasks_from_file = _extract_tasks_from_roadmap_content(
                content, stage_name, roadmap_file.name, task_counter, appendix_a_context if is_stage_roadmap else None
            )

            all_tasks.extend(tasks_from_file)
            task_counter += len(tasks_from_file)

        except Exception as e:
            print(f"Warning: Could not read {roadmap_file}: {e}")
            continue

    print(f"📋 Total tasks extracted from in-progress roadmaps: {len(all_tasks)}")
    return all_tasks

def _load_appendix_a_context(roadmap_dir: Path) -> Dict:
    """Load Appendix A content for context when processing stage roadmaps."""
    appendix_a_file = roadmap_dir / "appendix_a_rules.md"

    if not appendix_a_file.exists():
        return {}

    try:
        # Only load first 100 lines to avoid freeze on large files
        with open(appendix_a_file, 'r', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 100:  # Limit to first 100 lines
                    break
                lines.append(line)
            content = ''.join(lines)

        print("   📖 Loading Appendix A context (first 100 lines) for stage roadmap processing...")

        # Simplified context without complex parsing
        context = {
            "content": content[:2000],  # Further limit content size
            "functional_domains": [],  # Skip complex domain extraction to avoid freeze
            "file_path": str(appendix_a_file)
        }

        return context

    except Exception as e:
        print(f"Warning: Could not load Appendix A: {e}")
        return {}

def _extract_functional_domains(appendix_content: str) -> List[Dict]:
    """Extract functional domain specifications from Appendix A."""
    domains = []
    lines = appendix_content.splitlines()

    current_domain = None
    current_section = None

    for line in lines:
        line = line.strip()

        # Look for domain headers (like "Core Cognitive Domains")
        if line and not line.startswith("#") and not line.startswith(">") and not line.startswith("*"):
            if any(keyword in line.lower() for keyword in ["cognitive", "perception", "action", "domain"]):
                if current_domain:
                    domains.append(current_domain)

                current_domain = {
                    "name": line,
                    "sections": [],
                    "specifications": []
                }

        # Look for YAML specifications
        elif line.startswith("```yaml") and current_domain:
            current_section = "yaml_spec"
        elif line == "```" and current_section == "yaml_spec":
            current_section = None
        elif current_section == "yaml_spec" and current_domain:
            current_domain["specifications"].append(line)

    if current_domain:
        domains.append(current_domain)

    return domains

def _extract_tasks_from_roadmap_content(content: str, stage_name: str, roadmap_filename: str, start_counter: int, appendix_a_context: Dict = None) -> List[Dict]:
    """Extract tasks from a single roadmap file content with optional Appendix A context."""
    tasks = []

    # Focus on known task-containing sections with flexible matching
    known_task_sections = [
        "Engineering Milestones", "Biological Goals", "SOTA ML Practices",
        "Implementation Tasks", "Technical Requirements", "Development Tasks",
        "Acceptance Criteria", "Deliverables", "Key Milestones",
        "Functional Mapping"  # Add functional mapping sections
    ]

    lines = content.splitlines()

    for section_name in known_task_sections:
        # Find the section header with flexible matching
        section_start = -1
        original_header = ""

        # Look for the section in the content with partial matching
        for line in lines:
            if f"**{section_name}" in line:
                section_start = content.find(line)
                original_header = line.strip()
                break

        if section_start == -1:
            continue

        # Find the end of this section - look for next major section header
        search_start = section_start + len(original_header) + 10
        next_section_markers = [
            "**Engineering Milestones", "**Biological Goals", "**SOTA ML Practices",
            "**Acceptance Criteria", "**Implementation", "**Stage", "**Appendix"
        ]

        section_end = len(content)  # Default to end of file
        for marker in next_section_markers:
            if marker != original_header[:len(marker)]:  # Don't match the current section
                next_pos = content.find(marker, search_start)
                if next_pos != -1 and next_pos < section_end:
                    section_end = next_pos

        section_content = content[section_start:section_end]

        # Extract task lines with line-based matching
        section_lines = section_content.splitlines()

        for line in section_lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a task line
            if line.startswith("* [") and "]" in line:
                # Extract category and description
                bracket_end = line.find("]")
                if bracket_end > 2:
                    category = line[2:bracket_end].strip()
                    task_desc = line[bracket_end + 1:].strip()

                    # Clean up task description (remove KPI markers, DONE tags, etc.)
                    clean_desc = re.sub(r'\s*\*\*\(KPI:.*?\)\*\*', '', task_desc).strip()
                    clean_desc = re.sub(r'\s*DONE\s*$', '', clean_desc).strip()

                    # Skip if already marked as DONE
                    if "DONE" in task_desc:
                        continue

                    # Skip if too short or looks like header
                    if len(clean_desc.strip()) < 10 or "**" in clean_desc:
                        continue

                    task_id = f"{stage_name}_{category}_{start_counter + len(tasks)}_{hash(clean_desc) % 10000}"

                    task = {
                        "id": task_id,
                        "title": f"[{category}] {clean_desc}",
                        "status": "pending",
                        "source": f"{stage_name}",
                        "category": category,
                        "section": section_name,
                        "section_subtitle": f"{section_name} — {stage_name.replace('_', ' ').title()}",
                        "original_header": original_header,
                        "roadmap_file": roadmap_filename
                    }

                    tasks.append(task)

    # Add functional mapping tasks at the bottom for stage roadmaps
    if appendix_a_context and any(f"stage{i}" in stage_name for i in range(1, 7)):
        functional_mapping_tasks = _extract_functional_mapping_tasks(
            content, stage_name, roadmap_filename, appendix_a_context, len(tasks)
        )
        tasks.extend(functional_mapping_tasks)

    return tasks

def _extract_functional_mapping_tasks(content: str, stage_name: str, roadmap_filename: str, appendix_a_context: Dict, task_offset: int) -> List[Dict]:
    """Extract functional mapping tasks and integrate with Appendix A context."""
    functional_tasks = []

    # Look for Functional Mapping section
    functional_section_start = content.find("**Functional Mapping")
    if functional_section_start == -1:
        return functional_tasks

    print("      🔗 Processing Functional Mapping section with Appendix A context...")

    # Extract the functional mapping content
    section_end = content.find("---", functional_section_start)
    if section_end == -1:
        section_end = len(content)

    functional_content = content[functional_section_start:section_end]
    section_lines = functional_content.splitlines()

    # Extract links to Appendix A sections
    for line in section_lines:
        line = line.strip()
        if line.startswith("- [") and "#$cap-" in line:
            # Parse functional mapping link
            # Format: "- [Core Cognitive — Memory, Learning, Reasoning, Problem Solving](#$cap-1-core-cognitive)"
            link_match = re.search(r'- \[([^\]]+)\]\(#\$cap-(\d+)-([^)]+)\)', line)
            if link_match:
                domain_description = link_match.group(1)
                cap_number = link_match.group(2)
                cap_id = link_match.group(3)

                # Create functional mapping task with Appendix A context
                task_id = f"{stage_name}_functional_mapping_{cap_number}_{hash(domain_description) % 10000}"

                # Get relevant context from Appendix A
                appendix_context = _get_appendix_context_for_capability(appendix_a_context, cap_id)

                task = {
                    "id": task_id,
                    "title": f"[functional-mapping] {domain_description}",
                    "status": "pending",
                    "source": f"{stage_name}",
                    "category": "functional-mapping",
                    "section": "Functional Mapping",
                    "section_subtitle": f"Functional Mapping — {stage_name.replace('_', ' ').title()}",
                    "original_header": "**Functional Mapping (links to Appendix A):**",
                    "roadmap_file": roadmap_filename,
                    "appendix_link": f"#$cap-{cap_number}-{cap_id}",
                    "appendix_context": appendix_context,
                    "capability_number": cap_number,
                    "capability_id": cap_id
                }

                functional_tasks.append(task)

    if functional_tasks:
        print(f"         ✅ Added {len(functional_tasks)} functional mapping tasks with Appendix A context")

    return functional_tasks

def _get_appendix_context_for_capability(appendix_a_context: Dict, cap_id: str) -> Dict:
    """Get relevant context from Appendix A for a specific capability."""
    if not appendix_a_context:
        return {}

    content = appendix_a_context.get("content", "")

    # Find the section referenced by the capability ID
    cap_section_start = content.find('<a id="$cap-')
    if cap_section_start == -1:
        return {}

    # Extract a reasonable amount of context (next 500 chars)
    context_end = min(len(content), cap_section_start + 1000)
    context_section = content[cap_section_start:context_end]

    return {
        "context_snippet": context_section[:500],
        "full_appendix_available": True,
        "appendix_file": appendix_a_context.get("file_path", "")
    }

def update_roadmap_with_done_tag(completed_task: dict):
    """Update the roadmap file to add DONE tag to completed task."""
    from pathlib import Path

    source = completed_task.get("source", "")
    if not source.endswith("_rules"):
        return

    roadmap_file = Path("management/rules/roadmap") / f"{source}.md"
    if not roadmap_file.exists():
        print(f"⚠️  Roadmap file not found: {roadmap_file}")
        return

    try:
        content = roadmap_file.read_text(encoding='utf-8')

        # Find the task line and add DONE tag
        task_title = completed_task.get("title", "")
        category = completed_task.get("category", "")

        # Extract the description part (remove [category] prefix)
        if task_title.startswith(f"[{category}]"):
            task_desc = task_title[len(f"[{category}]"):].strip()
        else:
            task_desc = task_title

        # Look for the task line and add DONE if not already present
        lines = content.splitlines()
        updated = False

        for i, line in enumerate(lines):
            if f"[{category}]" in line and task_desc[:30] in line:  # Match first 30 chars
                if "DONE" not in line:
                    lines[i] = line.rstrip() + " DONE"
                    updated = True
                    break

        if updated:
            roadmap_file.write_text("\n".join(lines), encoding='utf-8')
            print(f"✅ Updated roadmap with DONE tag: {roadmap_file.name}")
        else:
            print(f"⚠️  Could not find task in roadmap to mark as DONE: {task_desc[:50]}...")

    except Exception as e:
        print(f"❌ Error updating roadmap file: {e}")
