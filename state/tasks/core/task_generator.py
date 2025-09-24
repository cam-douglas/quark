"""
Task Generator Module
=====================
Generates task documents from templates and roadmap milestones.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class TaskGenerator:
    """Generates tasks from templates and milestones."""
    
    def __init__(self, template_path: Path, output_dir: Path):
        self.template_path = template_path
        self.output_dir = output_dir
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_from_milestone(self, milestone: Dict, kpis: List[Dict]) -> Optional[Path]:
        """Generate a task file from a roadmap milestone."""
        # Create task ID
        category = milestone["category"].replace("-", "_")
        desc_slug = re.sub(r'[^a-z0-9]+', '_', milestone["description"].lower())[:30]
        task_id = f"{category}_{desc_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load template
        if not self.template_path.exists():
            return None
        
        with open(self.template_path) as f:
            template = f.read()
        
        # Replace template placeholders
        content = self._populate_template(template, milestone, task_id)
        
        # Add KPIs section
        content = self._add_kpi_section(content, milestone, kpis)
        
        # Save task file
        output_path = self.output_dir / f"{task_id}.md"
        with open(output_path, 'w') as f:
            f.write(content)
        
        return output_path
    
    def generate_from_input(self, category: str, task_info: Dict, kpis: List[Dict],
                           dependencies: Dict, risks: List[Dict]) -> Path:
        """Generate task document from user input."""
        # Create task ID
        task_id = f"{category}_{task_info['name'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Build document content
        content = self._build_task_document(category, task_id, task_info, kpis, 
                                           dependencies, risks)
        
        # Save to file
        output_path = self.output_dir / f"{task_id}.md"
        with open(output_path, "w") as f:
            f.write(content)
        
        return output_path
    
    def _populate_template(self, template: str, milestone: Dict, task_id: str) -> str:
        """Populate template with milestone data."""
        replacements = {
            "[TASK_CATEGORY]": milestone["category"].replace("_", " ").title(),
            "[TASK_NAME]": milestone["description"],
            "YYYY-MM-DD": datetime.now().strftime("%Y-%m-%d"),
            "[N]": "1",
            "[A-Z]": "A",
            "[PENDING/IN_PROGRESS/COMPLETED/BLOCKED]": milestone["status"],
            "[Roadmap Reference Link]": "management/rules/roadmap/stage1_embryonic_rules.md",
            "[Critical/High/Medium/Low]": "High",
            "[Justification]": "Roadmap milestone",
            "[parent_task_id]": f"stage1_{milestone['category']}",
            "[unique_task_identifier]": task_id,
            "[Comprehensive description of what this task accomplishes and why it's important to the overall system. Include the biological or technical motivation and expected outcomes.]": 
                f"{milestone['description']}. {milestone.get('details', '')}",
            "[How this task connects to other completed or planned components]": 
                "Integrates with foundation layer morphogen gradients and validation system"
        }
        
        content = template
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        return content
    
    def _add_kpi_section(self, content: str, milestone: Dict, kpis: List[Dict]) -> str:
        """Add KPI section to task content."""
        if milestone.get("kpi_refs"):
            kpi_section = "\n## 🎯 **Referenced KPIs**\n\n"
            for kpi_name in milestone["kpi_refs"].split(","):
                kpi_name = kpi_name.strip()
                # Find matching KPI from parsed list
                for kpi_def in kpis:
                    if kpi_def["name"] in kpi_name:
                        kpi_section += f"- `{kpi_def['name']}`: {kpi_def['target']}\n"
                        break
                else:
                    kpi_section += f"- `{kpi_name}`: [Target to be defined]\n"
            
            # Insert KPI section after main KPIs
            content = content.replace("## 🎯 **Task Breakdown**",
                                    kpi_section + "\n## 🎯 **Task Breakdown**")
        
        return content
    
    def _build_task_document(self, category: str, task_id: str, task_info: Dict,
                            kpis: List[Dict], dependencies: Dict, risks: List[Dict]) -> str:
        """Build complete task document from components."""
        content = f"""# 📋 {category.replace('_', ' ').title()} - {task_info['name']}

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Status**: Phase 1 ▸ Batch A ▸ Step 1.P1 - PENDING
**Priority**: {task_info['priority']}
**Task ID**: `{task_id}`
**Estimated Effort**: {task_info['estimated_effort']}

## 📋 Plan Overview
{task_info['description']}

## 🎯 Key Performance Indicators (KPIs)
"""
        
        for kpi in kpis:
            content += f"- `{kpi['name']}`: {kpi['target']}\n"
            content += f"  - Measurement: {kpi['measurement']}\n"
        
        content += "\n## 🔗 Dependencies & Integration Points\n\n"
        content += "### Upstream Dependencies\n"
        for dep in dependencies["upstream"]:
            content += f"- [ ] {dep}\n"
        
        content += "\n### Downstream Consumers\n"
        for dep in dependencies["downstream"]:
            content += f"- {dep}\n"
        
        content += "\n## ⚠️ Risk Assessment\n\n"
        for risk in risks:
            content += f"- **{risk['risk']}**\n"
            content += f"  - Probability: {risk['probability']} | Impact: {risk['impact']}\n"
            content += f"  - Mitigation: {risk['mitigation']}\n"
        
        return content
