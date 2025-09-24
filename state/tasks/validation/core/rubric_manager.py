#!/usr/bin/env python3
"""
Rubric Manager Module
=====================
Manages validation rubrics generation and updates.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class RubricManager:
    """Manage validation rubrics."""
    
    def __init__(self, validation_root: Path):
        self.validation_root = validation_root
        self.templates_dir = validation_root / "templates"
        self.checklists_dir = validation_root / "checklists"
        self.template_file = self.templates_dir / "RUBRIC_TEMPLATE.md"
    
    def generate_rubric(self, checklist_name: str) -> Path:
        """Generate a rubric from template for a checklist."""
        rubric_path = self.templates_dir / f"RUBRIC_{checklist_name}.md"
        
        if not self.template_file.exists():
            print(f"⚠️ Template not found: {self.template_file}")
            return rubric_path
        
        # Read template
        with open(self.template_file) as f:
            template = f.read()
        
        # Customize for specific checklist
        rubric_content = template.replace("[CHECKLIST_NAME]", checklist_name)
        rubric_content = rubric_content.replace("[DATE]", datetime.now().strftime("%Y-%m-%d"))
        rubric_content = rubric_content.replace("[VERSION]", "1.0")
        
        # Extract KPIs from checklist
        kpis = self._extract_kpis_from_checklist(checklist_name)
        
        if kpis:
            kpi_section = "\n### KPI-Specific Criteria\n\n"
            for kpi in kpis:
                kpi_section += f"#### {kpi['name']}\n"
                kpi_section += f"- **Target**: {kpi.get('target', 'TBD')}\n"
                kpi_section += f"- **Pass Criteria**: Value meets or exceeds target\n"
                kpi_section += f"- **Fail Criteria**: Value below target by >5%\n"
                kpi_section += f"- **Abstention**: Insufficient data or test failure\n\n"
            
            # Insert KPI section
            rubric_content = rubric_content.replace(
                "## Scoring Criteria",
                f"## Scoring Criteria\n{kpi_section}"
            )
        
        # Write rubric
        with open(rubric_path, "w") as f:
            f.write(rubric_content)
        
        print(f"✅ Rubric generated: {rubric_path.name}")
        return rubric_path
    
    def _extract_kpis_from_checklist(self, checklist_name: str) -> List[Dict]:
        """Extract KPI information from checklist."""
        checklist_path = self.checklists_dir / f"{checklist_name}.md"
        
        if not checklist_path.exists():
            return []
        
        kpis = []
        with open(checklist_path) as f:
            content = f.read()
        
        # Find KPI specifications
        kpi_pattern = r"\*\*KPI:\*\*\s+([^\n]+)"
        target_pattern = r"\*\*Target:\*\*\s+([^\n]+)"
        
        kpi_matches = re.findall(kpi_pattern, content)
        target_matches = re.findall(target_pattern, content)
        
        for i, kpi_name in enumerate(kpi_matches):
            kpi = {"name": kpi_name.strip()}
            if i < len(target_matches):
                kpi["target"] = target_matches[i].strip()
            kpis.append(kpi)
        
        return kpis
    
    def generate_all_rubrics(self) -> List[Path]:
        """Generate rubrics for all checklists."""
        generated = []
        
        for checklist_file in self.checklists_dir.glob("*.md"):
            checklist_name = checklist_file.stem
            rubric_path = self.generate_rubric(checklist_name)
            generated.append(rubric_path)
        
        print(f"\n✅ Generated {len(generated)} rubrics")
        return generated
    
    def update_rubric(self, rubric_path: Path, updates: Dict[str, str]) -> None:
        """Update an existing rubric with new information."""
        if not rubric_path.exists():
            print(f"⚠️ Rubric not found: {rubric_path}")
            return
        
        with open(rubric_path) as f:
            content = f.read()
        
        # Update version
        version_match = re.search(r"Version:\s+([\d.]+)", content)
        if version_match:
            old_version = version_match.group(1)
            new_version = self._increment_version(old_version)
            content = content.replace(
                f"Version: {old_version}",
                f"Version: {new_version}"
            )
        
        # Update date
        content = re.sub(
            r"Date:\s+\d{4}-\d{2}-\d{2}",
            f"Date: {datetime.now().strftime('%Y-%m-%d')}",
            content
        )
        
        # Apply custom updates
        for key, value in updates.items():
            if key in content:
                content = content.replace(key, value)
        
        # Write updated rubric
        with open(rubric_path, "w") as f:
            f.write(content)
        
        print(f"✅ Rubric updated: {rubric_path.name}")
    
    def _increment_version(self, version: str) -> str:
        """Increment version number."""
        parts = version.split(".")
        if len(parts) >= 2:
            minor = int(parts[1]) + 1
            return f"{parts[0]}.{minor}"
        return "1.1"
    
    def validate_rubric_links(self) -> Dict[str, bool]:
        """Validate that all checklists have rubric links."""
        validation = {}
        
        for checklist_file in self.checklists_dir.glob("*.md"):
            checklist_name = checklist_file.stem
            rubric_path = self.templates_dir / f"RUBRIC_{checklist_name}.md"
            
            # Check if rubric exists
            has_rubric = rubric_path.exists()
            
            # Check if checklist links to rubric
            has_link = False
            with open(checklist_file) as f:
                content = f.read()
                if f"RUBRIC_{checklist_name}" in content:
                    has_link = True
            
            validation[checklist_name] = has_rubric and has_link
        
        return validation
    
    def interactive_fill(self, rubric_path: Path) -> None:
        """Interactive rubric filling assistant."""
        print(f"\n📝 Interactive Rubric Filling: {rubric_path.name}")
        print("=" * 50)
        
        if not rubric_path.exists():
            print(f"⚠️ Rubric not found")
            return
        
        with open(rubric_path) as f:
            lines = f.readlines()
        
        updated_lines = []
        in_todo = False
        
        for line in lines:
            if "TODO" in line or "TBD" in line:
                in_todo = True
                print(f"\n🔍 Found: {line.strip()}")
                replacement = input("Enter replacement (or press Enter to skip): ")
                
                if replacement:
                    line = line.replace("TODO", replacement)
                    line = line.replace("TBD", replacement)
                    print(f"✅ Updated")
            
            updated_lines.append(line)
        
        if in_todo:
            save = input("\nSave changes? (y/n): ")
            if save.lower() == "y":
                with open(rubric_path, "w") as f:
                    f.writelines(updated_lines)
                print(f"✅ Rubric saved")
        else:
            print("✅ No TODOs found in rubric")
