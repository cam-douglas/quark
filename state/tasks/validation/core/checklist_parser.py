#!/usr/bin/env python3
"""
Checklist Parser Module
=======================
Parses and updates validation checklists.
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class ChecklistParser:
    """Parse and manipulate validation checklists."""
    
    def __init__(self, validation_root: Path):
        self.validation_root = validation_root
        self.checklists_dir = validation_root / "checklists"
        self.master_checklist = validation_root / "MASTER_VALIDATION_CHECKLIST.md"
    
    def parse_checklist(self, checklist_path: Path) -> Dict[str, Any]:
        """Parse a checklist file into structured data."""
        if not checklist_path.exists():
            return {"error": f"Checklist not found: {checklist_path}"}
        
        with open(checklist_path) as f:
            content = f.read()
            lines = content.split("\n")
        
        parsed = {
            "name": checklist_path.stem,
            "dependencies": [],
            "milestone_gates": [],
            "kpi_specifications": [],
            "completion_status": {}
        }
        
        # Parse dependencies
        parsed["dependencies"] = self._extract_dependencies(lines)
        
        # Parse milestone gates
        parsed["milestone_gates"] = self._extract_milestone_gates(lines)
        
        # Parse KPI specifications
        parsed["kpi_specifications"] = self._extract_kpis(content)
        
        # Calculate completion
        parsed["completion_status"] = self._calculate_completion(parsed)
        
        return parsed
    
    def _extract_dependencies(self, lines: List[str]) -> List[str]:
        """Extract dependencies from checklist lines."""
        deps = []
        in_deps = False
        
        for line in lines:
            if "## Dependencies" in line:
                in_deps = True
            elif in_deps and line.startswith("##"):
                break
            elif in_deps and "- [" in line:
                # Extract linked text
                match = re.search(r"\[([^\]]+)\]", line)
                if match:
                    deps.append(match.group(1))
        
        return deps
    
    def _extract_milestone_gates(self, lines: List[str]) -> List[Dict]:
        """Extract milestone gates from checklist."""
        gates = []
        in_gates = False
        
        for line in lines:
            if "### Milestone Gates" in line:
                in_gates = True
            elif in_gates and line.startswith("##") and "Milestone" not in line:
                break
            elif in_gates and line.strip().startswith("- ["):
                # Parse checkbox item
                checked = "[x]" in line or "[X]" in line
                text = line.split("]", 1)[1].strip() if "]" in line else ""
                
                gates.append({
                    "checked": checked,
                    "description": text
                })
        
        return gates
    
    def _extract_kpis(self, content: str) -> List[Dict]:
        """Extract KPI specifications from content."""
        kpis = []
        
        # Find all KPI blocks
        kpi_pattern = r"\*\*KPI:\*\*\s+([^\n]+).*?\*\*Target:\*\*\s+([^\n]+).*?\*\*Benchmark:\*\*\s+([^\n]+)"
        
        matches = re.finditer(kpi_pattern, content, re.DOTALL)
        
        for match in matches:
            kpi = {
                "name": match.group(1).strip(),
                "target": match.group(2).strip(),
                "benchmark": match.group(3).strip()
            }
            
            # Try to extract measurement and rubric
            block_end = content.find("\n\n", match.end())
            if block_end == -1:
                block_end = len(content)
            
            block = content[match.start():block_end]
            
            measurement_match = re.search(r"\*\*Measurement:\*\*\s+([^\n]+)", block)
            if measurement_match:
                kpi["measurement"] = measurement_match.group(1).strip()
            
            rubric_match = re.search(r"\*\*Rubric:\*\*\s+\[([^\]]+)\]", block)
            if rubric_match:
                kpi["rubric"] = rubric_match.group(1).strip()
            
            evidence_match = re.search(r"\*\*Evidence:\*\*\s+([^\n]+)", block)
            if evidence_match:
                kpi["evidence"] = evidence_match.group(1).strip()
            
            kpis.append(kpi)
        
        return kpis
    
    def _calculate_completion(self, parsed: Dict) -> Dict[str, Any]:
        """Calculate completion status."""
        total_gates = len(parsed["milestone_gates"])
        checked_gates = sum(1 for g in parsed["milestone_gates"] if g["checked"])
        
        total_kpis = len(parsed["kpi_specifications"])
        kpis_with_evidence = sum(1 for k in parsed["kpi_specifications"] if "evidence" in k)
        
        return {
            "gates_complete": f"{checked_gates}/{total_gates}",
            "gates_percentage": (checked_gates / total_gates * 100) if total_gates > 0 else 0,
            "kpis_with_evidence": f"{kpis_with_evidence}/{total_kpis}",
            "kpis_percentage": (kpis_with_evidence / total_kpis * 100) if total_kpis > 0 else 0
        }
    
    def update_checkbox(self, checklist_path: Path, gate_description: str, checked: bool) -> bool:
        """Update a checkbox in a checklist."""
        if not checklist_path.exists():
            return False
        
        with open(checklist_path) as f:
            lines = f.readlines()
        
        updated = False
        for i, line in enumerate(lines):
            if gate_description in line and "- [" in line:
                if checked:
                    lines[i] = line.replace("- [ ]", "- [x]")
                else:
                    lines[i] = line.replace("- [x]", "- [ ]").replace("- [X]", "- [ ]")
                updated = True
                break
        
        if updated:
            with open(checklist_path, "w") as f:
                f.writelines(lines)
        
        return updated
    
    def add_evidence_path(self, checklist_path: Path, kpi_name: str, evidence_path: str) -> bool:
        """Add evidence path to a KPI."""
        if not checklist_path.exists():
            return False
        
        with open(checklist_path) as f:
            content = f.read()
        
        # Find the KPI block
        kpi_pattern = rf"(\*\*KPI:\*\*\s+{re.escape(kpi_name)}.*?)(\n\n|$)"
        match = re.search(kpi_pattern, content, re.DOTALL)
        
        if match:
            kpi_block = match.group(1)
            
            # Check if evidence already exists
            if "**Evidence:**" in kpi_block:
                # Update existing evidence
                kpi_block = re.sub(
                    r"\*\*Evidence:\*\*\s+[^\n]+",
                    f"**Evidence:** {evidence_path}",
                    kpi_block
                )
            else:
                # Add evidence line
                kpi_block += f"\n**Evidence:** {evidence_path}"
            
            # Replace in content
            content = content[:match.start()] + kpi_block + content[match.end(1):]
            
            with open(checklist_path, "w") as f:
                f.write(content)
            
            return True
        
        return False
    
    def generate_completion_report(self) -> str:
        """Generate a completion report for all checklists."""
        report = []
        report.append("\n📋 VALIDATION COMPLETION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        total_gates = 0
        completed_gates = 0
        total_kpis = 0
        evidenced_kpis = 0
        
        for checklist_file in sorted(self.checklists_dir.glob("*.md")):
            parsed = self.parse_checklist(checklist_file)
            
            if "error" not in parsed:
                report.append(f"\n📄 {parsed['name']}")
                report.append("-" * 40)
                
                status = parsed["completion_status"]
                report.append(f"  Gates: {status['gates_complete']} ({status['gates_percentage']:.0f}%)")
                report.append(f"  KPIs with evidence: {status['kpis_with_evidence']} ({status['kpis_percentage']:.0f}%)")
                
                # Update totals
                gates = parsed["milestone_gates"]
                total_gates += len(gates)
                completed_gates += sum(1 for g in gates if g["checked"])
                
                kpis = parsed["kpi_specifications"]
                total_kpis += len(kpis)
                evidenced_kpis += sum(1 for k in kpis if "evidence" in k)
        
        # Add summary
        report.append("\n" + "=" * 50)
        report.append("OVERALL SUMMARY")
        report.append("=" * 50)
        
        gates_pct = (completed_gates / total_gates * 100) if total_gates > 0 else 0
        kpis_pct = (evidenced_kpis / total_kpis * 100) if total_kpis > 0 else 0
        
        report.append(f"Total Milestone Gates: {completed_gates}/{total_gates} ({gates_pct:.0f}%)")
        report.append(f"Total KPIs with Evidence: {evidenced_kpis}/{total_kpis} ({kpis_pct:.0f}%)")
        
        if gates_pct >= 80 and kpis_pct >= 80:
            report.append("\n✅ VALIDATION READY FOR REVIEW")
        elif gates_pct >= 50 and kpis_pct >= 50:
            report.append("\n🔶 VALIDATION IN PROGRESS")
        else:
            report.append("\n⚠️ VALIDATION NEEDS ATTENTION")
        
        return "\n".join(report)
    
    def validate_links(self) -> List[Tuple[str, str, bool]]:
        """Validate all links in checklists."""
        invalid_links = []
        
        for checklist_file in self.checklists_dir.glob("*.md"):
            with open(checklist_file) as f:
                content = f.read()
            
            # Find all markdown links
            links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
            
            for link_text, link_path in links:
                # Resolve relative paths
                if not link_path.startswith(("http", "https")):
                    if link_path.startswith("/"):
                        full_path = Path(link_path)
                    else:
                        full_path = checklist_file.parent / link_path
                    
                    # Check if target exists
                    exists = full_path.exists()
                    invalid_links.append((str(checklist_file), link_path, exists))
        
        return invalid_links

from datetime import datetime
